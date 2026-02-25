from typing import Dict, List, Optional, Union, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import base64
import json
import re


@dataclass
class InstructionPage:
    """说明书的单一页面，包含文字指引或截图"""
    page_num: int
    content_type: str  # "text" 或 "screenshot"
    content: Union[str, bytes]  # 文字内容或图像字节
    description: Optional[str] = None  # 页面描述
    
    def is_text(self) -> bool:
        """检查是否为文字页面"""
        return self.content_type == "text"
    
    def is_screenshot(self) -> bool:
        """检查是否为截图页面"""
        return self.content_type == "screenshot"
    
    def get_text(self) -> str:
        """获取文字内容，如果是截图则返回 base64"""
        if self.is_text():
            return self.content
        else:
            return base64.b64encode(self.content).decode('utf-8')


class Instruction:
    """
    表示一个软件的使用说明书，包含使用步骤的文字指引与相应步骤截图。
    存储用页码表示，每页仅包含文字指引或截图之一。
    并包括mcp工具用于让llm agent阅读说明书。
    
    Attributes:
        title: 说明书标题
        software_name: 软件名称
        version: 说明书版本
        pages: 按顺序存储的说明书页面
    """
    
    def __init__(self, title: str, software_name: str, version: str = "1.0"):
        """初始化说明书
        
        Args:
            title: 说明书标题
            software_name: 软件名称
            version: 说明书版本
        """
        self.title = title
        self.software_name = software_name
        self.version = version
        self.pages: List[InstructionPage] = []
    
    def add_text_page(self, content: str, description: Optional[str] = None) -> None:
        """添加文字页面
        
        Args:
            content: 文字内容
            description: 页面描述
        """
        page_num = len(self.pages) + 1
        page = InstructionPage(
            page_num=page_num,
            content_type="text",
            content=content,
            description=description
        )
        self.pages.append(page)
    
    def add_screenshot_page(
        self, 
        screenshot_bytes: bytes, 
        description: Optional[str] = None
    ) -> None:
        """添加截图页面
        
        Args:
            screenshot_bytes: 截图的字节数据（PNG/JPG 格式）
            description: 页面描述
        """
        page_num = len(self.pages) + 1
        page = InstructionPage(
            page_num=page_num,
            content_type="screenshot",
            content=screenshot_bytes,
            description=description
        )
        self.pages.append(page)
    
    def add_screenshot_page_from_file(
        self, 
        file_path: Union[str, Path], 
        description: Optional[str] = None
    ) -> None:
        """从文件添加截图页面
        
        Args:
            file_path: 截图文件路径
            description: 页面描述
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Screenshot file not found: {file_path}")
        
        with open(file_path, 'rb') as f:
            screenshot_bytes = f.read()
        
        self.add_screenshot_page(screenshot_bytes, description)
    
    def get_page(self, page_num: int) -> Optional[InstructionPage]:
        """获取指定页面
        
        Args:
            page_num: 页面号（1-indexed）
            
        Returns:
            指定的页面，如果不存在则返回 None
        """
        if 1 <= page_num <= len(self.pages):
            return self.pages[page_num - 1]
        return None
    
    def get_total_pages(self) -> int:
        """获取说明书的总页数"""
        return len(self.pages)
    
    def get_text_pages(self) -> List[InstructionPage]:
        """获取所有文字页面"""
        return [page for page in self.pages if page.is_text()]
    
    def get_screenshot_pages(self) -> List[InstructionPage]:
        """获取所有截图页面"""
        return [page for page in self.pages if page.is_screenshot()]
    
    def read_instruction(self, page_num: Optional[int] = None) -> str:
        """MCP 工具：读取说明书
        
        这是一个供 LLM agent 调用的工具，用于阅读说明书的指定页面或所有页面。
        
        Args:
            page_num: 要读取的页面号（1-indexed），如果为 None 则读取所有页面
            
        Returns:
            说明书内容的字符串表示
        """
        if page_num is None:
            # 返回所有页面的摘要
            result = []
            result.append(f"# {self.title}")
            result.append(f"软件：{self.software_name}")
            result.append(f"版本：{self.version}")
            result.append(f"总页数：{self.get_total_pages()}\n")
            
            for page in self.pages:
                result.append(f"## 第 {page.page_num} 页")
                if page.description:
                    result.append(f"描述：{page.description}")
                
                if page.is_text():
                    result.append(page.content)
                else:
                    result.append(f"[截图页面 - 大小: {len(page.content)} 字节]")
                result.append("")
            
            return "\n".join(result)
        else:
            # 返回指定页面
            page = self.get_page(page_num)
            if page is None:
                return f"错误：页面 {page_num} 不存在。总页数为 {self.get_total_pages()}。"
            
            result = []
            result.append(f"# {self.title} - 第 {page_num} 页")
            
            if page.description:
                result.append(f"描述：{page.description}")
            
            if page.is_text():
                result.append("\n内容：")
                result.append(page.content)
            else:
                # 返回截图的 base64 编码
                img_b64 = base64.b64encode(page.content).decode('utf-8')
                result.append(f"\n[截图页面]")
                result.append(f"图像数据（Base64）：{img_b64[:100]}...")  # 显示前 100 字符
                result.append(f"完整数据可通过 get_screenshot_page 方法获取")
            
            return "\n".join(result)
    
    def get_screenshot_page(self, page_num: int) -> Optional[bytes]:
        """MCP 工具：获取截图页面的原始字节
        
        Args:
            page_num: 截图页面号（1-indexed）
            
        Returns:
            截图的字节数据，如果页面不存在或不是截图则返回 None
        """
        page = self.get_page(page_num)
        if page and page.is_screenshot():
            return page.content
        return None
    
    def search_pages(self, keyword: str) -> List[InstructionPage]:
        """MCP 工具：搜索说明书中包含关键字的页面
        
        Args:
            keyword: 搜索关键字
            
        Returns:
            包含关键字的页面列表
        """
        results = []
        for page in self.pages:
            if page.is_text():
                if keyword.lower() in page.content.lower():
                    results.append(page)
            elif page.description and keyword.lower() in page.description.lower():
                results.append(page)
        
        return results
    
    def to_dict(self) -> Dict:
        """将说明书转换为字典格式
        
        Returns:
            说明书的字典表示
        """
        return {
            "title": self.title,
            "software_name": self.software_name,
            "version": self.version,
            "total_pages": self.get_total_pages(),
            "pages": [
                {
                    "page_num": page.page_num,
                    "content_type": page.content_type,
                    "description": page.description,
                    "content_preview": (
                        (page.content[:100] + "..." if len(page.content) > 100 else page.content)
                        if page.is_text()
                        else f"[Screenshot - {len(page.content)} bytes]"
                    )
                }
                for page in self.pages
            ]
        }
    
    def to_json(self) -> str:
        """将说明书转换为 JSON 字符串"""
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)
    
    @classmethod
    def from_dict(cls, data: Dict) -> "Instruction":
        """从字典创建 Instruction 实例
        
        Args:
            data: 包含说明书信息的字典
            
        Returns:
            新的 Instruction 实例
        """
        instruction = cls(
            title=data.get("title", ""),
            software_name=data.get("software_name", ""),
            version=data.get("version", "1.0")
        )
        return instruction
    
    @classmethod
    def from_markdown_file(cls, markdown_file: Union[str, Path]) -> "Instruction":
        """从 Markdown 文件创建 Instruction 实例
        
        Markdown 文件格式：
        ```
        # 标题
        软件名：XXX
        版本：1.0
        
        ## 第一步
        这是第一步的说明文字。
        
        ![步骤截图](./image1.png)
        
        ## 第二步
        这是第二步的说明文字。
        
        ![步骤截图](./image2.png)
        ```
        
        Args:
            markdown_file: Markdown 文件路径
            
        Returns:
            新的 Instruction 实例
            
        Raises:
            FileNotFoundError: 如果 Markdown 文件不存在
        """
        markdown_file = Path(markdown_file)
        if not markdown_file.exists():
            raise FileNotFoundError(f"Markdown file not found: {markdown_file}")
        
        with open(markdown_file, 'r', encoding='utf-8') as f:
            markdown_content = f.read()
        
        # 获取文件所在目录，用于解析相对路径
        base_dir = markdown_file.parent
        
        return cls.from_markdown(markdown_content, base_dir=base_dir)
    
    @classmethod
    def from_markdown(cls, markdown_content: str, base_dir: Optional[Union[str, Path]] = None) -> "Instruction":
        """从 Markdown 字符串创建 Instruction 实例
        
        Markdown 格式规范：
        - 第一行的 # 标题为说明书标题
        - 识别 "软件名:" 或 "软件名：" 开头的行作为软件名
        - 识别 "版本:" 或 "版本：" 开头的行作为版本
        - ## 或更低级别的标题用于区分不同步骤
        - ![描述](路径) 格式的图片被识别为截图
        - 其他文字内容作为说明文字
        
        Args:
            markdown_content: Markdown 格式的字符串
            base_dir: 基础目录，用于解析图片的相对路径。如果为 None，则使用当前工作目录
            
        Returns:
            新的 Instruction 实例
        """
        if base_dir is None:
            base_dir = Path.cwd()
        else:
            base_dir = Path(base_dir)
        
        # 初始化变量
        title = "未命名说明书"
        software_name = "未指定"
        version = "1.0"
        
        # 逐行解析 Markdown
        lines = markdown_content.split('\n')
        
        # 提取标题、软件名、版本
        for i, line in enumerate(lines):
            if line.startswith('# ') and not title.startswith("# "):
                title = line[2:].strip()
            elif re.match(r'^软件名[:：]', line):
                software_name = re.sub(r'^软件名[:：]\s*', '', line).strip()
            elif re.match(r'^版本[:：]', line):
                version = re.sub(r'^版本[:：]\s*', '', line).strip()
        
        # 创建 Instruction 实例
        instruction = cls(title=title, software_name=software_name, version=version)
        
        # 将内容按步骤分块
        current_text = []
        i = 0
        while i < len(lines):
            line = lines[i]
            
            # 检查是否为图片行
            img_match = re.match(r'!\[([^\]]*)\]\(([^)]+)\)', line)
            if img_match:
                # 保存之前累积的文字
                text_content = '\n'.join(current_text).strip()
                if text_content:
                    instruction.add_text_page(text_content)
                    current_text = []
                
                # 处理图片
                img_desc = img_match.group(1)
                img_path = img_match.group(2)
                
                try:
                    full_img_path = base_dir / img_path
                    if full_img_path.exists():
                        instruction.add_screenshot_page_from_file(full_img_path, description=img_desc)
                    else:
                        # 如果文件不存在，记录警告但继续
                        print(f"警告：图片文件不存在：{full_img_path}")
                except Exception as e:
                    print(f"警告：无法加载图片 {img_path}：{e}")
            
            elif line.startswith('#'):
                # 跳过标题行（但不是 # 开头的标题）
                if not re.match(r'^软件名|^版本', line):
                    # 保存之前的文字
                    text_content = '\n'.join(current_text).strip()
                    if text_content:
                        instruction.add_text_page(text_content)
                        current_text = []
                    
                    # 添加标题作为新文字页面
                    if line.startswith('## '):
                        section_title = line[3:].strip()
                    elif line.startswith('# '):
                        section_title = line[2:].strip()
                    else:
                        section_title = line.lstrip('#').strip()
                    
                    if section_title and section_title != title:
                        current_text.append(section_title)
            else:
                # 普通文字行
                if line.strip() or current_text:  # 避免开头的空行
                    current_text.append(line)
            
            i += 1
        
        # 保存最后的文字内容
        text_content = '\n'.join(current_text).strip()
        if text_content:
            instruction.add_text_page(text_content)
        
        return instruction