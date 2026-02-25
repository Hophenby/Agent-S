"""
说明书阅读和步骤匹配Agent

此模块提供一个Agent，能够：
1. 读取说明书（Instruction实例）
2. 比较当前observation与说明书上的步骤
3. 选择与当前状况最相近的说明书图片
"""

from functools import partial
import logging
import base64
import textwrap
from typing import Dict, List, Optional, Tuple
from io import BytesIO
from PIL import Image

from core.mllm import LMMAgent
from utils.patch_locator import locate_patch
from utils.common_utils import call_llm_formatted, create_pyautogui_code, parse_code_from_string
from memory.procedural_memory import PROCEDURAL_MEMORY
from utils.formatters import CODE_VALID_FORMATTER, SINGLE_ACTION_FORMATTER
from agents.instruction import Instruction, InstructionPage

logger = logging.getLogger("desktopenv.agent")


class InstructionReader:
    """
    说明书阅读Agent
    
    能够加载说明书，比较当前的observation与说明书上的步骤，
    并选择最相近的说明书截图。
    """
    llm_client: Optional[LMMAgent] = None
    reader_tools: Optional['ReaderTools'] = None
    
    def __init__(self, engine_params: Dict = None, platform: str = "windows", llm_client: LMMAgent = None, temperature: float = 0.0, use_thinking: bool = True):
        """初始化说明书阅读Agent
        
        Args:
            engine_params: 引擎参数（可选）
            platform: 操作系统平台
        """
        self.engine_params = engine_params or {}
        self.platform = platform
        self.instruction: Optional[Instruction] = None
        self.turn_count = 0
        self.llm_client = llm_client
        self.temperature = temperature
        self.use_thinking = use_thinking
        llm_client.add_system_prompt(PROCEDURAL_MEMORY.construct_simple_worker_procedural_memory(
            ReaderTools, 
            skipped_actions=[], 
            guidelines=SYSTEM_PROMPT, 
            formatting_instructions=RESPONSE_FORMAT_PROMPT
        ).replace("CURRENT_OS", platform))
        self.reader_tools = ReaderTools(self)
        
    def load_instruction(self, instruction: Instruction) -> None:
        """加载说明书
        
        Args:
            instruction: Instruction实例
        """
        self.instruction = instruction
        self.reader_tools.instruction = instruction
        if self.llm_client:
            for page in instruction.pages:
                page_desc = page.description or "无描述"
                if page.is_text():
                    content = f"第 {page.page_num} 页（文字）：{page_desc}\n{page.content}"
                    self.llm_client.add_message(text_content=content, role="user")
                else:
                    content = f"第 {page.page_num} 页（截图）：{page_desc}"
                    self.llm_client.add_message(
                        text_content=content,
                        image_content=page.content,
                        role="user",
                    )
        logger.info(
            f"成功加载说明书：{instruction.title} "
            f"（{instruction.software_name} v{instruction.version}）"
        )
        logger.info(f"总页数：{instruction.get_total_pages()}")
    
    def load_instruction_from_markdown(self, markdown_file: str) -> None:
        """从Markdown文件加载说明书
        
        Args:
            markdown_file: Markdown文件路径
        """
        instruction = Instruction.from_markdown_file(markdown_file)
        self.load_instruction(instruction)
    
    def get_instruction_summary(self) -> str:
        """获取说明书摘要
        
        Returns:
            说明书的文字描述
        """
        if self.instruction is None:
            return "没有加载说明书"
        
        return self.instruction.read_instruction()
    
    def get_instruction_text_pages(self) -> List[InstructionPage]:
        """获取所有说明书中的文字页面
        
        Returns:
            文字页面列表
        """
        if self.instruction is None:
            return []
        
        return self.instruction.get_text_pages()
    
    def get_instruction_screenshot_pages(self) -> List[InstructionPage]:
        """获取所有说明书中的截图页面
        
        Returns:
            截图页面列表
        """
        if self.instruction is None:
            return []
        
        return self.instruction.get_screenshot_pages()
    
    def _screenshot_to_base64(self, screenshot_bytes: bytes) -> str:
        """将截图字节转换为Base64编码
        
        Args:
            screenshot_bytes: 截图字节数据
            
        Returns:
            Base64编码的字符串
        """
        return base64.b64encode(screenshot_bytes).decode('utf-8')
    
    def find_matching_instruction_page(
        self,
        current_screenshot: bytes,
        task_instruction: str,
        top_k: int = 3
    ) -> List[Tuple[InstructionPage, float, str]]:
        """找到与当前状况最匹配的说明书页面
        
        使用LLM比较当前截图与说明书中的每一张截图，
        找到最相近的几张，并返回相似度分数。
        
        Args:
            current_screenshot: 当前的截图字节数据
            task_instruction: 任务指令
            top_k: 返回前k个最匹配的页面
            
        Returns:
            列表，每个元素为 (页面, 相似度分数0-1, 理由)
            
        Raises:
            ValueError: 如果没有加载说明书或说明书中没有截图
        """
        if self.instruction is None:
            raise ValueError("没有加载说明书")
        
        instruction_screenshots = self.get_instruction_screenshot_pages()
        if not instruction_screenshots:
            raise ValueError("说明书中没有截图页面")
        
        # 将当前截图转换为Base64
        current_b64 = self._screenshot_to_base64(current_screenshot)
        
        # 为每张说明书截图计算相似度
        similarities: List[Tuple[InstructionPage, float, str]] = []
        
        for inst_page in instruction_screenshots:
            if self.llm_client:
                # 如果提供了LLM客户端，使用智能比较
                score, reason = self._compare_with_llm(
                    current_b64=current_b64,
                    instruction_page=inst_page,
                    task_instruction=task_instruction
                )
            else:
                # 否则使用基础的启发式比较
                score, reason = self._compare_heuristic(
                    current_screenshot=current_screenshot,
                    instruction_page=inst_page,
                    task_instruction=task_instruction
                )
            
            similarities.append((inst_page, score, reason))
            
            logger.debug(
                f"比较第{inst_page.page_num}页：相似度={score:.2f}，理由={reason}"
            )
        
        # 按相似度降序排序并返回前k个
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
    
    def get_matching_pages_with_context(
        self,
        current_screenshot: bytes,
        task_instruction: str,
        top_k: int = 3
    ) -> Dict:
        """获取匹配的页面及其上下文信息
        
        Args:
            current_screenshot: 当前截图
            task_instruction: 任务指令
            top_k: 返回前k个匹配
            
        Returns:
            包含详细信息的字典
        """
        try:
            matches = self.find_matching_instruction_page(
                current_screenshot,
                task_instruction,
                top_k=top_k
            )
            
            result = {
                "instruction_title": self.instruction.title if self.instruction else "",
                "instruction_software": self.instruction.software_name if self.instruction else "",
                "current_turn": self.turn_count,
                "matches": []
            }
            
            for i, (page, score, reason) in enumerate(matches):
                # 获取相邻页面的信息，提供更多上下文
                context = {
                    "rank": i + 1,
                    "page_num": page.page_num,
                    "similarity_score": score,
                    "reason": reason,
                    "description": page.description
                }
                
                # 添加前一页的文字信息（如果存在）
                prev_page = self.instruction.get_page(page.page_num - 1) if self.instruction else None
                if prev_page and prev_page.is_text():
                    context["previous_step_text"] = prev_page.content[:200]
                
                # 添加后一页的文字信息（如果存在）
                next_page = self.instruction.get_page(page.page_num + 1) if self.instruction else None
                if next_page and next_page.is_text():
                    context["next_step_text"] = next_page.content[:200]
                
                result["matches"].append(context)
            
            self.turn_count += 1
            return result
        
        except Exception as e:
            logger.error(f"获取匹配页面及其上下文时出错：{e}")
            return {
                "error": str(e),
                "matches": []
            }

    def run_generation(
        self,
        observation: Dict,
        instruction: str,
        generator_message: str
    ) -> Tuple[Dict, List]:
        """运行说明书阅读与匹配
        Args:
            observation: 当前的UI状态观察
            instruction: 任务指令
            generator_message: 以前的操作消息
        """
    
        self.llm_client.add_message(
            text_content=current_step_message(instruction, generator_message),
            image_content=observation["screenshot"],
            role="user",
        )


        # Generate the plan and next action
        format_checkers = [
            SINGLE_ACTION_FORMATTER,
            partial(CODE_VALID_FORMATTER, self.reader_tools, None),
        ]

        max_attempts = 10
        while max_attempts > 0:
            response = call_llm_formatted(
                self.llm_client,
                format_checkers,
                temperature=self.temperature,
                use_thinking=self.use_thinking,
            )
            self.llm_client.add_message(text_content=response, role="assistant")
            logger.info(f"LLM Response:\n{response}")
            plan_code = parse_code_from_string(response)
            result = create_pyautogui_code(self.reader_tools, plan_code, None)
            if (isinstance(result, dict)):
                if ("matched" in result):
                    return result, [plan_code]
                self.llm_client.add_message(
                    text_content=f"错误：匹配结果失败，返回内容：{result}",
                    role="assistant",
                )
            else:
                self.llm_client.add_message(
                    text_content=f"{result}",
                    role="assistant",
                )
            max_attempts -= 1

        return {
            "success": False,
            "error": "无法生成有效的匹配结果",
        }, []
    

# Agent action decorator
def agent_action(func):
    func.is_agent_action = True
    return func

class ReaderTools:
    """说明书阅读工具集"""
    
    instruction_reader: InstructionReader
    
    def __init__(self, instruction_reader: InstructionReader):
        self.instruction_reader = instruction_reader
        self.instruction = instruction_reader.instruction
    @agent_action
    def read_instruction_summary(self) -> str:
        """MCP 工具：读取说明书摘要
        
        供 LLM 调用的工具，用于获取说明书的完整概览。
        
        Returns:
            说明书的摘要信息，包括标题、软件名、版本和所有页面的简要描述
        """
        if self.instruction is None:
            return "错误：没有加载说明书。请先加载说明书。"
        
        return self.instruction.read_instruction()

    @agent_action
    def read_page(self, page_num: int) -> str:
        """MCP 工具：读取指定页面的详细内容
        
        供 LLM 调用的工具，用于读取说明书的特定页面。
        
        Args:
            page_num: 要读取的页码（从 1 开始）
            
        Returns:
            页面的详细内容，包括文字说明或截图信息
        """
        if self.instruction is None:
            return "错误：没有加载说明书。"
        
        return self.instruction.read_instruction(page_num)

    @agent_action
    def list_screenshot_pages(self) -> str:
        """MCP 工具：列出所有截图页面
        
        供 LLM 调用的工具，列出说明书中所有的截图页面及其描述。
        
        Returns:
            所有截图页面的列表，格式化为易读的字符串
        """
        if self.instruction is None:
            return "错误：没有加载说明书。"
        
        screenshot_pages = self.instruction_reader.get_instruction_screenshot_pages()
        
        if not screenshot_pages:
            return "说明书中没有截图页面。"
        
        result = [f"说明书共有 {len(screenshot_pages)} 张截图页面：\n"]
        
        for page in screenshot_pages:
            desc = page.description if page.description else "无描述"
            result.append(f"  第 {page.page_num} 页：{desc}")
        
        return "\n".join(result)
    

    @agent_action
    def search_pages(self, keyword: str) -> str:
        """MCP 工具：搜索包含关键字的页面
        
        供 LLM 调用的工具，在说明书中搜索包含特定关键字的页面。
        
        Args:
            keyword: 要搜索的关键字
            
        Returns:
            包含关键字的页面列表，格式化为易读的字符串
        """
        if self.instruction is None:
            return "错误：没有加载说明书。"
        
        pages = self.instruction.search_pages(keyword)
        
        if not pages:
            return f"没有找到包含关键字 '{keyword}' 的页面。"
        
        result = [f"找到 {len(pages)} 个包含关键字 '{keyword}' 的页面：\n"]
        
        for page in pages:
            page_type = "文字页面" if page.is_text() else "截图页面"
            preview = page.content[:50] if page.is_text() else page.description or "无描述"
            result.append(f"  第 {page.page_num} 页 [{page_type}]：{preview}...")
        
        return "\n".join(result)
    

    @agent_action
    def get_page_context(self, page_num: int) -> str:
        """MCP 工具：获取页面及其上下文
        
        供 LLM 调用的工具，获取指定页面及其前后页面的信息，
        帮助理解当前步骤在整个流程中的位置。
        
        Args:
            page_num: 页码（从 1 开始）
            
        Returns:
            页面及其上下文信息，格式化为易读的字符串
        """
        if self.instruction is None:
            return "错误：没有加载说明书。"
        
        page = self.instruction.get_page(page_num)
        if not page:
            return f"错误：页面 {page_num} 不存在。总页数为 {self.instruction.get_total_pages()}。"
        
        result = [f"=== 第 {page_num} 页 ==="]
        
        # 显示当前页面信息
        page_type = "文字页面" if page.is_text() else "截图页面"
        result.append(f"类型：{page_type}")
        if page.description:
            result.append(f"描述：{page.description}")
        
        if page.is_text():
            result.append(f"\n内容：\n{page.content}")
        else:
            result.append(f"\n[这是一张截图页面，大小：{len(page.content)} 字节]")
        
        # 显示前一页
        if page_num > 1:
            prev_page = self.instruction.get_page(page_num - 1)
            if prev_page:
                result.append(f"\n--- 前一页（第 {page_num - 1} 页）---")
                if prev_page.is_text():
                    preview = prev_page.content[:100]
                    result.append(f"文字预览：{preview}...")
                else:
                    result.append(f"截图描述：{prev_page.description or '无描述'}")
        
        # 显示后一页
        if page_num < self.instruction.get_total_pages():
            next_page = self.instruction.get_page(page_num + 1)
            if next_page:
                result.append(f"\n--- 后一页（第 {page_num + 1} 页）---")
                if next_page.is_text():
                    preview = next_page.content[:100]
                    result.append(f"文字预览：{preview}...")
                else:
                    result.append(f"截图描述：{next_page.description or '无描述'}")
        
        return "\n".join(result)
    

    @agent_action
    def pass_page(self, page_num: int, reason: str) -> Dict[str, any]:
        """MCP 工具：选择并通过一个说明书页面
        
        供 LLM 调用的工具，用于告诉系统当前选择了哪个说明书页面，
        以及选择该页面的理由。选择的页面必须是截图页面。
        
        Args:
            page_num: 选择的页码（从 1 开始）
            reason: 选择该页面的理由
            
        Returns:
            包含结果信息的字典
        """
        if self.instruction is None:
            return {
                "success": False,
                "error": "没有加载说明书",
                "message": "错误：没有加载说明书。无法选择页面。"
            }
        
        page = self.instruction.get_page(page_num)
        
        if not page:
            return {
                "success": False,
                "error": "页面不存在",
                "message": f"错误：页面 {page_num} 不存在。总页数为 {self.instruction.get_total_pages()}。",
                "total_pages": self.instruction.get_total_pages()
            }
        
        if not page.is_screenshot():
            return {
                "success": False,
                "error": "页面类型错误",
                "message": f"错误：第 {page_num} 页不是截图页面。请选择截图页面。",
                "page_num": page_num,
                "page_type": "text"
            }
        
        # 成功选择页面
        logger.info(f"✅ LLM 选择了说明书第 {page_num} 页")
        logger.info(f"   描述：{page.description or '无描述'}")
        logger.info(f"   理由：{reason}")
        
        return {
            "success": True,
            "matched": True,
            "page_num": page_num,
            "description": page.description,
            "reason": reason,
            "message": f"✅ 成功选择第 {page_num} 页（{page.description or '无描述'}）",
            "page_content": page.content  # 返回截图的字节数据
        }
    

    @agent_action
    def fail_to_match_page(self, reason: str) -> Dict[str, any]:
        """MCP 工具：报告无法匹配任何页面
        
        供 LLM 调用的工具，当 LLM 判断当前截图与说明书中的任何页面
        都不相似时，使用此工具报告失败原因。
        
        Args:
            reason: 无法匹配的原因
            
        Returns:
            包含结果信息的字典
        """
        logger.warning(f"⚠️  LLM 报告无法匹配页面")
        logger.warning(f"   理由：{reason}")
        
        return {
            "success": False,
            "matched": False,
            "reason": reason,
            "message": f"⚠️  无法找到匹配的说明书页面。\n理由：{reason}",
            "suggestion": "建议：检查当前操作是否符合说明书流程，或说明书是否完整。"
        }
    

    @agent_action
    def get_instruction_info(self) -> Dict[str, any]:
        """MCP 工具：获取说明书基本信息
        
        供 LLM 调用的工具，快速获取说明书的基本元信息。
        
        Returns:
            包含说明书基本信息的字典
        """
        if self.instruction is None:
            return {
                "loaded": False,
                "message": "没有加载说明书"
            }
        
        screenshot_pages = self.instruction_reader.get_instruction_screenshot_pages()
        text_pages = self.instruction_reader.get_instruction_text_pages()
        
        return {
            "loaded": True,
            "title": self.instruction.title,
            "software_name": self.instruction.software_name,
            "version": self.instruction.version,
            "total_pages": self.instruction.get_total_pages(),
            "text_pages": len(text_pages),
            "screenshot_pages": len(screenshot_pages),
            "message": f"已加载：{self.instruction.title}（{self.instruction.software_name} v{self.instruction.version}）"
        }
    

    



SYSTEM_PROMPT = textwrap.dedent(
    """
    你是一个专业的GUI软件操作专家与python代码编程专家。
    你可以使用软件说明书阅读工具与代码执行工具来帮助你完成任务。
    一开始你将得到目标软件的说明书, 上面记录着如何使用这个软件完成指定任务的步骤。
    你将有一系列的工具帮助你阅读说明书并完成任务。
    之后，用户会给你一些任务指令和当前的屏幕截图。
    你的任务是根据说明书和当前截图，判断当前软件状态与说明书上哪个步骤最相似，并据此通过选择一张指导下一步操作的说明书页面来指导用户完成任务。
    一般来说，指导下一步操作的图像页面上会有当前软件界面某个步骤要求用户点击或输入的位置。比如按钮、菜单项、文本框等。你需要选择这样的页面来指导用户下一步操作。
        # GUIDELINES

        ## 工具使用指南
    你可以使用说明书阅读工具来查找与当前步骤所需操作匹配的说明书页面。
    这些工具的调用均以python函数的形式出现。
    你需要根据当前的任务指令和截图，选择最合适的说明书页面，并据此生成下一步的操作指令。
    需要使用agent.pass_page()工具来向用户传递你选择的说明书页面，并解释你的选择理由。选择的页面必须为图片页面。
    如果当前的截图与说明书上的任何页面都不相似，你需要使用agent.fail_to_match_page()来告诉用户没有找到匹配的页面。
可用的阅读工具与使用方法如下：
""")


RESPONSE_FORMAT_PROMPT = textwrap.dedent(
    """
    ### 响应格式
    你的回复必须严格遵守以下格式：
        (Instruction Analysis)
        当你阅读了说明书并分析了当前任务指令后，描述你对任务的理解和你的计划。
    
        (Screenshot Analysis)
        仔细分析当前截图与说明书页面的相似之处，描述你观察到的关键元素和特征。若用户没有提供截图，请说明无法进行分析。
        
        (Next Action)
        根据你的分析，选择指示下一步操作的说明书页面，并解释你的选择理由。
        如果没有任何页面与你的当前截图匹配，请解释原因并说明你将如何处理这种情况。

        (Grounded Action)
        将你的选择通过调用相应的工具传达给用户。如果你需要进一步阅读说明书页面以做出决定，也请调用相应的工具。所有的工具调用格式必须如下：
        ```python
        agent.get_page_context(2)
        ```
    注意：
    - 你只能使用提供的工具来获取信息和传达你的选择。
    - 一次只能调用一个工具。
    - 每次仅返回一个代码块，而且代码块内必须且只能包含一个工具调用。
    - 确保你的工具调用参数正确无误。
    - 当你确信你找到了最合适的说明书页面时，使用agent.pass_page()来传达你的选择。
    - agent.pass_page()的参数page_num必须是对应一张截图页面。
    - 如果没有任何页面匹配当前截图，使用agent.fail_to_match_page()来报告。
    """
)

current_step_message = lambda instruction, generator_message: textwrap.dedent(
    f"""
    当前任务指令：{instruction}
    以前进行过的操作：{generator_message}
    请根据当前任务指令和以前的操作，以及目前状态的屏幕截图，在说明书中找到最匹配这一步需要操作的按钮或位置的页面。
    """
)
def process_generation_result(observation: Dict, result: Dict) -> Dict:
    """处理生成结果，提取关键信息
    
    Args:
        result: 生成结果字典
        
    Returns:
        处理后的结果字典
    """
    # logger.info(f"Processing generation result: {result}")
    if result.get("page_content", None):
        content_image = Image.open(BytesIO(result["page_content"]))
        obs_image = Image.open(BytesIO(observation["original_screenshot"]))
        match_result = locate_patch(obs_image, content_image, score_threshold=0.5)
        if match_result:
            result["match_box"] = {
                "center_x": match_result.x/obs_image.width + match_result.width/2/obs_image.width,
                "center_y": match_result.y/obs_image.height + match_result.height/2/obs_image.height,
                "width": match_result.width/obs_image.width,
                "height": match_result.height/obs_image.height,
                "score": match_result.score,
            }
            logger.info(f"Located match box: {result['match_box']}")
        else:
            logger.warning("No match box located.")

    return result