"""测试 Instruction 类的 Markdown 解析功能"""

from instruction import Instruction
from pathlib import Path
from PIL import Image
import io



def test_markdown_parsing():
    """测试 Markdown 解析功能"""
    
    print("=" * 60)
    print("测试 Instruction 类的 Markdown 解析功能")
    print("=" * 60)
    
    # 创建测试图片
    print("\n📁 准备测试图片...")
    
    # 从 Markdown 文件创建 Instruction 实例
    print("\n📖 解析 Markdown 文件...")
    markdown_file = Path(__file__).parent / "example_instruction.md"
    
    try:
        instruction = Instruction.from_markdown_file(markdown_file)
        
        print(f"\n✅ 成功创建说明书实例！")
        print(f"   标题：{instruction.title}")
        print(f"   软件名：{instruction.software_name}")
        print(f"   版本：{instruction.version}")
        print(f"   总页数：{instruction.get_total_pages()}")
        
        # 显示页面摘要
        print(f"\n📄 页面摘要：")
        for i, page in enumerate(instruction.pages, 1):
            if page.is_text():
                preview = page.content[:50].replace('\n', ' ')
                if len(page.content) > 50:
                    preview += "..."
                print(f"   第 {i} 页 [文字]: {preview}")
            else:
                print(f"   第 {i} 页 [截图]: {len(page.content)} 字节")
                if page.description:
                    print(f"            描述: {page.description}")
        
        # 测试 read_instruction 方法（读取所有页面）
        print(f"\n📋 完整说明书内容：")
        print("-" * 60)
        full_content = instruction.read_instruction()
        print(full_content)
        
        # 测试读取单独页面
        print(f"\n📄 读取第 2 页：")
        print("-" * 60)
        page_content = instruction.read_instruction(page_num=2)
        print(page_content)
        
        # 测试搜索功能
        print(f"\n🔍 搜索关键字 '保存'：")
        print("-" * 60)
        search_results = instruction.search_pages("保存")
        print(f"找到 {len(search_results)} 个匹配页面")
        for page in search_results:
            print(f"   - 第 {page.page_num} 页")
        
        # 测试转换为 JSON
        print(f"\n📦 转换为 JSON：")
        print("-" * 60)
        json_str = instruction.to_json()
        print(json_str)
        
        print(f"\n✅ 所有测试通过！")
        
    except FileNotFoundError as e:
        print(f"\n❌ 错误：{e}")
    except Exception as e:
        print(f"\n❌ 发生错误：{e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_markdown_parsing()
