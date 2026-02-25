"""
测试 InstructionReader Agent 的功能

演示：
1. 加载说明书
2. 比较当前observation与说明书步骤
3. 选择最相近的说明书图片
"""

import sys
from pathlib import Path
from io import BytesIO
from PIL import Image

# 添加必要的路径
current_dir = Path(__file__).parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

from instruction import Instruction
from instruction_reader import InstructionReader


def create_mock_screenshot(color: tuple = (200, 220, 240)) -> bytes:
    """创建一个模拟的截图字节数据
    
    Args:
        color: RGB颜色元组
        
    Returns:
        PNG格式的字节数据
    """
    img = Image.new('RGB', (800, 600), color=color)
    buffer = BytesIO()
    img.save(buffer, format='PNG')
    return buffer.getvalue()


def test_instruction_reader():
    """测试 InstructionReader 功能"""
    
    print("=" * 70)
    print("测试 InstructionReader Agent")
    print("=" * 70)
    
    # 步骤1：加载说明书
    print("\n【步骤1】加载说明书...")
    print("-" * 70)
    
    markdown_file = Path(__file__).parent / "example_instruction.md"
    
    if not markdown_file.exists():
        print(f"❌ 说明书文件不存在：{markdown_file}")
        return
    
    try:
        reader = InstructionReader(
            engine_params={
                "model": "gpt-4",
                "temperature": 0.0,
                "engine_type": "openai"
            },
            platform="windows"
        )
        
        reader.load_instruction_from_markdown(str(markdown_file))
        print(f"✅ 成功加载说明书")
        print(f"   标题：{reader.instruction.title}")
        print(f"   软件：{reader.instruction.software_name}")
        print(f"   版本：{reader.instruction.version}")
        print(f"   总页数：{reader.instruction.get_total_pages()}")
        
    except Exception as e:
        print(f"❌ 加载说明书失败：{e}")
        return
    
    # 步骤2：显示说明书中的截图页面
    print("\n【步骤2】显示说明书中的截图页面...")
    print("-" * 70)
    
    screenshot_pages = reader.get_instruction_screenshot_pages()
    print(f"找到 {len(screenshot_pages)} 张截图页面：")
    for page in screenshot_pages:
        print(f"   • 第 {page.page_num} 页：{page.description or '无描述'}")
    
    # 步骤3：创建模拟的当前截图
    print("\n【步骤3】创建模拟的当前截图...")
    print("-" * 70)
    
    # 创建不同的模拟截图（模拟不同的应用状态）
    current_screenshot = create_mock_screenshot(color=(220, 240, 200))
    print(f"✅ 创建了模拟的当前截图（{len(current_screenshot)} 字节）")
    
    # 步骤4：获取说明书摘要
    print("\n【步骤4】获取说明书摘要...")
    print("-" * 70)
    
    summary = reader.get_instruction_summary()
    # 只显示前500个字符
    summary_preview = summary[:500] + "..." if len(summary) > 500 else summary
    print(summary_preview)
    
    # 步骤5：找到最匹配的说明书页面
    print("\n【步骤5】找到最匹配的说明书页面...")
    print("-" * 70)
    
    task_instruction = "按照说明书逐步操作Word应用程序"
    
    try:
        # 获取最佳匹配页面
        best_page, score, reason = reader.get_best_matching_page(
            current_screenshot,
            task_instruction
        )
        
        if best_page:
            print(f"✅ 找到最匹配的页面")
            print(f"   页号：{best_page.page_num}")
            print(f"   相似度：{score:.2%}")
            print(f"   理由：{reason}")
            if best_page.description:
                print(f"   描述：{best_page.description}")
        else:
            print(f"⚠️  没有找到匹配的页面")
        
    except Exception as e:
        print(f"❌ 匹配失败（这是正常的，因为没有配置真实的LLM）：{type(e).__name__}")
        print(f"   注意：此功能需要配置有效的LLM API密钥")
        print(f"   详细错误：{e}")
    
    # 步骤6：获取带上下文的匹配信息
    print("\n【步骤6】获取带上下文的匹配信息...")
    print("-" * 70)
    
    try:
        context_info = reader.get_matching_pages_with_context(
            current_screenshot,
            task_instruction,
            top_k=3
        )
        
        if context_info.get("error"):
            print(f"⚠️  获取上下文失败（这是正常的）：{context_info['error']}")
        else:
            print(f"✅ 获取上下文信息成功")
            print(f"   说明书：{context_info['instruction_title']}")
            print(f"   软件：{context_info['instruction_software']}")
            print(f"   当前轮次：{context_info['current_turn']}")
            print(f"   匹配数量：{len(context_info['matches'])}")
            
            for match in context_info['matches']:
                print(f"\n   【匹配 {match['rank']}】")
                print(f"      页号：{match['page_num']}")
                print(f"      相似度：{match['similarity_score']:.2%}")
                print(f"      理由：{match['reason']}")
                if match.get('previous_step_text'):
                    print(f"      上一步提示：{match['previous_step_text'][:50]}...")
                if match.get('next_step_text'):
                    print(f"      下一步提示：{match['next_step_text'][:50]}...")
    
    except Exception as e:
        print(f"⚠️  获取上下文失败（这是正常的）：{type(e).__name__}")
        print(f"   注意：此功能需要配置有效的LLM API密钥")
    
    # 步骤7：演示API使用示例
    print("\n【步骤7】API 使用示例...")
    print("-" * 70)
    
    print("""
可用的主要方法：

1. load_instruction(instruction)
   - 加载一个 Instruction 实例

2. load_instruction_from_markdown(markdown_file)
   - 从 Markdown 文件加载说明书

3. get_instruction_summary()
   - 获取说明书的完整摘要

4. get_instruction_text_pages()
   - 获取所有文字页面列表

5. get_instruction_screenshot_pages()
   - 获取所有截图页面列表

6. find_matching_instruction_page(current_screenshot, task_instruction, top_k=3)
   - 找到前k个最匹配的说明书页面
   - 返回：[(页面, 相似度分数, 理由), ...]

7. get_best_matching_page(current_screenshot, task_instruction)
   - 获取单个最匹配的页面
   - 返回：(页面, 相似度分数, 理由)

8. get_matching_pages_with_context(current_screenshot, task_instruction, top_k=3)
   - 获取匹配的页面及其上下文信息
   - 返回：包含详细信息的字典
    """)
    
    print("\n" + "=" * 70)
    print("✅ 测试完成！")
    print("=" * 70)


if __name__ == "__main__":
    test_instruction_reader()
