"""
测试 MCP 工具功能

此文件测试 InstructionReader 中新增的 MCP 工具方法，
验证它们能够正确地供 LLM 调用。
"""

import sys
import json
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from gui_agents.s3.agents.instruction_reader import InstructionReader


def test_mcp_tools():
    """测试所有 MCP 工具"""
    
    print("=" * 80)
    print("测试 InstructionReader MCP 工具")
    print("=" * 80)
    
    # 创建 InstructionReader 实例
    reader = InstructionReader(
        engine_params={},
        platform="ubuntu",
        llm_client=None  # 不需要 LLM 客户端来测试这些工具
    )
    
    # 测试：在没有加载说明书时调用工具
    print("\n[测试 1] 未加载说明书时的行为")
    print("-" * 80)
    
    result = reader.mcp_get_instruction_info()
    print(f"mcp_get_instruction_info(): {json.dumps(result, indent=2, ensure_ascii=False)}")
    
    summary = reader.mcp_read_instruction_summary()
    print(f"\nmcp_read_instruction_summary(): {summary}")
    
    # 加载示例说明书
    markdown_file = Path(__file__).parent / "example_instruction.md"
    if not markdown_file.exists():
        print(f"\n⚠️  警告：找不到示例文件 {markdown_file}")
        print("跳过后续测试")
        return
    
    print(f"\n[测试 2] 加载说明书：{markdown_file.name}")
    print("-" * 80)
    
    reader.load_instruction_from_markdown(str(markdown_file))
    
    # 测试获取说明书信息
    print("\n[测试 3] 获取说明书基本信息")
    print("-" * 80)
    
    info = reader.mcp_get_instruction_info()
    print(json.dumps(info, indent=2, ensure_ascii=False))
    
    # 测试读取摘要
    print("\n[测试 4] 读取说明书摘要")
    print("-" * 80)
    
    summary = reader.mcp_read_instruction_summary()
    print(summary)
    
    # 测试列出截图页面
    print("\n[测试 5] 列出所有截图页面")
    print("-" * 80)
    
    screenshot_list = reader.mcp_list_screenshot_pages()
    print(screenshot_list)
    
    # 测试读取特定页面
    print("\n[测试 6] 读取特定页面")
    print("-" * 80)
    
    page_content = reader.mcp_read_page(1)
    print(f"第 1 页内容：\n{page_content}")
    
    page_content = reader.mcp_read_page(3)
    print(f"\n第 3 页内容：\n{page_content}")
    
    # 测试获取页面上下文
    print("\n[测试 7] 获取页面上下文")
    print("-" * 80)
    
    context = reader.mcp_get_page_context(3)
    print(context)
    
    # 测试搜索页面
    print("\n[测试 8] 搜索包含关键字的页面")
    print("-" * 80)
    
    search_result = reader.mcp_search_pages("EndNote")
    print(f"搜索 'EndNote':\n{search_result}")
    
    search_result = reader.mcp_search_pages("不存在的关键字xyz")
    print(f"\n搜索 '不存在的关键字xyz':\n{search_result}")
    
    # 测试 pass_page - 成功案例（截图页面）
    print("\n[测试 9] 测试 mcp_pass_page - 成功案例")
    print("-" * 80)
    
    # 假设第 3 页是截图页面
    result = reader.mcp_pass_page(
        page_num=3,
        reason="当前屏幕显示的是 EndNote 主界面，与说明书第 3 页的截图完全一致"
    )
    print(json.dumps({k: v for k, v in result.items() if k != 'screenshot_bytes'}, 
                     indent=2, ensure_ascii=False))
    if result.get('success'):
        print(f"✅ 成功选择页面，截图数据大小：{len(result['screenshot_bytes'])} 字节")
    
    # 测试 pass_page - 错误案例（文字页面）
    print("\n[测试 10] 测试 mcp_pass_page - 错误案例（选择文字页面）")
    print("-" * 80)
    
    result = reader.mcp_pass_page(
        page_num=1,
        reason="尝试选择文字页面（应该失败）"
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    
    # 测试 pass_page - 错误案例（页面不存在）
    print("\n[测试 11] 测试 mcp_pass_page - 错误案例（页面不存在）")
    print("-" * 80)
    
    result = reader.mcp_pass_page(
        page_num=999,
        reason="尝试选择不存在的页面"
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    
    # 测试 fail_to_match_page
    print("\n[测试 12] 测试 mcp_fail_to_match_page")
    print("-" * 80)
    
    result = reader.mcp_fail_to_match_page(
        reason="当前屏幕显示的是一个错误对话框，说明书中没有关于此错误的处理步骤"
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    
    # 测试获取工具 schema
    print("\n[测试 13] 获取 MCP 工具 Schema")
    print("-" * 80)
    
    schema = InstructionReader.get_mcp_tools_schema()
    print(f"共有 {len(schema)} 个 MCP 工具：")
    for tool in schema:
        params = tool['parameters']['properties']
        param_str = ", ".join([f"{k}: {v['type']}" for k, v in params.items()]) if params else "无参数"
        print(f"  - {tool['name']}({param_str})")
        print(f"    {tool['description']}")
    
    # 测试获取工具描述
    print("\n[测试 14] 获取 MCP 工具描述文档")
    print("-" * 80)
    
    description = reader.get_mcp_tools_description()
    print(description)
    
    print("\n" + "=" * 80)
    print("✅ 所有测试完成！")
    print("=" * 80)


def test_schema_export():
    """测试导出工具 schema 为 JSON 文件"""
    
    print("\n[附加测试] 导出工具 Schema 为 JSON 文件")
    print("-" * 80)
    
    schema = InstructionReader.get_mcp_tools_schema()
    
    output_file = Path(__file__).parent / "mcp_tools_schema.json"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(schema, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Schema 已导出到：{output_file}")
    print(f"   文件大小：{output_file.stat().st_size} 字节")


if __name__ == "__main__":
    test_mcp_tools()
    test_schema_export()
