"""
InstructionReader 集成示例

展示如何将 InstructionReader 集成到现有的 Worker Agent 中，
用于比较当前observation与说明书步骤，选择最相近的图片。
"""

from typing import Dict, List, Optional, Tuple
from instruction import Instruction
from instruction_reader import InstructionReader


class WorkerWithInstructionSupport:
    """
    扩展后的 Worker Agent，支持说明书指导
    
    在生成下一个行动时，可以参考说明书中的步骤和截图，
    来更好地理解当前状态并做出更准确的决策。
    """
    
    def __init__(self, worker_engine_params: Dict = None, markdown_instruction_file: Optional[str] = None):
        """初始化
        
        Args:
            worker_engine_params: Worker agent 的参数
            markdown_instruction_file: Markdown 说明书文件路径（可选）
        """
        self.worker_engine_params = worker_engine_params or {}
        self.instruction_reader = InstructionReader(
            engine_params=self.worker_engine_params,
            platform="windows",
        )
        
        # 加载说明书（如果提供）
        if markdown_instruction_file:
            try:
                self.instruction_reader.load_instruction_from_markdown(markdown_instruction_file)
                print(f"✅ 成功加载说明书：{markdown_instruction_file}")
            except Exception as e:
                print(f"⚠️ 无法加载说明书：{e}")
    
    def set_instruction(self, instruction: Instruction) -> None:
        """设置说明书
        
        Args:
            instruction: Instruction 实例
        """
        self.instruction_reader.load_instruction(instruction)
    
    def generate_next_action_with_instruction_context(
        self,
        instruction: str,
        observation: Dict,
        use_instruction_guidance: bool = True
    ) -> Tuple[Dict, List, Optional[Dict]]:
        """生成下一个行动，同时参考说明书指导
        
        Args:
            instruction: 任务指令
            observation: 当前观察（包含 'screenshot' 字段）
            use_instruction_guidance: 是否使用说明书指导
            
        Returns:
            (agent_info, actions, instruction_context)
            其中 instruction_context 包含说明书的匹配信息
        """
        
        # 先执行标准的 worker 逻辑来生成行动
        # 这里是伪代码，实际实现时需要调用真实的 worker.generate_next_action()
        agent_info = {}
        actions = []
        instruction_context = None
        
        # 如果启用说明书指导，则尝试从说明书获取上下文
        if use_instruction_guidance and self.instruction_reader.instruction:
            try:
                # 从 observation 中获取当前截图
                if 'screenshot' in observation and isinstance(observation['screenshot'], bytes):
                    current_screenshot = observation['screenshot']
                    
                    # 获取与当前状态最匹配的说明书页面
                    instruction_context = self.instruction_reader.get_matching_pages_with_context(
                        current_screenshot,
                        task_instruction=instruction,
                        top_k=3
                    )
                    
                    # 可以将说明书的上下文信息添加到 agent_info 中
                    if instruction_context.get('matches'):
                        best_match = instruction_context['matches'][0]
                        agent_info['instruction_guidance'] = {
                            'matched_page': best_match['page_num'],
                            'similarity_score': best_match['similarity_score'],
                            'description': best_match['description'],
                            'guidance': best_match.get('next_step_text', 'No guidance available')
                        }
            
            except Exception as e:
                print(f"⚠️ 获取说明书指导失败：{e}")
        
        return agent_info, actions, instruction_context
    
    def get_instruction_summary(self) -> str:
        """获取当前加载的说明书摘要"""
        return self.instruction_reader.get_instruction_summary()
    
    def get_instruction_guidance(self, current_screenshot: bytes, task_instruction: str) -> Optional[Dict]:
        """获取说明书指导
        
        Args:
            current_screenshot: 当前截图
            task_instruction: 任务指令
            
        Returns:
            包含说明书指导的字典，或 None
        """
        if not self.instruction_reader.instruction:
            return None
        
        try:
            return self.instruction_reader.get_matching_pages_with_context(
                current_screenshot,
                task_instruction,
                top_k=3
            )
        except Exception as e:
            print(f"⚠️ 获取说明书指导失败：{e}")
            return None


def example_usage():
    """使用示例"""
    
    print("=" * 70)
    print("WorkerWithInstructionSupport 使用示例")
    print("=" * 70)
    
    # 创建支持说明书的 Worker
    markdown_file = "example_instruction.md"
    worker = WorkerWithInstructionSupport(
        worker_engine_params={
            "model": "gpt-4",
            "temperature": 0.0,
            "engine_type": "openai"
        },
        markdown_instruction_file=markdown_file
    )
    
    print("\n【说明书摘要】")
    print("-" * 70)
    summary = worker.get_instruction_summary()
    summary_preview = summary[:300] + "..." if len(summary) > 300 else summary
    print(summary_preview)
    
    # 模拟获取观察和生成行动
    print("\n【模拟生成带说明书指导的行动】")
    print("-" * 70)
    
    from io import BytesIO
    from PIL import Image
    
    # 创建模拟截图
    img = Image.new('RGB', (800, 600), color=(200, 220, 240))
    buffer = BytesIO()
    img.save(buffer, format='PNG')
    current_screenshot = buffer.getvalue()
    
    observation = {
        'screenshot': current_screenshot
    }
    
    task_instruction = "按照说明书的步骤操作应用程序"
    
    # 生成行动，同时获取说明书上下文
    agent_info, actions, instruction_context = worker.generate_next_action_with_instruction_context(
        instruction=task_instruction,
        observation=observation,
        use_instruction_guidance=True
    )
    
    print(f"✅ 生成行动成功")
    print(f"   Agent Info: {agent_info}")
    print(f"   Actions: {actions}")
    
    if instruction_context:
        print(f"\n【说明书指导】")
        print("-" * 70)
        matches = instruction_context.get('matches', [])
        if matches:
            best = matches[0]
            print(f"最匹配的步骤：第 {best['page_num']} 页")
            print(f"相似度：{best['similarity_score']:.2%}")
            print(f"描述：{best['description']}")
            if best.get('next_step_text'):
                print(f"下一步提示：{best['next_step_text'][:100]}...")
    
    print("\n" + "=" * 70)
    print("示例完成！")
    print("=" * 70)


# 高级示例：直接在 worker 中使用
def advanced_example():
    """高级用法示例"""
    
    print("\n" + "=" * 70)
    print("高级用法示例：直接使用 InstructionReader")
    print("=" * 70)
    
    # 直接创建 InstructionReader
    reader = InstructionReader(
        engine_params={"model": "gpt-4"},
        platform="windows"
    )
    
    # 从 Markdown 加载
    reader.load_instruction_from_markdown("example_instruction.md")
    
    print(f"\n✅ 加载的说明书信息：")
    print(f"   标题：{reader.instruction.title}")
    print(f"   软件：{reader.instruction.software_name}")
    print(f"   总页数：{reader.instruction.get_total_pages()}")
    
    # 获取说明书中的截图
    screenshots = reader.get_instruction_screenshot_pages()
    print(f"\n   说明书中包含 {len(screenshots)} 张截图：")
    for page in screenshots:
        print(f"      - 第 {page.page_num} 页：{page.description}")
    
    # 获取所有文字指导
    text_pages = reader.get_instruction_text_pages()
    print(f"\n   说明书中包含 {len(text_pages)} 页文字说明")
    
    print("\n" + "=" * 70)
    print("高级示例完成！")
    print("=" * 70)


if __name__ == "__main__":
    try:
        example_usage()
    except Exception as e:
        print(f"⚠️ 示例执行失败：{e}")
    
    try:
        advanced_example()
    except Exception as e:
        print(f"⚠️ 高级示例执行失败：{e}")
