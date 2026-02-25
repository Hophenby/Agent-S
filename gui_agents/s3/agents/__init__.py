"""
Agent S3 - Agents Module

包含各种 Agent 组件：
- Agent 基类和具体实现
- 说明书加载和匹配功能
- 工作流和协调逻辑
"""

# 导入说明书相关类
try:
    from .instruction import Instruction, InstructionPage
    from .instruction_reader import InstructionReader
    from .instruction_integration_example import WorkerWithInstructionSupport
except ImportError:
    # 如果导入失败，仍然允许模块加载
    pass

__all__ = [
    'Instruction',
    'InstructionPage',
    'InstructionReader',
    'WorkerWithInstructionSupport',
]
