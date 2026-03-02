import os
import time
import traceback
from typing import Callable, Dict

from PySide6.QtCore import QThread, Signal
import pyautogui
import pyperclip

from aci_funcs import mouseClick, mouseMove
from core import CommandType, WorkflowConfig, WorkflowNode
from utils import AtomicObject


# --------------------------
# 核心逻辑 (原 waterRPA.py)
# --------------------------

class WorkerThread(QThread):
    log_signal = Signal(str)
    finished_signal = Signal()

    def __init__(self, engine, tasks, loop_forever):
        super().__init__()
        self.engine = engine
        self.tasks = tasks
        self.loop_forever = loop_forever

    def run(self):
        self.engine.run_tasks(self.tasks, self.loop_forever, self.log_callback)
        self.finished_signal.emit()

    def log_callback(self, msg):
        self.log_signal.emit(msg)


class RPAEngine:
    def __init__(self):
        self.is_running = False
        self.stop_requested = AtomicObject(False)

    def stop(self):
        self.stop_requested.value = True
        self.is_running = False

    def _normalize_workflow(self, tasks):
        """兼容旧版 list/dict 配置，并统一为 dataclass 节点格式"""
        return WorkflowConfig.from_raw(tasks)

    def _execute_node(self, task, callback_msg:Callable=None):
        assert isinstance(task, WorkflowNode), f"节点必须是 WorkflowNode 实例，当前类型: {type(task)}"
        cmd_type = CommandType.from_raw(task.type)
        cmd_value = task.value
        timeout_second = float(task.timeout_second)

        try:
            if cmd_type == CommandType.LEFT_CLICK: # 单击左键
                success = mouseClick(1, "left", cmd_value, timeout_second, self.stop_requested)
                if callback_msg: callback_msg(f"单击左键: {cmd_value} {'成功' if success else '失败'}")
                return success

            elif cmd_type == CommandType.LEFT_DOUBLE_CLICK: # 双击左键
                success = mouseClick(2, "left", cmd_value, timeout_second, self.stop_requested)
                if callback_msg: callback_msg(f"双击左键: {cmd_value} {'成功' if success else '失败'}")
                return success

            elif cmd_type == CommandType.RIGHT_CLICK: # 右键
                success = mouseClick(1, "right", cmd_value, timeout_second, self.stop_requested)
                if callback_msg: callback_msg(f"右键单击: {cmd_value} {'成功' if success else '失败'}")
                return success

            elif cmd_type == CommandType.INPUT_TEXT: # 输入
                pyperclip.copy(str(cmd_value))
                pyautogui.hotkey('ctrl', 'v')
                time.sleep(0.5)
                if callback_msg: callback_msg(f"输入文本: {cmd_value}")
                return True

            elif cmd_type == CommandType.WAIT: # 等待
                sleep_time = float(cmd_value)
                time.sleep(sleep_time)
                if callback_msg: callback_msg(f"等待 {sleep_time} 秒")
                return True

            elif cmd_type == CommandType.SCROLL: # 滚轮
                scroll_val = int(cmd_value)
                pyautogui.scroll(scroll_val)
                if callback_msg: callback_msg(f"滚轮滑动 {scroll_val}")
                return True

            elif cmd_type == CommandType.HOTKEY: # 系统按键 (组合键)
                keys = str(cmd_value).lower().split('+')
                keys = [k.strip() for k in keys]
                pyautogui.hotkey(*keys)
                if callback_msg: callback_msg(f"按键组合: {cmd_value}")
                return True

            elif cmd_type == CommandType.HOVER: # 鼠标悬停
                success = mouseMove(cmd_value, timeout_second, self.stop_requested)
                if callback_msg: callback_msg(f"鼠标悬停: {cmd_value} {'成功' if success else '失败'}")
                return success

            elif cmd_type == CommandType.SCREENSHOT: # 截图保存
                path = str(cmd_value)
                if os.path.isdir(path):
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    filename = os.path.join(path, f"screenshot_{timestamp}.png")
                else:
                    filename = path
                    if not filename.endswith(('.png', '.jpg', '.bmp')):
                        filename += '.png'

                pyautogui.screenshot(filename)
                if callback_msg: callback_msg(f"截图已保存: {filename}")
                return True

            if callback_msg: callback_msg(f"未知操作类型: {cmd_type}")
            return False

        except Exception as e:
            if callback_msg: callback_msg(f"节点执行异常: {e}")
            return False

    def run_tasks(self, tasks, loop_forever=False, callback_msg=None):
        """
        """
        self.is_running = True
        self.stop_requested.value = False

        try:
            workflow = self._normalize_workflow(tasks)
            node_map:Dict[str, WorkflowNode] = {node.node_id: node for node in workflow.nodes}

            if len(node_map) != len(workflow.nodes):
                raise ValueError("存在重复的节点ID")

            if workflow.start_node not in node_map:
                raise ValueError(f"起始节点不存在: {workflow.start_node}")

            while True:
                current_node_id = workflow.start_node

                while current_node_id:
                    if self.stop_requested.value:
                        if callback_msg: callback_msg("任务已停止")
                        return

                    task = node_map.get(current_node_id)
                    if not task:
                        if callback_msg: callback_msg(f"节点不存在: {current_node_id}，流程终止")
                        break

                    if callback_msg:
                        callback_msg(f"执行节点 {current_node_id}: 类型={task.type}, 内容={task.value}")

                    success = self._execute_node(task, callback_msg)

                    next_node = str(task.next).strip()
                    fallback_next = str(task.fallback_next).strip()
                    target_node = next_node if success else (fallback_next or next_node)

                    if callback_msg:
                        route = "成功" if success else "失败"
                        callback_msg(f"节点 {current_node_id} {route}，跳转 -> {target_node or '结束'}")

                    current_node_id = target_node

                if not loop_forever:
                    break

                if callback_msg: callback_msg("等待 0.1 秒进入下一轮循环...")
                time.sleep(0.1)

        except Exception as e:
            if callback_msg: callback_msg(f"执行出错: {e}")
            traceback.print_exc()
        finally:
            self.is_running = False
            if callback_msg: callback_msg("任务结束")
