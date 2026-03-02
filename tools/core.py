
import enum
from dataclasses import dataclass
from typing import Any


class NodeCategory(enum.Enum):
    START = "起始节点"
    END = "结束节点"
    MIDDLE = "中间节点"


class CommandType(enum.Enum):
    LEFT_CLICK = 1.0
    LEFT_DOUBLE_CLICK = 2.0
    RIGHT_CLICK = 3.0
    INPUT_TEXT = 4.0
    WAIT = 5.0
    SCROLL = 6.0
    HOTKEY = 7.0
    HOVER = 8.0
    SCREENSHOT = 9.0

    @classmethod
    def ordered(cls):
        return [
            cls.LEFT_CLICK,
            cls.LEFT_DOUBLE_CLICK,
            cls.RIGHT_CLICK,
            cls.INPUT_TEXT,
            cls.WAIT,
            cls.SCROLL,
            cls.HOTKEY,
            cls.HOVER,
            cls.SCREENSHOT,
        ]

    @property
    def label(self):
        return {
            CommandType.LEFT_CLICK: "左键单击",
            CommandType.LEFT_DOUBLE_CLICK: "左键双击",
            CommandType.RIGHT_CLICK: "右键单击",
            CommandType.INPUT_TEXT: "输入文本",
            CommandType.WAIT: "等待(秒)",
            CommandType.SCROLL: "滚轮滑动",
            CommandType.HOTKEY: "系统按键",
            CommandType.HOVER: "鼠标悬停",
            CommandType.SCREENSHOT: "截图保存",
        }[self]

    @classmethod
    def from_label(cls, label: str):
        normalized = (label or "").strip()
        for cmd in cls.ordered():
            if cmd.label == normalized:
                return cmd
        raise ValueError(f"未知操作类型文本: {label}")

    @classmethod
    def from_raw(cls, raw_type: Any):
        if isinstance(raw_type, cls):
            return raw_type

        if isinstance(raw_type, str):
            normalized = raw_type.strip()
            if normalized:
                try:
                    return cls.from_label(normalized)
                except ValueError:
                    pass

        try:
            return cls(float(raw_type))
        except Exception as e:
            raise ValueError(f"未知操作类型值: {raw_type}") from e

@dataclass
class WorkflowNode:
    node_id: str
    type: CommandType
    value: str
    timeout_second: float = 0
    next: str = ""
    fallback_next: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any], index: int = 0, total: int = 0):
        node_id = str(data.get("node_id", index + 1)).strip() or str(index + 1)
        next_node = str(data.get("next", "")).strip()
        # if not next_node and index + 1 < total:
        #     next_node = str(index + 2)

        return cls(
            node_id=node_id,
            type=CommandType.from_raw(data.get("type", CommandType.LEFT_CLICK.value)),
            value=str(data.get("value", "")),
            timeout_second=float(data.get("timeout_second", data.get("retry", 0))),
            next=next_node,
            fallback_next=str(data.get("fallback_next", "")).strip()
        )

    def to_dict(self):
        return {
            "node_id": self.node_id,
            "type": self.type.value,
            "value": self.value,
            "timeout_second": self.timeout_second,
            "next": self.next,
            "fallback_next": self.fallback_next
        }

@dataclass
class WorkflowConfig:
    start_node: str
    nodes: list[WorkflowNode]

    @classmethod
    def from_raw(cls, data: Any):
        if isinstance(data, cls):
            return data

        if isinstance(data, dict):
            raw_nodes = data.get("nodes", [])
            start_node = str(data.get("start_node", "")).strip()
        else:
            raw_nodes = data or []
            start_node = ""

        if not isinstance(raw_nodes, list):
            raise ValueError("配置格式错误: nodes 必须是列表")

        nodes: list[WorkflowNode] = []
        for idx, node in enumerate(raw_nodes):
            if isinstance(node, WorkflowNode):
                current = WorkflowNode(
                    node_id=str(node.node_id).strip() or str(idx + 1),
                    type=CommandType.from_raw(node.type),
                    value=str(node.value),
                    timeout_second=float(getattr(node, "timeout_second", 0)),
                    next=str(node.next).strip() or (str(idx + 2) if idx + 1 < len(raw_nodes) else ""),
                    fallback_next=str(node.fallback_next).strip()
                )
            elif isinstance(node, dict):
                current = WorkflowNode.from_dict(node, idx, len(raw_nodes))
            else:
                raise ValueError(f"节点格式错误: 第 {idx+1} 项不是对象")
            nodes.append(current)

        used_ids: set[str] = set()
        for idx, node in enumerate(nodes):
            base_id = str(node.node_id).strip() or str(idx + 1)
            unique_id = base_id

            suffix = 2
            while unique_id in used_ids:
                unique_id = f"{base_id}_{suffix}"
                suffix += 1

            node.node_id = unique_id
            used_ids.add(unique_id)

        if not nodes:
            raise ValueError("没有可执行的节点")

        if not start_node:
            start_node = nodes[0].node_id

        if start_node not in used_ids:
            start_node = nodes[0].node_id

        return cls(start_node=start_node, nodes=nodes)

    def to_dict(self):
        return {
            "start_node": self.start_node,
            "nodes": [node.to_dict() for node in self.nodes]
        }

