
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Software:
    name: str
    version: str


@dataclass
class Metadata:
    title: str
    software: Software
    language: Optional[str] = None
    source_markdown: Optional[str] = None
    author: Optional[str] = None
    updated_at: Optional[str] = None


@dataclass
class Images:
    step_image: Optional[str] = None
    result_image: Optional[str] = None


@dataclass
class MouseInput:
    button: Optional[str] = None
    action: Optional[str] = None
    clicks: Optional[int] = None
    duration_ms: Optional[int] = None
    position: Optional[Dict[str, Any]] = None
    target: Optional[str] = None
    modifiers: List[str] = field(default_factory=list)


@dataclass
class KeyInput:
    keys: List[str] = field(default_factory=list)
    hold_ms: Optional[int] = None
    repeat: Optional[int] = None


@dataclass
class TextInput:
    text: Optional[str] = None
    clear_before: Optional[bool] = None
    input_method: Optional[str] = None


@dataclass
class ScrollInput:
    direction: Optional[str] = None
    amount: Optional[int] = None
    unit: Optional[str] = None
    position: Optional[Dict[str, Any]] = None
    target: Optional[str] = None


@dataclass
class DragDrop:
    from_target: Optional[str] = None
    to_target: Optional[str] = None
    button: Optional[str] = None
    duration_ms: Optional[int] = None


@dataclass
class HoverInput:
    target: Optional[str] = None
    duration_ms: Optional[int] = None


@dataclass
class WindowInput:
    action: Optional[str] = None
    title: Optional[str] = None


@dataclass
class FileInput:
    dialog_action: Optional[str] = None
    path: Optional[str] = None
    filename: Optional[str] = None


@dataclass
class ClipboardInput:
    action: Optional[str] = None
    text: Optional[str] = None


@dataclass
class WaitInput:
    condition: Optional[str] = None
    timeout_sec: Optional[int] = None


@dataclass
class SpecialInput:
    description: Optional[str] = None


@dataclass
class Actions:
    mouse_input: List[MouseInput] = field(default_factory=list)
    key_input: List[KeyInput] = field(default_factory=list)
    text_input: List[TextInput] = field(default_factory=list)
    scroll_input: List[ScrollInput] = field(default_factory=list)
    drag_drop: List[DragDrop] = field(default_factory=list)
    hover_input: List[HoverInput] = field(default_factory=list)
    window_input: List[WindowInput] = field(default_factory=list)
    file_input: List[FileInput] = field(default_factory=list)
    clipboard_input: List[ClipboardInput] = field(default_factory=list)
    wait: List[WaitInput] = field(default_factory=list)
    special: List[SpecialInput] = field(default_factory=list)


@dataclass
class Step:
    id: str
    name: str
    description: Optional[str] = None
    action: List[str] = field(default_factory=list)
    actions: Optional[Actions] = None
    images: Images = field(default_factory=Images)
    expected_result: Optional[str] = None
    timeout_sec: Optional[int] = None
    pre_processing_delay_millisec: Optional[float] = None
    post_processing_delay_millisec: Optional[float] = None
    retry: Optional[int] = None
    element_text: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Job:
    name: str
    runs_on: Optional[str]
    steps: List[Step]


@dataclass
class YamlInstruction:
    name: str
    metadata: Metadata
    on: Dict[str, Any]
    jobs: Dict[str, Job]

    @property
    def steps(self) -> List[Step]:
        all_steps = []
        for job in self.jobs.values():
            all_steps.extend(job.steps)
        return all_steps


class InstructionParseError(ValueError):
    pass
