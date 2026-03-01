from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from instruction.yaml.yaml_instruction import *



def _require(mapping: Dict[str, Any], key: str, ctx: str) -> Any:
    if key not in mapping:
        raise InstructionParseError(f"Missing required key '{key}' in {ctx}.")
    return mapping[key]


def _as_optional_str(value: Any) -> Optional[str]:
    if value is None:
        return None
    return str(value)


def _as_optional_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _as_optional_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, bool):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_optional_bool(value: Any) -> Optional[bool]:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "yes", "1"}:
            return True
        if lowered in {"false", "no", "0"}:
            return False
    return None


def _as_action_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    return [str(value)]


def _parse_images(data: Any) -> Images:
    if not isinstance(data, dict):
        return Images()
    return Images(
        step_image=_as_optional_str(data.get("step_image")),
        result_image=_as_optional_str(data.get("result_image")),
    )


def _parse_template_matching(data: Any) -> TemplateMatching:
    if not isinstance(data, dict):
        return TemplateMatching()

    scales = data.get("scales")
    parsed_scales: List[float] = []
    if isinstance(scales, list):
        for value in scales:
            scale = _as_optional_float(value)
            if scale is not None:
                parsed_scales.append(scale)

    return TemplateMatching(
        threshold=_as_optional_float(data.get("threshold")),
        scales=parsed_scales,
        min_scale=_as_optional_float(data.get("min_scale")),
        max_scale=_as_optional_float(data.get("max_scale")),
        scale_step=_as_optional_float(data.get("scale_step")),
    )


def _parse_transition_condition(data: Any) -> TransitionCondition:
    if not isinstance(data, dict):
        return TransitionCondition()
    return TransitionCondition(
        status=_as_optional_str(data.get("status")),
        matched_postcondition=_as_optional_str(data.get("matched_postcondition")),
        result_image_matched=_as_optional_bool(data.get("result_image_matched")),
        expected_result_matched=_as_optional_bool(data.get("expected_result_matched")),
        text_present=_as_optional_str(data.get("text_present")),
        text_absent=_as_optional_str(data.get("text_absent")),
        always=_as_optional_bool(data.get("always")),
    )


def _parse_transitions(items: Any) -> List[Transition]:
    if not isinstance(items, list):
        return []

    results: List[Transition] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        results.append(
            Transition(
                goto=_as_optional_str(item.get("goto")),
                when=_parse_transition_condition(item.get("when")),
                else_branch=bool(_as_optional_bool(item.get("else"))),
                label=_as_optional_str(item.get("label")),
            )
        )
    return results


def _parse_postconditions(items: Any, ctx: str) -> List[Postcondition]:
    if not isinstance(items, list):
        return []

    results: List[Postcondition] = []
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            continue
        postcondition_id = _as_optional_str(item.get("id"))
        if not postcondition_id:
            raise InstructionParseError(
                f"Missing required key 'id' in {ctx}.postconditions[{index}]."
            )
        results.append(
            Postcondition(
                id=postcondition_id,
                result_image=_as_optional_str(item.get("result_image")),
                expected_text=_as_optional_str(
                    item.get("expected_text", item.get("expected_result"))
                ),
                description=_as_optional_str(item.get("description")),
            )
        )
    return results


def _as_string_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    return [str(value)]


def _parse_mouse_inputs(items: Any) -> List[MouseInput]:
    if not isinstance(items, list):
        return []
    results: List[MouseInput] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        results.append(
            MouseInput(
                button=_as_optional_str(item.get("button")),
                action=_as_optional_str(item.get("action")),
                clicks=_as_optional_int(item.get("clicks")),
                duration_ms=_as_optional_int(item.get("duration_ms")),
                position=item.get("position") if isinstance(item.get("position"), dict) else None,
                target=_as_optional_str(item.get("target")),
                modifiers=_as_string_list(item.get("modifiers")),
            )
        )
    return results


def _parse_key_inputs(items: Any) -> List[KeyInput]:
    if not isinstance(items, list):
        return []
    results: List[KeyInput] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        results.append(
            KeyInput(
                keys=_as_string_list(item.get("keys")),
                hold_ms=_as_optional_int(item.get("hold_ms")),
                repeat=_as_optional_int(item.get("repeat")),
            )
        )
    return results


def _parse_text_inputs(items: Any) -> List[TextInput]:
    if not isinstance(items, list):
        return []
    results: List[TextInput] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        results.append(
            TextInput(
                text=_as_optional_str(item.get("text")),
                clear_before=_as_optional_bool(item.get("clear_before")),
                input_method=_as_optional_str(item.get("input_method")),
            )
        )
    return results


def _parse_scroll_inputs(items: Any) -> List[ScrollInput]:
    if not isinstance(items, list):
        return []
    results: List[ScrollInput] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        results.append(
            ScrollInput(
                direction=_as_optional_str(item.get("direction")),
                amount=_as_optional_int(item.get("amount")),
                unit=_as_optional_str(item.get("unit")),
                position=item.get("position") if isinstance(item.get("position"), dict) else None,
                target=_as_optional_str(item.get("target")),
            )
        )
    return results


def _parse_drag_drops(items: Any) -> List[DragDrop]:
    if not isinstance(items, list):
        return []
    results: List[DragDrop] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        results.append(
            DragDrop(
                from_target=_as_optional_str(item.get("from", item.get("from_target"))),
                to_target=_as_optional_str(item.get("to", item.get("to_target"))),
                button=_as_optional_str(item.get("button")),
                duration_ms=_as_optional_int(item.get("duration_ms")),
            )
        )
    return results


def _parse_hover_inputs(items: Any) -> List[HoverInput]:
    if not isinstance(items, list):
        return []
    results: List[HoverInput] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        results.append(
            HoverInput(
                target=_as_optional_str(item.get("target")),
                duration_ms=_as_optional_int(item.get("duration_ms")),
            )
        )
    return results


def _parse_window_inputs(items: Any) -> List[WindowInput]:
    if not isinstance(items, list):
        return []
    results: List[WindowInput] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        results.append(
            WindowInput(
                action=_as_optional_str(item.get("action")),
                title=_as_optional_str(item.get("title")),
            )
        )
    return results


def _parse_file_inputs(items: Any) -> List[FileInput]:
    if not isinstance(items, list):
        return []
    results: List[FileInput] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        results.append(
            FileInput(
                dialog_action=_as_optional_str(item.get("dialog_action")),
                path=_as_optional_str(item.get("path")),
                filename=_as_optional_str(item.get("filename")),
            )
        )
    return results


def _parse_clipboard_inputs(items: Any) -> List[ClipboardInput]:
    if not isinstance(items, list):
        return []
    results: List[ClipboardInput] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        results.append(
            ClipboardInput(
                action=_as_optional_str(item.get("action")),
                text=_as_optional_str(item.get("text")),
            )
        )
    return results


def _parse_wait_inputs(items: Any) -> List[WaitInput]:
    if not isinstance(items, list):
        return []
    results: List[WaitInput] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        results.append(
            WaitInput(
                condition=_as_optional_str(item.get("condition")),
                timeout_sec=_as_optional_int(item.get("timeout_sec")),
            )
        )
    return results


def _parse_special_inputs(items: Any) -> List[SpecialInput]:
    if not isinstance(items, list):
        return []
    results: List[SpecialInput] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        results.append(SpecialInput(description=_as_optional_str(item.get("description"))))
    return results


def _parse_actions(data: Any) -> Optional[Actions]:
    if data is None:
        return None
    if not isinstance(data, dict):
        return None
    return Actions(
        mouse_input=_parse_mouse_inputs(data.get("mouse_input")),
        key_input=_parse_key_inputs(data.get("key_input")),
        text_input=_parse_text_inputs(data.get("text_input")),
        scroll_input=_parse_scroll_inputs(data.get("scroll_input")),
        drag_drop=_parse_drag_drops(data.get("drag_drop")),
        hover_input=_parse_hover_inputs(data.get("hover_input")),
        window_input=_parse_window_inputs(data.get("window_input")),
        file_input=_parse_file_inputs(data.get("file_input")),
        clipboard_input=_parse_clipboard_inputs(data.get("clipboard_input")),
        wait=_parse_wait_inputs(data.get("wait")),
        special=_parse_special_inputs(data.get("special")),
    )


def _parse_step(step_data: Dict[str, Any]) -> Step:
    step_id = str(_require(step_data, "id", "step"))
    name = str(_require(step_data, "name", f"step '{step_id}'"))
    description = _as_optional_str(step_data.get("description"))
    action = _as_action_list(step_data.get("action"))
    actions = _parse_actions(step_data.get("actions"))
    images = _parse_images(step_data.get("images"))
    template_matching = _parse_template_matching(step_data.get("template_matching"))
    expected_result = _as_optional_str(step_data.get("expected_result"))
    element_text = _as_optional_str(step_data.get("element_text"))
    on_success = _as_optional_str(step_data.get("on_success"))
    on_failure = _as_optional_str(step_data.get("on_failure"))
    postconditions = _parse_postconditions(
        step_data.get("postconditions"),
        f"step '{step_id}'",
    )
    transitions = _parse_transitions(step_data.get("transitions"))
    timeout_sec = _as_optional_float(step_data.get("timeout_sec"))
    pre_processing_delay_millisec = _as_optional_int(step_data.get("pre_processing_delay_millisec"))
    post_processing_delay_millisec = _as_optional_int(step_data.get("post_processing_delay_millisec"))
    retry = _as_optional_int(step_data.get("retry"))

    known_keys = {
        "id",
        "name",
        "description",
        "action",
        "actions",
        "images",
        "template_matching",
        "expected_result",
        "timeout_sec",
        "retry",
        "element_text",
        "on_success",
        "on_failure",
        "postconditions",
        "transitions",
        "pre_processing_delay_millisec",
        "post_processing_delay_millisec",
    }
    extra = {key: value for key, value in step_data.items() if key not in known_keys}

    return Step(
        id=step_id,
        name=name,
        description=description,
        action=action,
        actions=actions,
        images=images,
        template_matching=template_matching,
        expected_result=expected_result,
        element_text=element_text,
        on_success=on_success,
        on_failure=on_failure,
        postconditions=postconditions,
        transitions=transitions,
        timeout_sec=timeout_sec,
        retry=retry,
        extra=extra,
        pre_processing_delay_millisec=pre_processing_delay_millisec,
        post_processing_delay_millisec=post_processing_delay_millisec,
    )


def _parse_job(job_data: Dict[str, Any], job_id: str) -> Job:
    name = str(_require(job_data, "name", f"job '{job_id}'"))
    runs_on = _as_optional_str(job_data.get("runs-on", job_data.get("runs_on")))
    steps_data = _require(job_data, "steps", f"job '{job_id}'")
    if not isinstance(steps_data, list):
        raise InstructionParseError(f"Expected 'steps' to be a list in job '{job_id}'.")
    steps = [_parse_step(step) for step in steps_data]
    return Job(name=name, runs_on=runs_on, steps=steps)


def _parse_metadata(data: Dict[str, Any]) -> Metadata:
    title = str(_require(data, "title", "metadata"))
    software_data = _require(data, "software", "metadata")
    if not isinstance(software_data, dict):
        raise InstructionParseError("Expected 'software' to be an object in metadata.")
    software = Software(
        name=str(_require(software_data, "name", "metadata.software")),
        version=str(_require(software_data, "version", "metadata.software")),
    )

    return Metadata(
        title=title,
        software=software,
        language=_as_optional_str(data.get("language")),
        source_markdown=_as_optional_str(data.get("source_markdown")),
        author=_as_optional_str(data.get("author")),
        updated_at=_as_optional_str(data.get("updated_at")),
    )


def parse_instruction(data: Dict[str, Any]) -> YamlInstruction:
    if not isinstance(data, dict):
        raise InstructionParseError("Instruction YAML must parse into a mapping/object.")

    name = str(_require(data, "name", "root"))
    metadata = _parse_metadata(_require(data, "metadata", "root"))
    on_section = data.get("on")
    if on_section is None and True in data:
        on_section = data[True]
    if on_section is None:
        raise InstructionParseError("Missing required key 'on' in root.")
    if not isinstance(on_section, dict):
        raise InstructionParseError("Expected 'on' to be an object in root.")

    jobs_data = _require(data, "jobs", "root")
    if not isinstance(jobs_data, dict):
        raise InstructionParseError("Expected 'jobs' to be an object in root.")

    jobs: Dict[str, Job] = {}
    for job_id, job_data in jobs_data.items():
        if not isinstance(job_data, dict):
            raise InstructionParseError(f"Job '{job_id}' must be an object.")
        jobs[job_id] = _parse_job(job_data, job_id)

    return YamlInstruction(name=name, metadata=metadata, on=on_section, jobs=jobs)


def load_instruction(path: str | Path) -> YamlInstruction:
    instruction_path = Path(path)
    with instruction_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    instruction = parse_instruction(raw)
    return resolve_instruction_paths(instruction, instruction_path.parent)


__all__ = [
    "YamlInstruction",
    "InstructionParseError",
    "Job",
    "Metadata",
    "Software",
    "Step",
    "Images",
    "MouseInput",
    "KeyInput",
    "TextInput",
    "ScrollInput",
    "DragDrop",
    "HoverInput",
    "WindowInput",
    "FileInput",
    "ClipboardInput",
    "WaitInput",
    "SpecialInput",
    "Actions",
    "load_instruction",
    "parse_instruction",
]
