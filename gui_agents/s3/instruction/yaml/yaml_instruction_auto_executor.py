from __future__ import annotations

from io import BytesIO
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import cv2
from cv2.typing import MatLike
import numpy as np
from PIL import Image
import pyautogui
import pytesseract

from core.observation import Observation
from agents.execution_summary import ExecutionSummary
from instruction.yaml.yaml_instruction import Actions, Step


@dataclass
class TemplateMatch:
    x: int
    y: int
    width: int
    height: int
    confidence: float


def _load_image(path: bytes| str | Path) -> MatLike:
    if isinstance(path, bytes):
        image = Image.open(BytesIO(path)).convert("RGB")
        return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    else:
        image = cv2.imread(str(path))
    if image is None:
        raise FileNotFoundError(f"Image not found or unreadable: {path}")
    return image


def find_template_bbox(
    screenshot_path: bytes,
    template_path: str | Path,
    threshold: float = 0.8,
) -> Optional[TemplateMatch]:
    screenshot = _load_image(screenshot_path)
    template = _load_image(Path(template_path))

    result = cv2.matchTemplate(screenshot, template, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    if max_val < threshold:
        return None

    height, width = template.shape[:2]
    return TemplateMatch(
        x=max_loc[0],
        y=max_loc[1],
        width=width,
        height=height,
        confidence=float(max_val),
    )


def _center_of_bbox(bbox: TemplateMatch) -> Tuple[int, int]:
    return (bbox.x + bbox.width // 2, bbox.y + bbox.height // 2)


def _resolve_point(position: Optional[Dict[str, Any]], bbox: Optional[TemplateMatch]) -> Optional[Tuple[int, int]]:
    if isinstance(position, dict) and "x" in position and "y" in position:
        return (int(position["x"]), int(position["y"]))
    if bbox is None:
        return None
    return _center_of_bbox(bbox)


def _apply_mouse_input(actions: Actions, bbox: Optional[TemplateMatch]) -> None:
    for item in actions.mouse_input:
        point = _resolve_point(item.position, bbox)
        if point is not None:
            pyautogui.moveTo(point[0], point[1])

        if item.modifiers:
            for key in item.modifiers:
                pyautogui.keyDown(key)

        if item.action in {"press", "long_press"}:
            pyautogui.mouseDown(button=item.button or "left")
            if item.duration_ms:
                time.sleep(item.duration_ms / 1000)
            if item.action != "press":
                pyautogui.mouseUp(button=item.button or "left")
        elif item.action == "release":
            pyautogui.mouseUp(button=item.button or "left")
        elif item.action == "double_click":
            pyautogui.doubleClick(button=item.button or "left")
        else:
            pyautogui.click(button=item.button or "left", clicks=item.clicks or 1)

        if item.modifiers:
            for key in reversed(item.modifiers):
                pyautogui.keyUp(key)


def _apply_key_input(actions: Actions) -> None:
    for item in actions.key_input:
        keys = item.keys or []
        if not keys:
            continue
        if item.hold_ms:
            for key in keys:
                pyautogui.keyDown(key)
            time.sleep(item.hold_ms / 1000)
            for key in reversed(keys):
                pyautogui.keyUp(key)
        elif len(keys) > 1:
            pyautogui.hotkey(*keys)
        else:
            pyautogui.press(keys[0], presses=item.repeat or 1)


def _apply_text_input(actions: Actions) -> None:
    for item in actions.text_input:
        if item.clear_before:
            pyautogui.hotkey("ctrl", "a")
            pyautogui.press("backspace")
        if item.text:
            pyautogui.write(item.text)


def _apply_scroll_input(actions: Actions, bbox: Optional[TemplateMatch]) -> None:
    for item in actions.scroll_input:
        point = _resolve_point(item.position, bbox)
        if point is not None:
            pyautogui.moveTo(point[0], point[1])

        amount = item.amount or 0
        unit = (item.unit or "notch").lower()
        if unit == "page":
            amount *= 800
        elif unit == "notch":
            amount *= 120

        direction = (item.direction or "down").lower()
        if direction in {"left", "right"}:
            pyautogui.hscroll(amount if direction == "right" else -amount)
        else:
            pyautogui.scroll(amount if direction == "up" else -amount)


def _apply_drag_drop(actions: Actions, bbox: Optional[TemplateMatch]) -> None:
    for item in actions.drag_drop:
        from_point = _resolve_point(item.from_target if isinstance(item.from_target, dict) else None, bbox)
        to_point = _resolve_point(item.to_target if isinstance(item.to_target, dict) else None, bbox)
        if from_point is None and isinstance(item.from_target, str) and bbox is not None:
            from_point = _center_of_bbox(bbox)
        if to_point is None and isinstance(item.to_target, str) and bbox is not None:
            to_point = _center_of_bbox(bbox)
        if from_point is None or to_point is None:
            continue
        pyautogui.moveTo(from_point[0], from_point[1])
        pyautogui.mouseDown(button=item.button or "left")
        pyautogui.moveTo(to_point[0], to_point[1], duration=(item.duration_ms or 200) / 1000)
        pyautogui.mouseUp(button=item.button or "left")


def _apply_hover_input(actions: Actions, bbox: Optional[TemplateMatch]) -> None:
    for item in actions.hover_input:
        point = _resolve_point(None, bbox)
        if point is None:
            continue
        pyautogui.moveTo(point[0], point[1])
        if item.duration_ms:
            time.sleep(item.duration_ms / 1000)


def _apply_window_input(actions: Actions) -> None:
    for item in actions.window_input:
        if item.action == "close":
            pyautogui.hotkey("alt", "f4")
        elif item.action == "minimize":
            pyautogui.hotkey("win", "down")
        elif item.action == "maximize":
            pyautogui.hotkey("win", "up")


def _apply_file_input(actions: Actions) -> None:
    for item in actions.file_input:
        if item.path:
            pyautogui.write(item.path)
        if item.filename:
            pyautogui.write(item.filename)
        pyautogui.press("enter")


def _apply_clipboard_input(actions: Actions) -> None:
    for item in actions.clipboard_input:
        if item.action == "copy":
            pyautogui.hotkey("ctrl", "c")
        elif item.action == "cut":
            pyautogui.hotkey("ctrl", "x")
        elif item.action == "paste":
            pyautogui.hotkey("ctrl", "v")
        elif item.action == "set_text" and item.text:
            pyautogui.write(item.text)


def _apply_wait(actions: Actions) -> None:
    for item in actions.wait:
        if item.timeout_sec:
            time.sleep(item.timeout_sec)


def execute_step(
    step: Step,
    obs: Observation,
    step_index: int,
    template_threshold: float = 0.8,
) -> Optional[ExecutionSummary]:
    _ = step_index
    step_image = step.images.step_image if step.images else None
    bbox = None
    if step_image:
        bbox = find_template_bbox(obs.original_screenshot, step_image, threshold=template_threshold)

    if bbox is None and step.element_text:
        screenshot = _load_image(obs.original_screenshot)
        ocr_result = pytesseract.image_to_data(screenshot, output_type=pytesseract.Output.DICT)
        for i in range(len(ocr_result["text"])):
            if step.element_text.lower() in ocr_result["text"][i].lower():
                bbox = TemplateMatch(
                    x=ocr_result["left"][i],
                    y=ocr_result["top"][i],
                    width=ocr_result["width"][i],
                    height=ocr_result["height"][i],
                    confidence=1.0,
                )
                break

    if step.actions is None:
        return None

    def call_executable():
        time.sleep(step.pre_processing_delay_millisec / 1000 if step.pre_processing_delay_millisec else 0.05)
        _apply_mouse_input(step.actions, bbox)
        _apply_key_input(step.actions)
        _apply_text_input(step.actions)
        _apply_scroll_input(step.actions, bbox)
        _apply_drag_drop(step.actions, bbox)
        _apply_hover_input(step.actions, bbox)
        _apply_window_input(step.actions)
        _apply_file_input(step.actions)
        _apply_clipboard_input(step.actions)
        _apply_wait(step.actions)
        time.sleep(step.post_processing_delay_millisec / 1000 if step.post_processing_delay_millisec else 0.05)

    return ExecutionSummary(
        plan=step.description or "",
        plan_code="",  # No code generation in this executor
        executable=call_executable,
        reflection="called by yaml configuration",
    )


__all__ = [
    "TemplateMatch",
    "find_template_bbox",
    "execute_step",
]
