import os
import re
from collections import defaultdict
from io import BytesIO
from typing import Any, Callable, Dict, List, Optional, Tuple

import pytesseract
from PIL import Image
from pytesseract import Output
import cv2
import numpy as np
import base64
from io import BytesIO

from core.observation import Observation
from agents.LegacyACIResult import LegacyACIResult
from memory.procedural_memory import PROCEDURAL_MEMORY
from core.mllm import LMMAgent
from utils.common_utils import RUNTIME_LOG_PATH, call_llm_safe
from agents.code_agent import CodeAgent
import logging
import json

logger = logging.getLogger("desktopenv.agent")

feedback_renderer: Optional[Callable[[bytes], bytes]] = None

def get_feedback_renderer() -> Optional[Callable[[bytes], bytes]]:
    return feedback_renderer
class ACI:
    def __init__(self):
        self.notes: List[str] = []


# Agent action decorator
def agent_action(func):
    func.is_agent_action = True
    return func


UBUNTU_APP_SETUP = f"""import subprocess;
import difflib;
import pyautogui;
pyautogui.press('escape');
time.sleep(0.5);
output = subprocess.check_output(['wmctrl', '-lx']);
output = output.decode('utf-8').splitlines();
window_titles = [line.split(None, 4)[2] for line in output];
closest_matches = difflib.get_close_matches('APP_NAME', window_titles, n=1, cutoff=0.1);
if closest_matches:
    closest_match = closest_matches[0];
    for line in output:
        if closest_match in line:
            window_id = line.split()[0]
            break;
subprocess.run(['wmctrl', '-ia', window_id])
subprocess.run(['wmctrl', '-ir', window_id, '-b', 'add,maximized_vert,maximized_horz'])
"""


SET_CELL_VALUES_CMD = """import uno
import subprocess
import unicodedata, json

def identify_document_type(component):
    if component.supportsService("com.sun.star.sheet.SpreadsheetDocument"):
        return "Calc"

    if component.supportsService("com.sun.star.text.TextDocument"):
        return "Writer"

    if component.supportsService("com.sun.star.sheet.PresentationDocument"):
        return "Impress"

    return None

def _norm_name(s: str | None) -> str | None:
    if s is None:
        return None
    if "\\\\u" in s or "\\\\U" in s or "\\\\x" in s:
        try:
            # json.loads handles all the escape forms safely
            s = json.loads(f"{{s}}")
        except Exception:
            # fallback: best-effort
            try:
                s = s.encode("utf-8").decode("unicode_escape")
            except Exception:
                pass
    # Normalize (NFC works well across platforms)
    return unicodedata.normalize("NFC", s)

def cell_ref_to_indices(cell_ref):
    column_letters = ''.join(filter(str.isalpha, cell_ref))
    row_number = ''.join(filter(str.isdigit, cell_ref))

    col = sum((ord(char.upper()) - ord('A') + 1) * (26**idx) for idx, char in enumerate(reversed(column_letters))) - 1
    row = int(row_number) - 1
    return col, row

def set_cell_values(new_cell_values: dict[str, str], app_name: str = "Untitled 1", sheet_name: str = "Sheet1"):
    app_name  = _norm_name(app_name)
    sheet_name = _norm_name(sheet_name)

    new_cell_values_idx = {{}}
    for k, v in new_cell_values.items():
        try:
            col, row = cell_ref_to_indices(k)
        except:
            col = row = None

        if col is not None and row is not None:
            new_cell_values_idx[(col, row)] = v

    # Clean up previous TCP connections.
    subprocess.run(
        'echo \"osworld-public-evaluation\" | sudo -S ss --kill --tcp state TIME-WAIT sport = :2002',
        shell=True,
        check=True,
        text=True,
        capture_output=True
    )

    # Dynamically allow soffice to listen on port 2002.
    subprocess.run(
        [
            "soffice",
            "--accept=socket,host=localhost,port=2002;urp;StarOffice.Service"
        ]
    )

    local_context = uno.getComponentContext()
    resolver = local_context.ServiceManager.createInstanceWithContext(
        "com.sun.star.bridge.UnoUrlResolver", local_context
    )
    context = resolver.resolve(
        f"uno:socket,host=localhost,port=2002;urp;StarOffice.ComponentContext"
    )
    desktop = context.ServiceManager.createInstanceWithContext(
        "com.sun.star.frame.Desktop", context
    )

    # Collect all LibreOffice-related opened windows.
    documents = []
    for i, component in enumerate(desktop.Components):
        title = component.Title
        doc_type = identify_document_type(component)
        documents.append((i, component, title, doc_type))

    # Find the LibreOffice Calc app and the sheet of interest.
    spreadsheet = [doc for doc in documents if doc[3] == "Calc"]
    selected_spreadsheet = [doc for doc in spreadsheet if doc[2] == app_name]
    if spreadsheet:
        try:
            if selected_spreadsheet:
                spreadsheet = selected_spreadsheet[0][1]
            else:
                spreadsheet = spreadsheet[0][1]

            sheet = spreadsheet.Sheets.getByName(sheet_name)
        except:
            raise ValueError(f"Could not find sheet {{sheet_name}} in {{app_name}}.")

        for (col, row), value in new_cell_values_idx.items():
            cell = sheet.getCellByPosition(col, row)

            # Set the cell value.
            if isinstance(value, (int, float)):
                cell.Value = value
            elif isinstance(value, str):
                if value.startswith("="):
                    cell.Formula = value
                else:
                    cell.String = value
            elif isinstance(value, bool):
                cell.Value = 1 if value else 0
            elif value is None:
                cell.clearContents(0)
            else:
                raise ValueError(f"Unsupported cell value type: {{type(value)}}")

    else:
        raise ValueError(f"Could not find LibreOffice Calc app corresponding to {{app_name}}.")

set_cell_values(new_cell_values={cell_values}, app_name="{app_name}", sheet_name="{sheet_name}")        
"""

class LegacyACI(ACI):
    def __init__(self, width: int = 1920, height: int = 1080):
        super().__init__()
        # default screen size (can be overridden)
        self.width = width
        self.height = height
        # last operation info for feedback drawing
        self._last_op = None
        # assignable screenshot (raw bytes or PIL image)
        self.obs = None
        # Cache for OCR elements to avoid repeated extraction
        self._ocr_cache = None
        self._ocr_cache_screenshot_hash = None

    def assign_screenshot(self, obs: Observation):
        """MCP 接口：分配当前截图（供后续操作和反馈绘制使用）。

        名称: LegacyACI.assign_screenshot

        参数:
            obs (dict): 必需，包含键 'screenshot' 的字典。screenshot 支持以下格式：
                - base64 编码的 JPEG/PNG 字符串
                - 原始 image bytes
                - PIL.Image 对象
                - numpy.ndarray

        返回:
            dict: {"result": "OK"}（在本地实现中将直接设置并无复杂返回值）

        错误:
            ValueError: 当提供的图片无法解析时可抛出 invalid_image 错误（调用者可捕获并处理）。

        示例:
            agent.assign_screenshot({"screenshot": "<base64-string>"})
        """
        self.obs = obs
        # Invalidate OCR cache when screenshot changes
        self._ocr_cache = None
        self._ocr_cache_screenshot_hash = None
    
    def _get_screenshot_hash(self) -> str:
        """Generate a hash of the current screenshot for caching."""
        import hashlib
        if not self.obs:
            return ""
        screenshot = self.obs.get("screenshot") if isinstance(self.obs, dict) else self.obs
        if isinstance(screenshot, str):
            return hashlib.md5(screenshot.encode()).hexdigest()[:16]
        elif isinstance(screenshot, (bytes, bytearray)):
            return hashlib.md5(screenshot).hexdigest()[:16]
        return str(id(screenshot))[:16]
    
    def get_ocr_elements(self, force_refresh: bool = False) -> List:
        """Extract OCR elements from the current screenshot with caching.
        
        Args:
            force_refresh: If True, bypass cache and re-extract
        
        Returns:
            List of OCRElement objects from agents.ocr module
        """
        from agents.ocr import extract_ocr_elements, get_ocr_instance
        
        screenshot_hash = self._get_screenshot_hash()
        
        # Use cache if available and screenshot hasn't changed
        if not force_refresh and self._ocr_cache is not None and screenshot_hash == self._ocr_cache_screenshot_hash:
            logger.debug("Using cached OCR elements")
            return self._ocr_cache
        
        # Extract elements from current screenshot
        img_bgr = self._load_image_as_bgr("original_screenshot")
        # Convert BGR to RGB for OCR
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
        try:
            ocr_instance = get_ocr_instance()
            elements = extract_ocr_elements(img_rgb, ocr_instance)
            
            # Cache the results
            self._ocr_cache = elements
            self._ocr_cache_screenshot_hash = screenshot_hash
            
            logger.info(f"Extracted {len(elements)} OCR elements from screenshot")
            with open(
                os.path.join(RUNTIME_LOG_PATH, f"step_{screenshot_hash[:6]}_screenshot.json"), "w"
            ) as f:
                json.dump([e.to_dict() for e in elements], f, indent=2)
            return elements
        except Exception as e:
            logger.error(f"Failed to extract OCR elements: {e}")
            return []

    def _to_abs_point(self, rel):
        """Convert relative coordinate or bbox to absolute point (x,y).
        Accepts:
          - point: (x_rel, y_rel) in [0,1]
          - bbox: (x_rel, y_rel, w_rel, h_rel) where x_rel,y_rel are top-left
        Returns (x,y) integers in pixels.
        """
        if rel is None:
            return None
        if len(rel) == 2:
            x = int(round(rel[0] * self.width))
            y = int(round(rel[1] * self.height))
            return x, y
        elif len(rel) >= 4:
            x = int(round((rel[0] + rel[2] / 2.0) * self.width))
            y = int(round((rel[1] + rel[3] / 2.0) * self.height))
            return x, y
        else:
            raise ValueError("Relative coordinate must be len 2 or >=4")

    def _to_abs_bbox(self, rel):
        """Convert relative bbox (x,y,w,h) to absolute bbox in pixels: (x1,y1,x2,y2)"""
        if rel is None:
            return None
        if len(rel) >= 4:
            x1 = int(round(rel[0] * self.width))
            y1 = int(round(rel[1] * self.height))
            w = int(round(rel[2] * self.width))
            h = int(round(rel[3] * self.height))
            x2 = x1 + w
            y2 = y1 + h
            return x1, y1, x2, y2
        return None

    def _load_image_as_bgr(self, image_key = "screenshot") -> np.ndarray:
        """Load screenshot from self.obs into a BGR numpy array for cv2 drawing.
        Returns a copy of the image sized to (height,width).
        """
        if not self.obs:
            # blank canvas
            img = np.zeros((self.height, self.width, 3), dtype=np.uint8) + 255
            return img

        screenshot: bytes = self.obs[image_key]

        # if bytes
        if isinstance(screenshot, (bytes, bytearray)):
            pil = Image.open(BytesIO(screenshot)).convert("RGB")
            arr = np.array(pil)
            # PIL gives RGB, convert to BGR
            img = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        else:
            img = np.zeros((self.height, self.width, 3), dtype=np.uint8) + 255

        # Resize/crop/pad to target dims
        img_h, img_w = img.shape[:2]
        if (img_w, img_h) != (self.width, self.height):
            img = cv2.resize(img, (self.width, self.height))
        return img

    def _encode_img_to_bytes(self, img_bgr) -> bytes:
        _, buf = cv2.imencode('.jpg', img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        return buf.tobytes()

    def _draw_feedback(self, coords_rel=None, bbox_rel=None, text: str = '') -> bytes:
        """Draw the last operation (point or bbox) and annotation text on the screenshot and return raw jpg bytes."""
        img = self._load_image_as_bgr()
        # draw bbox if provided
        if bbox_rel is not None:
            bb = self._to_abs_bbox(bbox_rel)
            if bb:
                x1, y1, x2, y2 = bb
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 3)
                cv2.putText(img, text, (x1, max(20, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        elif coords_rel is not None:
            x, y = self._to_abs_point(coords_rel)
            cv2.circle(img, (x, y), 12, (0, 0, 255), -1)
            cv2.putText(img, text, (max(10, x - 10), max(20, y - 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        else:
            cv2.putText(img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 128, 0), 2)

        return self._encode_img_to_bytes(img)

    @agent_action
    def find_element_by_text(self, text_query: str, region: Optional[Tuple[float, float, float, float]] = None,
                            fuzzy: bool = True, min_confidence: float = 0.5) -> Optional[Tuple[float, float]]:
        """
        Find an element by text using OCR and return its relative coordinates.
        This is useful when there are few text elements on screen and we want to locate one by its text.
        
        Args:
            text_query: Text to search for (e.g., "Save", "File Menu")
            region: Optional region to search within (x_rel, y_rel, w_rel, h_rel) in [0,1]
            fuzzy: If True, use fuzzy text matching
            min_confidence: Minimum OCR confidence threshold
        
        Returns:
            Tuple of (x_rel, y_rel) if found, None otherwise
        """
        from agents.ocr import find_text_elements, find_elements_in_region, get_relative_coords
        
        elements = self.get_ocr_elements()
        
        if not elements:
            logger.warning("No OCR elements found in screenshot")
            return None
        
        # Filter by region if specified
        if region:
            elements = find_elements_in_region(elements, region, self.width, self.height)
        
        # Find matching text elements
        matches = find_text_elements(elements, text_query, fuzzy=fuzzy, min_confidence=min_confidence)
        
        if not matches:
            logger.debug(f"No OCR elements matching '{text_query}' found")
            return None
        
        # Return the best match (highest confidence)
        best_match = matches[0]
        rel_coords = get_relative_coords(best_match, self.width, self.height)
        
        logger.info(f"Found '{text_query}' at relative coords {rel_coords} with confidence {best_match.confidence:.2f}")
        return rel_coords
    

    def _wrap_result(self, result: str, coords_rel=None, bbox_rel=None, text: str = '') -> 'LegacyACIResult':
        """Return a dict containing the execution result, a base64-encoded feedback image, and annotation text."""
        global feedback_renderer
        img_bytes = self._draw_feedback(coords_rel=coords_rel, bbox_rel=bbox_rel, text=text)

        # Publish a renderer callable to the module-level `feedback_renderer` so external
        # callers (e.g., cli_app) can render this same annotation on arbitrary screenshots.
        def _renderer(screenshot_input: bytes) -> bytes:
            """Render the same feedback (coords_rel/bbox_rel/text) onto the provided screenshot.

            Accepts PIL.Image, raw bytes, numpy array, or dict{'screenshot': ...}.
            Returns a PIL.Image with the annotation drawn.
            """

            # Use the instance helper to generate JPEG bytes with annotation
            try:
                jpeg_bytes = self.draw_feedback_bytes(screenshot_input, coords_rel=coords_rel, bbox_rel=bbox_rel, text=text)
                return jpeg_bytes
            except Exception:
                # Fallback: attempt to draw by temporarily setting obs and using _load_image_as_bgr/_draw_feedback
                prev_obs = getattr(self, "obs", None)
                try:
                    if isinstance(screenshot_input, Observation):
                        self.obs = screenshot_input
                    else:
                        self.obs = Observation(screenshot=screenshot_input)
                    bytes = self._draw_feedback(coords_rel=coords_rel, bbox_rel=bbox_rel, text=text)
                    return bytes
                finally:
                    try:
                        self.obs = prev_obs
                    except Exception:
                        self.obs = None

        # assign to module-level variable so cli_app can import and call it
        try:
            feedback_renderer = _renderer
            print("Assigned feedback_renderer callable for external use.")
        except Exception:
            pass

        return LegacyACIResult(result=result, feedback_image_bytes=img_bytes, annotation=text)

    # Public helpers to render feedback on an arbitrary screenshot (no side-effects)
    def draw_feedback_bytes(self, screenshot_input, coords_rel=None, bbox_rel=None, text: str = "") -> bytes:
        """Render feedback (circle/bbox + text) on a given screenshot and return a raw JPEG bytes.

        Args:
            screenshot_input: one of the supported screenshot formats (bytes, PIL.Image, numpy.ndarray, or a dict {'screenshot': ...}).
            coords_rel: optional relative point (x_rel,y_rel) in [0,1]
            bbox_rel: optional relative bbox (x,y,w,h) in [0,1]
            text: annotation text to draw near the operation

        Returns:
            raw bytes of the annotated image.

        Notes:
            - This method temporarily sets `self.obs` to the provided screenshot for rendering and restores the previous value on return.
            - It does not change any other state (best-effort).
        """
        prev_obs: Observation = getattr(self, "obs", None)
        try:
            # Accept either raw image payload or a wrapped dict
            if isinstance(screenshot_input, Observation):
                self.obs = screenshot_input
            else:
                self.obs = Observation(screenshot=screenshot_input)

            return self._draw_feedback(coords_rel=coords_rel, bbox_rel=bbox_rel, text=text)
        finally:
            # restore previous obs
            try:
                self.obs = prev_obs
            except Exception:
                self.obs = None

    # Common actions using relative coordinates (values in [0,1])
    @agent_action
    def click(self, rel_point: Tuple[float, float], num_clicks: int = 1, button_type: str = "left"):
        """MCP 接口：在相对坐标点执行点击。

        名称: LegacyACI.click

        参数:
            rel_point (tuple[float, float]): 必需，相对坐标 [x_rel, y_rel]，范围 0.0-1.0。
            num_clicks (int): 可选，点击次数，默认 1。
            button_type (str): 可选，按钮类型，'left'|'middle'|'right'，默认 'left'

        返回:
            dict: 包含以下字段的字典：
                - result (str): 可执行命令字符串（例如 pyautogui 调用）。
                - feedback_image_base64 (str): base64 编码的 JPEG，图像上标注了点击点和文本注释。
                - annotation (str): 简短文本描述，例如 "Clicked at (960,540)"。

        错误:
            ValueError: 当 rel_point 坐标不在 [0,1] 范围内时可抛出 invalid_coordinate。

        示例请求 (JSON-RPC):
            {"method": "LegacyACI.click", "params": {"rel_point": [0.5,0.5], "num_clicks": 1}}
        """
        x_abs, y_abs = self._to_abs_point(rel_point)
        cmd = f"import pyautogui; pyautogui.click({x_abs}, {y_abs}, clicks={num_clicks}, button={repr(button_type)})"
        text = f"clicked here last time"
        return self._wrap_result(cmd, coords_rel=rel_point, text=text)

    @agent_action
    def drag_and_drop(self, rel_start: Tuple[float, float], rel_end: Tuple[float, float], duration: float = 1.0):
        """MCP 接口：从起始相对点拖拽到结束相对点。

        名称: LegacyACI.drag_and_drop

        参数:
            rel_start (tuple[float,float]): 必需，起始相对坐标 [x_rel,y_rel]。
            rel_end (tuple[float,float]): 必需，结束相对坐标 [x_rel,y_rel]。
            duration (float): 可选，拖拽时长（秒），默认 1.0。

        返回:
            dict: {result, feedback_image_base64, annotation}，其中 feedback_image_base64 的图像会绘制覆盖拖拽区域的矩形。

        错误:
            ValueError: 当坐标不在 [0,1] 范围内时抛出 invalid_coordinate。

        示例:
            {"method":"LegacyACI.drag_and_drop","params":{"rel_start":[0.2,0.2],"rel_end":[0.8,0.5]}}
        """
        x1, y1 = self._to_abs_point(rel_start)
        x2, y2 = self._to_abs_point(rel_end)
        cmd = f"import pyautogui; pyautogui.moveTo({x1},{y1}); pyautogui.dragTo({x2},{y2}, duration={duration}, button='left'); pyautogui.mouseUp()"
        text = f"Dragged from ({x1},{y1}) to ({x2},{y2})"
        # create bbox covering the drag line
        bbox_rel = (min(rel_start[0], rel_end[0]), min(rel_start[1], rel_end[1]), abs(rel_end[0]-rel_start[0]), abs(rel_end[1]-rel_start[1]))
        return self._wrap_result(cmd, bbox_rel=bbox_rel, text=text)

    @agent_action
    def type(self, rel_point: Optional[Tuple[float, float]] = None, text_to_type: str = "", enter: bool = False):
        """MCP 接口：在可选相对位置点击后输入文本，并可选择回车。

        名称: LegacyACI.type

        参数:
            rel_point (tuple|None): 可选，相对坐标点；若提供会先点击该点。
            text_to_type (str): 要输入的文本（可包含 Unicode）。
            enter (bool): 是否在输入后按回车。

        返回:
            dict: 包含 result（命令字符串）、feedback_image_base64（标注点击点）和 annotation（已输入文本的简短描述）。

        示例:
            {"method":"LegacyACI.type","params":{"rel_point":[0.5,0.5],"text_to_type":"Hello","enter":true}}
        """
        cmd = "import pyautogui; "
        if rel_point is not None:
            x, y = self._to_abs_point(rel_point)
            cmd += f"pyautogui.click({x},{y}); "
        # naive handling for unicode: use clipboard approach would complicate; keep simple
        cmd += f"pyautogui.write({repr(text_to_type)}); "
        if enter:
            cmd += "pyautogui.press('enter'); "
        ann = f"Typed text: {text_to_type[:80]}"
        return self._wrap_result(cmd, coords_rel=rel_point, text=ann)

    @agent_action
    def scroll(self, rel_point: Tuple[float, float], clicks: int, horizontal: bool = False):
        """MCP 接口：在相对坐标处执行滚轮滚动。

        名称: LegacyACI.scroll

        参数:
            rel_point (tuple[float,float]): 必需，相对坐标点。
            clicks (int): 必需，滚动步数。正数/负数表示方向（依平台约定）。
            horizontal (bool): 可选，是否水平滚动，默认为 False（垂直）。

        返回:
            dict: 包含 result（命令字符串）、feedback_image_base64（标注滚动点）和 annotation（描述）。
        """
        x, y = self._to_abs_point(rel_point)
        if horizontal:
            cmd = f"import pyautogui; pyautogui.moveTo({x},{y}); pyautogui.hscroll({clicks})"
        else:
            cmd = f"import pyautogui; pyautogui.moveTo({x},{y}); pyautogui.vscroll({clicks})"
        ann = f"Scrolled at ({x},{y}) by {clicks}"
        return self._wrap_result(cmd, coords_rel=rel_point, text=ann)

    @agent_action
    def hotkey(self, keys: List[str], repeat: int = 1):
        """MCP 接口：按下组合键。

        名称: LegacyACI.hotkey

        参数:
            keys (list[str]): 必需，表示按键序列，例如 ["ctrl","c"]。
            repeat (int): 可选，表示需要重复按下的次数，例如.hotkey('left', 5) 会按下左箭头键 5 次。默认值为 1。

        返回:
            dict: {result, feedback_image_base64, annotation}。annotation 示例: "Hotkey pressed: ctrl+c"。

        示例:
            {"method":"LegacyACI.hotkey","params":{"keys":["ctrl","s"]}}
        """
        keys_quoted = [f"'{k}'" for k in keys]
        cmd = "import pyautogui;" + f"pyautogui.hotkey({', '.join(keys_quoted)});" * repeat
        ann = f"Hotkey pressed: {'+'.join(keys)} by {repeat} times"
        return self._wrap_result(cmd, text=ann)

    @agent_action
    def wait(self, seconds: float):
        """MCP 接口：等待指定秒数。

        名称: LegacyACI.wait

        参数:
            seconds (float): 必需，等待时长（秒），必须 >= 0。

        返回:
            dict: {result, feedback_image_base64, annotation}。
        """
        cmd = f"import time; time.sleep({seconds})"
        ann = f"Waited for {seconds} seconds"
        return self._wrap_result(cmd, text=ann)

    @agent_action
    def switch_applications(self, app_code: str):
        """MCP 接口：按名称切换到已有应用（占位实现）。

        名称: LegacyACI.switch_applications

        参数:
            app_code (str): 必需，目标应用的标识或名称。

        返回:
            dict: {result, feedback_image_base64, annotation}。result 为占位命令字符串，annotation 为简短描述。
        """
        # best-effort: no coordinate used, return simple annotation
        cmd = f"# switch_applications placeholder for {app_code}"
        ann = f"Switch to app: {app_code}"
        return self._wrap_result(cmd, text=ann)

    @agent_action
    def open(self, app_or_filename: str):
        """MCP 接口：打开应用或文件（占位）。

        名称: LegacyACI.open

        参数:
            app_or_filename (str): 必需，应用名或文件名。

        返回:
            dict: {result, feedback_image_base64, annotation}。
        """
        cmd = f"# open placeholder for {app_or_filename}"
        ann = f"Open: {app_or_filename}"
        return self._wrap_result(cmd, text=ann)

    @agent_action
    def done(self):
        """MCP 接口：标记任务完成（成功）。

        名称: LegacyACI.done

        参数: 无

        返回:
            dict: {result: "DONE", feedback_image_base64, annotation}
        """
        return self._wrap_result("DONE", text="Task completed")

    @agent_action
    def fail(self):
        """MCP 接口：标记任务失败。

        名称: LegacyACI.fail

        参数: 无

        返回:
            dict: {result: "FAIL", feedback_image_base64, annotation}
        """
        return self._wrap_result("FAIL", text="Task failed")
    
