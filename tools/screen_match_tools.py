import os
from functools import lru_cache

import cv2
import numpy as np
import pyautogui


def _normalize_region(region):
    if region is None:
        return None
    if not isinstance(region, (tuple, list)) or len(region) != 4:
        raise ValueError("region 必须是 (left, top, width, height)")
    left, top, width, height = [int(v) for v in region]
    if width <= 0 or height <= 0:
        raise ValueError("region 的 width 和 height 必须大于 0")
    return left, top, width, height

@lru_cache(maxsize=128)
def _load_template(image_path, grayscale=True):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"模板图片不存在: {image_path}")

    read_mode = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
    template = cv2.imread(image_path, read_mode)
    if template is None:
        raise ValueError(f"无法读取模板图片: {image_path}")
    return template

def _capture_screen(region=None, grayscale=True):
    screenshot = pyautogui.screenshot(region=region)
    frame = np.array(screenshot)

    if grayscale:
        return cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

def locate_center_on_screen_multiscale(
    image_path,
    confidence=0.8,
    return_immediately_confidence=0.99,
    scale_min=0.5,
    scale_max=1.8,
    scale_step=0.05,
    region=None,
    grayscale=True,
):
    """
    多尺度模板匹配，返回与 pyautogui.locateCenterOnScreen 类似的 Point 或 None。

    参数:
    - image_path: 模板图片路径
    - confidence: 匹配阈值 [0,1]
    - return_immediately_confidence: 如果匹配分数超过这个值，立即返回结果
    - scale_min/scale_max/scale_step: 模板缩放区间
    - region: 局部搜索区域 (left, top, width, height)
    - grayscale: 是否灰度匹配
    """
    region = _normalize_region(region)
    left_offset = region[0] if region else 0
    top_offset = region[1] if region else 0

    frame = _capture_screen(region=region, grayscale=grayscale)
    template = _load_template(image_path, grayscale=grayscale)

    frame_h, frame_w = frame.shape[:2]
    tpl_h, tpl_w = template.shape[:2]

    if tpl_h <= 0 or tpl_w <= 0:
        return None

    if scale_step <= 0:
        raise ValueError("scale_step 必须大于 0")

    scales = np.arange(scale_min, scale_max + 1e-9, scale_step)
    best_score = -1.0
    best_loc = None
    best_size = None

    for scale in scales:
        scaled_w = max(1, int(tpl_w * float(scale)))
        scaled_h = max(1, int(tpl_h * float(scale)))

        if scaled_w > frame_w or scaled_h > frame_h:
            continue

        resized = cv2.resize(template, (scaled_w, scaled_h), interpolation=cv2.INTER_AREA)
        result = cv2.matchTemplate(frame, resized, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)

        if max_val >= return_immediately_confidence:
            center_x = int(max_loc[0] + scaled_w / 2) + left_offset
            center_y = int(max_loc[1] + scaled_h / 2) + top_offset
            return pyautogui.Point(x=center_x, y=center_y)

        if max_val > best_score:
            best_score = float(max_val)
            best_loc = max_loc
            best_size = (scaled_w, scaled_h)

    if best_loc is None or best_score < confidence:
        return None

    center_x = int(best_loc[0] + best_size[0] / 2) + left_offset
    center_y = int(best_loc[1] + best_size[1] / 2) + top_offset

    return pyautogui.Point(x=center_x, y=center_y)


def locate_text_pytesseract(text, region=None, min_confidence=0.6):
    """
    使用 pytesseract 定位屏幕上的文本，返回文本中心坐标或 None。

    参数:
    - text: 要搜索的文本字符串
    - region: 局部搜索区域 (left, top, width, height)
    - min_confidence: 最小置信度阈值 [0,1]
    """
    try:
        import pytesseract
    except ImportError:
        raise ImportError("请安装 pytesseract 库以使用文本定位功能")

    region = _normalize_region(region)
    left_offset = region[0] if region else 0
    top_offset = region[1] if region else 0

    frame = _capture_screen(region=region, grayscale=True)
    data = pytesseract.image_to_data(frame, output_type=pytesseract.Output.DICT)

    for i in range(len(data['text'])):
        conf = float(data['conf'][i])
        if conf >= min_confidence and data['text'][i].strip() == text:
            x = int(data['left'][i]) + int(data['width'][i] / 2) + left_offset
            y = int(data['top'][i]) + int(data['height'][i] / 2) + top_offset
            return pyautogui.Point(x=x, y=y)

    return None