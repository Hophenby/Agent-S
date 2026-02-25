from .patch_locator import MatchResult, draw_match_box, locate_patch
from .red_box_detector import (
    RedBox,
    detect_red_boxes,
    extract_box_content,
    find_largest_red_box,
    visualize_detections,
)

__all__ = [
    "MatchResult",
    "locate_patch",
    "draw_match_box",
    "RedBox",
    "detect_red_boxes",
    "extract_box_content",
    "find_largest_red_box",
    "visualize_detections",
]
