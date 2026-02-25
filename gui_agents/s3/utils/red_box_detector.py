"""
Detector for locating red-colored rectangular boxes in screenshots and extracting regions.

Example
-------
from gui_agents.s3.utils.red_box_detector import detect_red_boxes, extract_box_content

boxes = detect_red_boxes("/path/screenshot.png")
if boxes:
    print(f"Found {len(boxes)} red boxes")
    cropped = extract_box_content("/path/screenshot.png", boxes[0])
    cropped.save("/path/extracted.png")

Notes
-----
- Red detection uses HSV color space for robustness to lighting variations.
- Multiple boxes can be detected and sorted by area (largest first).
- Requires OpenCV (cv2) to be installed.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
from PIL import Image

try:
    import cv2
except ImportError:
    raise ImportError(
        "OpenCV (cv2) is required for red_box_detector. Install with: pip install opencv-python"
    )


ImageLike = Union[str, Path, Image.Image, np.ndarray]


@dataclass
class RedBox:
    """Represents a detected red rectangular box."""
    
    x: int
    y: int
    width: int
    height: int
    area: int
    confidence: float  # Ratio of red pixels in the detected contour
    
    @property
    def bbox(self) -> Tuple[int, int, int, int]:
        """Return bounding box as (x1, y1, x2, y2)."""
        return self.x, self.y, self.x + self.width, self.y + self.height
    
    @property
    def center(self) -> Tuple[int, int]:
        """Return center point as (cx, cy)."""
        return self.x + self.width // 2, self.y + self.height // 2


def _to_cv_image(img: ImageLike) -> np.ndarray:
    """Convert various image formats to OpenCV BGR format."""
    if isinstance(img, (str, Path)):
        # OpenCV loads as BGR by default
        return cv2.imread(str(img))
    
    if isinstance(img, Image.Image):
        # Convert PIL to RGB then to BGR
        rgb = np.array(img.convert("RGB"))
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    
    if isinstance(img, np.ndarray):
        # Assume already in BGR if 3 channels
        if img.ndim == 2:
            # Grayscale to BGR
            return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.shape[2] == 4:
            # RGBA to BGR
            return cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
        return img
    
    raise TypeError(f"Unsupported image type: {type(img)}")


def detect_red_boxes(
    image: ImageLike,
    *,
    min_area: int = 100,
    max_area: Optional[int] = None,
    red_lower_1: Tuple[int, int, int] = (0, 70, 50),
    red_upper_1: Tuple[int, int, int] = (10, 255, 255),
    red_lower_2: Tuple[int, int, int] = (170, 70, 50),
    red_upper_2: Tuple[int, int, int] = (180, 255, 255),
    morphology_iterations: int = 2,
    min_aspect_ratio: float = 0.2,
    max_aspect_ratio: float = 5.0,
) -> List[RedBox]:
    """
    Detect red rectangular boxes in an image using HSV color space and contour detection.
    
    Args:
        image: Input image (path, PIL Image, or numpy array).
        min_area: Minimum box area in pixels to consider valid.
        max_area: Maximum box area in pixels (None for no limit).
        red_lower_1: Lower HSV bound for red (first range, around 0°).
        red_upper_1: Upper HSV bound for red (first range).
        red_lower_2: Lower HSV bound for red (second range, around 180°).
        red_upper_2: Upper HSV bound for red (second range).
        morphology_iterations: Number of morphological operations for noise reduction.
        min_aspect_ratio: Minimum width/height or height/width ratio.
        max_aspect_ratio: Maximum width/height or height/width ratio.
    
    Returns:
        List of RedBox objects, sorted by area (largest first).
    
    Notes:
        Red color wraps around in HSV (0° and 180° are both red), so two ranges are needed.
    """
    
    # Load and convert to BGR
    bgr_image = _to_cv_image(image)
    if bgr_image is None:
        return []
    
    # Convert to HSV for better color detection
    hsv_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)
    
    # Create masks for red color (two ranges due to hue wrap-around)
    mask1 = cv2.inRange(hsv_image, np.array(red_lower_1), np.array(red_upper_1))
    mask2 = cv2.inRange(hsv_image, np.array(red_lower_2), np.array(red_upper_2))
    red_mask = cv2.bitwise_or(mask1, mask2)
    
    # Morphological operations to reduce noise and connect broken lines
    kernel = np.ones((3, 3), np.uint8)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel, iterations=morphology_iterations)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # Find contours
    contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    detected_boxes = []
    img_height, img_width = bgr_image.shape[:2]
    
    for contour in contours:
        area = cv2.contourArea(contour)
        
        # Filter by area
        if area < min_area:
            continue
        if max_area and area > max_area:
            continue
        
        # Get bounding rectangle
        x, y, w, h = cv2.boundingRect(contour)
        
        # Filter by aspect ratio (reject very elongated shapes)
        aspect_ratio = max(w, h) / max(min(w, h), 1)
        if aspect_ratio < min_aspect_ratio or aspect_ratio > max_aspect_ratio:
            continue
        
        # Calculate confidence as the ratio of red pixels inside the bounding box
        roi_mask = red_mask[y:y+h, x:x+w]
        red_pixel_count = cv2.countNonZero(roi_mask)
        perimeter = 2 * (w + h)
        confidence = min(red_pixel_count / max(perimeter, 1), 1.0)
        
        detected_boxes.append(
            RedBox(
                x=int(x),
                y=int(y),
                width=int(w),
                height=int(h),
                area=int(area),
                confidence=float(confidence),
            )
        )
    
    # Sort by area (largest first)
    detected_boxes.sort(key=lambda box: box.area, reverse=True)
    
    return detected_boxes


def extract_box_content(
    image: ImageLike,
    box: RedBox,
    *,
    padding: int = 0,
    return_type: str = "pil",
) -> Union[Image.Image, np.ndarray]:
    """
    Extract the content inside a detected red box.
    
    Args:
        image: Input image (path, PIL Image, or numpy array).
        box: RedBox object specifying the region to extract.
        padding: Number of pixels to add around the box (can be negative to shrink).
        return_type: "pil" for PIL Image, "numpy" for numpy array (BGR format).
    
    Returns:
        Extracted image region as PIL Image or numpy array.
    """
    
    bgr_image = _to_cv_image(image)
    img_height, img_width = bgr_image.shape[:2]
    
    # Apply padding and clamp to image bounds
    x1 = max(0, box.x - padding)
    y1 = max(0, box.y - padding)
    x2 = min(img_width, box.x + box.width + padding)
    y2 = min(img_height, box.y + box.height + padding)
    
    # Extract region
    cropped = bgr_image[y1:y2, x1:x2]
    
    if return_type == "pil":
        # Convert BGR to RGB for PIL
        rgb_cropped = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb_cropped)
    elif return_type == "numpy":
        return cropped
    else:
        raise ValueError(f"Invalid return_type: {return_type}. Use 'pil' or 'numpy'.")


def visualize_detections(
    image: ImageLike,
    boxes: List[RedBox],
    *,
    box_color: Tuple[int, int, int] = (0, 255, 0),  # Green in BGR
    box_thickness: int = 2,
    show_labels: bool = True,
    label_color: Tuple[int, int, int] = (0, 255, 0),
) -> Image.Image:
    """
    Draw detected boxes on the image for visualization.
    
    Args:
        image: Input image (path, PIL Image, or numpy array).
        boxes: List of RedBox objects to visualize.
        box_color: Color for bounding boxes in BGR format.
        box_thickness: Thickness of bounding box lines.
        show_labels: Whether to show box index labels.
        label_color: Color for labels in BGR format.
    
    Returns:
        PIL Image with boxes drawn.
    """
    
    bgr_image = _to_cv_image(image).copy()
    
    for idx, box in enumerate(boxes):
        # Draw rectangle
        cv2.rectangle(
            bgr_image,
            (box.x, box.y),
            (box.x + box.width, box.y + box.height),
            box_color,
            box_thickness,
        )
        
        # Draw label
        if show_labels:
            label = f"#{idx} ({box.width}x{box.height})"
            label_pos = (box.x, max(box.y - 5, 10))
            cv2.putText(
                bgr_image,
                label,
                label_pos,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                label_color,
                1,
                cv2.LINE_AA,
            )
    
    # Convert BGR to RGB for PIL
    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb_image)


def find_largest_red_box(image: ImageLike, **kwargs) -> Optional[RedBox]:
    """
    Convenience function to find the largest red box in an image.
    
    Args:
        image: Input image (path, PIL Image, or numpy array).
        **kwargs: Additional arguments passed to detect_red_boxes.
    
    Returns:
        The largest RedBox or None if no boxes found.
    """
    boxes = detect_red_boxes(image, **kwargs)
    return boxes[0] if boxes else None
