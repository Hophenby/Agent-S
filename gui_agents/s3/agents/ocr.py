"""
OCR module for extracting axis information from chart images.
Ported from gbline's GbFigureTools.py with enhancements.
"""

import re
import os
import logging
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict, Any
import numpy as np
import cv2

# Configure logging
logger = logging.getLogger(__name__)

# Global OCR instance (lazy initialization)
_ocr_instance = None


def get_ocr_instance():
    """Get or create the PaddleOCR instance."""
    global _ocr_instance
    if _ocr_instance is None:
        try:
            from paddleocr import PaddleOCR
            
            # Get model path from environment or use default
            model_path = os.getenv("OCR_MODEL_PATH", r"E:\qqfiles\1094009969\FileRecv\gbline-a\gbline-1\ocr_model")
            
            _ocr_instance = PaddleOCR(
                use_angle_cls=True,
                use_gpu=False,  # Set to True if GPU is available
                det_model_dir=os.path.join(model_path, 'en_PP-OCRv3_det_infer'),
                rec_model_dir=os.path.join(model_path, 'en_PP-OCRv3_rec_infer'),
                cls_model_dir=os.path.join(model_path, 'ch_ppocr_mobile_v2.0_cls_infer'),
                show_log=False
            )
            logger.info("PaddleOCR initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize PaddleOCR: {e}")
            raise
    
    return _ocr_instance


@dataclass
class OCRElement:
    """Represents a detected text element with visual features."""
    text: str
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2) absolute coordinates
    confidence: float
    center: Tuple[int, int] = field(init=False)
    area: int = field(init=False)
    
    def __post_init__(self):
        x1, y1, x2, y2 = self.bbox
        self.center = ((x1 + x2) // 2, (y1 + y2) // 2)
        self.area = (x2 - x1) * (y2 - y1)
    
    def contains_point(self, x: int, y: int) -> bool:
        """Check if a point is inside this element's bounding box."""
        x1, y1, x2, y2 = self.bbox
        return x1 <= x <= x2 and y1 <= y <= y2
    
    def distance_to(self, x: int, y: int) -> float:
        """Calculate distance from a point to this element's center."""
        cx, cy = self.center
        return np.sqrt((cx - x) ** 2 + (cy - y) ** 2)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the OCRElement to a dictionary."""
        return {
            "text": self.text,
            "bbox": self.bbox,
            "confidence": self.confidence,
            "center": self.center,
            "area": self.area,
        }


def extract_ocr_elements(image: np.ndarray, ocr_instance=None) -> List[OCRElement]:
    """Extract all text elements from an image using PaddleOCR.
    
    Args:
        image: Input image as numpy array (RGB or BGR)
        ocr_instance: Optional PaddleOCR instance (will create if None)
    
    Returns:
        List of OCRElement objects containing detected text and locations
    """
    if ocr_instance is None:
        ocr_instance = get_ocr_instance()
    
    try:
        # PaddleOCR expects RGB or BGR
        results = ocr_instance.ocr(image, cls=True)
        
        if not results or not results[0]:
            logger.debug("No OCR results found")
            return []
        
        elements = []
        for line in results[0]:
            bbox_points = line[0]  # [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
            text_info = line[1]    # (text, confidence)
            
            text = text_info[0]
            confidence = text_info[1]
            
            # Convert polygon to bounding box (x1, y1, x2, y2)
            xs = [p[0] for p in bbox_points]
            ys = [p[1] for p in bbox_points]
            x1, y1 = int(min(xs)), int(min(ys))
            x2, y2 = int(max(xs)), int(max(ys))
            
            if text.strip():  # Only keep non-empty text
                elements.append(OCRElement(
                    text=text.strip(),
                    bbox=(x1, y1, x2, y2),
                    confidence=confidence
                ))

        
        logger.debug(f"Extracted {len(elements)} OCR elements")
        return elements
        
    except Exception as e:
        logger.error(f"OCR extraction failed: {e}")
        return []


def find_text_elements(elements: List[OCRElement], query: str, 
                       fuzzy: bool = True, min_confidence: float = 0.5) -> List[OCRElement]:
    """Find OCR elements matching a text query.
    
    Args:
        elements: List of OCRElement objects
        query: Text to search for
        fuzzy: If True, use fuzzy matching (case-insensitive substring)
        min_confidence: Minimum OCR confidence threshold
    
    Returns:
        List of matching OCRElement objects, sorted by confidence
    """
    query_lower = query.lower()
    matches = []
    
    for elem in elements:
        if elem.confidence < min_confidence:
            continue
        
        elem_text_lower = elem.text.lower()
        
        if fuzzy:
            # Fuzzy matching: substring or similar
            if query_lower in elem_text_lower or elem_text_lower in query_lower:
                matches.append(elem)
            # Also check for word-level matching
            elif any(word in elem_text_lower for word in query_lower.split()):
                matches.append(elem)
        else:
            # Exact matching (case-insensitive)
            if query_lower == elem_text_lower:
                matches.append(elem)
    
    # Sort by confidence (highest first)
    matches.sort(key=lambda e: e.confidence, reverse=True)
    return matches


def find_elements_in_region(elements: List[OCRElement], 
                           region: Tuple[float, float, float, float],
                           image_width: int, image_height: int) -> List[OCRElement]:
    """Filter OCR elements within a specific region.
    
    Args:
        elements: List of OCRElement objects
        region: Relative region (x_rel, y_rel, w_rel, h_rel) in [0,1]
        image_width: Image width in pixels
        image_height: Image height in pixels
    
    Returns:
        List of OCRElement objects within the region
    """
    x_rel, y_rel, w_rel, h_rel = region
    x1 = int(x_rel * image_width)
    y1 = int(y_rel * image_height)
    x2 = int((x_rel + w_rel) * image_width)
    y2 = int((y_rel + h_rel) * image_height)
    
    filtered = []
    for elem in elements:
        ex1, ey1, ex2, ey2 = elem.bbox
        # Check if element center is within region
        cx, cy = elem.center
        if x1 <= cx <= x2 and y1 <= cy <= y2:
            filtered.append(elem)
    
    return filtered


def get_relative_coords(element: OCRElement, image_width: int, image_height: int) -> Tuple[float, float]:
    """Get relative coordinates (0-1) of an OCR element's center.
    
    Args:
        element: OCRElement object
        image_width: Image width in pixels
        image_height: Image height in pixels
    
    Returns:
        Tuple of (x_rel, y_rel) in [0,1]
    """
    cx, cy = element.center
    return (cx / image_width, cy / image_height)


def find_nearby_elements(elements: List[OCRElement], anchor_elem: OCRElement,
                        direction: str = "right", max_distance: int = 200) -> List[OCRElement]:
    """Find elements near an anchor element in a specific direction.
    
    Args:
        elements: List of OCRElement objects
        anchor_elem: Reference element
        direction: 'left', 'right', 'above', 'below'
        max_distance: Maximum distance in pixels
    
    Returns:
        List of nearby elements sorted by distance
    """
    ax, ay = anchor_elem.center
    nearby = []
    
    for elem in elements:
        if elem == anchor_elem:
            continue
        
        ex, ey = elem.center
        dist = anchor_elem.distance_to(ex, ey)
        
        if dist > max_distance:
            continue
        
        # Check direction
        if direction == "right" and ex > ax:
            nearby.append((dist, elem))
        elif direction == "left" and ex < ax:
            nearby.append((dist, elem))
        elif direction == "above" and ey < ay:
            nearby.append((dist, elem))
        elif direction == "below" and ey > ay:
            nearby.append((dist, elem))
    
    # Sort by distance and return elements only
    nearby.sort(key=lambda x: x[0])
    return [elem for _, elem in nearby]
