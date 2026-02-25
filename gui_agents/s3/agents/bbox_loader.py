"""
Module for loading and processing bounding box annotations from guitools.
Integrates GUI element metadata into agent observations.
"""

import json
import os
from typing import Dict, List, Optional, Any


class BBoxAnnotationLoader:
    """Load and manage bounding box annotations from JSON files."""

    def __init__(self):
        self.annotations: List[Dict] = []
        self.window_info: Optional[Dict] = None

    def load_from_file(self, filepath: str) -> bool:
        """
        Load annotations from a JSON file exported by bbox_labeler.
        
        Args:
            filepath: Path to the JSON file
            
        Returns:
            True if loaded successfully, False otherwise
        """
        if not os.path.exists(filepath):
            print(f"⚠️  Annotation file not found: {filepath}")
            return False

        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)

            self.window_info = data.get("window")
            self.annotations = data.get("annotations", [])
            print(f"✅ Loaded {len(self.annotations)} annotations from {filepath}")
            return True
        except Exception as e:
            print(f"❌ Failed to load annotations: {e}")
            return False

    def get_annotations(self) -> List[Dict]:
        """Get all loaded annotations."""
        return list(self.annotations)

    def get_annotation_text(self) -> str:
        """
        Convert annotations to human-readable text format.
        Useful for including in LLM prompts.
        """
        if not self.annotations:
            return "No UI elements annotated."

        lines = ["Available UI Elements:"]
        for i, ann in enumerate(self.annotations, 1):
            name = ann.get("name", "unknown")
            x = ann.get("x", 0)
            y = ann.get("y", 0)
            w = ann.get("width", 0)
            h = ann.get("height", 0)
            lines.append(f"  {i}. {name}: bbox=({x}, {y}, {x+w}, {y+h}), size={w}x{h}")

        return "\n".join(lines)

    def get_annotation_dict(self) -> Dict[str, Any]:
        """
        Get annotations in a structured dictionary format.
        Suitable for including in observation dictionaries.
        """
        return {
            "window": self.window_info,
            "annotations": self.annotations,
            "annotation_count": len(self.annotations),
        }

    def clear(self):
        """Clear all loaded annotations."""
        self.annotations.clear()
        self.window_info = None


class BBoxFileWatcher:
    """Monitor a directory for new annotation JSON files."""

    def __init__(self, watch_dir: str = "gui_annotations"):
        self.watch_dir = watch_dir
        self.last_file: Optional[str] = None

    def get_latest_annotation(self) -> Optional[Dict]:
        """
        Get the latest annotation JSON file from the watch directory.
        
        Returns:
            Annotation data dict or None if no file found
        """
        if not os.path.exists(self.watch_dir):
            return None

        json_files = [
            f for f in os.listdir(self.watch_dir) 
            if f.endswith(".json")
        ]

        if not json_files:
            return None

        # Sort by modification time, get the most recent
        json_files.sort(
            key=lambda f: os.path.getmtime(os.path.join(self.watch_dir, f)),
            reverse=True
        )

        latest = json_files[0]
        latest_path = os.path.join(self.watch_dir, latest)

        # Check if it's a new file
        if latest_path == self.last_file:
            return None

        self.last_file = latest_path

        try:
            with open(latest_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            print(f"📂 Detected new annotation file: {latest}")
            return data
        except Exception as e:
            print(f"❌ Failed to read annotation file {latest}: {e}")
            return None


def format_annotations_for_prompt(annotations: List[Dict], annotation_image_path: str = "D:\\agents\\guitools\\annotations-annotated.png") -> str:
    """
    Format bounding box annotations for inclusion in LLM prompts.
    Directs the LLM to reference the annotated image for UI element locations.
    
    Args:
        annotations: List of annotation dicts with name, x, y, width, height
        annotation_image_path: Path to the annotated image file showing all UI elements
        
    Returns:
        Formatted string directing to the annotated image for UI element reference
    """
    if not annotations:
        return ""

    prompt_lines = [
        "\n======== UI ELEMENT GUIDANCE ========",
        "IMPORTANT: The screenshot you receive contains TWO IMAGES SIDE BY SIDE:",
        "",
        "LEFT SIDE  📋 | UI Element Instructions (Annotation Image)",
        "RIGHT SIDE 📸 | Current Screenshot (What you see on screen now)",
        "",
        "HOW TO USE:",
        "1. Look at the LEFT image (📋) to see where all UI elements are located and their names",
        "2. Look at the RIGHT image (📸) to see the current state of the desktop",
        "3. Match the element names and positions from the left image to interact with the right image",
        "",
        f"Total UI elements marked in the instruction image: {len(annotations)}",
        "",
        "Available UI element names:",
    ]

    # List only element names, coordinates are shown in the image
    for i, ann in enumerate(annotations, 1):
        name = ann.get("name", "?")
        prompt_lines.append(f"  {i}. {name}")

    prompt_lines.append("======================================\n")
    return "\n".join(prompt_lines)
