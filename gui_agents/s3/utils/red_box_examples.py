"""
Simple usage examples for red_box_detector module.

Run this file to see example outputs:
    python -m gui_agents.s3.utils.red_box_examples
"""

from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw

from red_box_detector import (
    detect_red_boxes,
    extract_box_content,
    visualize_detections,
    find_largest_red_box,
)


def create_test_image_with_red_box(
    size: tuple = (800, 600),
    box_rect: tuple = (100, 100, 400, 300),
    box_color: str = "red",
    box_width: int = 5,
) -> Image.Image:
    """
    Create a test image with a red rectangular box.
    
    Args:
        size: Image size as (width, height).
        box_rect: Box as (x1, y1, x2, y2).
        box_color: Color name or RGB tuple.
        box_width: Line width in pixels.
    
    Returns:
        PIL Image with red box drawn.
    """
    img = Image.new("RGB", size, color="white")
    draw = ImageDraw.Draw(img)
    
    # Draw some background content
    draw.text((50, 50), "Test Screenshot", fill="black")
    draw.ellipse((500, 200, 700, 400), outline="blue", width=2)
    
    # Draw red box
    draw.rectangle(box_rect, outline=box_color, width=box_width)
    
    return img


def example_basic_detection():
    """Example 1: Basic red box detection."""
    print("=" * 60)
    print("Example 1: Basic Red Box Detection")
    print("=" * 60)
    
    # Create test image
    test_img = create_test_image_with_red_box(
        size=(800, 600),
        box_rect=(150, 100, 650, 450),
        box_width=4,
    )
    
    # Detect red boxes
    boxes = detect_red_boxes(test_img, min_area=500)
    
    print(f"Found {len(boxes)} red box(es)")
    for idx, box in enumerate(boxes):
        print(f"\nBox #{idx}:")
        print(f"  Position: ({box.x}, {box.y})")
        print(f"  Size: {box.width} x {box.height}")
        print(f"  Area: {box.area} pixels")
        print(f"  Confidence: {box.confidence:.2f}")
        print(f"  BBox: {box.bbox}")
        print(f"  Center: {box.center}")
    
    return test_img, boxes


def example_extract_content():
    """Example 2: Extract box content."""
    print("\n" + "=" * 60)
    print("Example 2: Extract Box Content")
    print("=" * 60)
    
    # Create test image with content inside box
    test_img = create_test_image_with_red_box(
        size=(800, 600),
        box_rect=(200, 150, 600, 400),
        box_width=3,
    )
    
    # Add content inside the box
    draw = ImageDraw.Draw(test_img)
    draw.text((220, 170), "Content inside red box", fill="black")
    draw.ellipse((250, 220, 550, 380), fill="lightblue", outline="darkblue", width=2)
    
    # Detect box
    box = find_largest_red_box(test_img, min_area=500)
    
    if box:
        print(f"Detected box at ({box.x}, {box.y}) with size {box.width}x{box.height}")
        
        # Extract content
        cropped = extract_box_content(test_img, box, padding=0)
        print(f"Extracted image size: {cropped.size}")
        
        # Extract with padding
        cropped_padded = extract_box_content(test_img, box, padding=10)
        print(f"Extracted with padding: {cropped_padded.size}")
        
        return test_img, box, cropped
    else:
        print("No red box detected!")
        return None, None, None


def example_multiple_boxes():
    """Example 3: Detect multiple red boxes."""
    print("\n" + "=" * 60)
    print("Example 3: Multiple Red Boxes")
    print("=" * 60)
    
    # Create image with multiple boxes
    img = Image.new("RGB", (1000, 700), color="white")
    draw = ImageDraw.Draw(img)
    
    # Draw multiple red boxes
    boxes_to_draw = [
        (100, 100, 300, 250),
        (400, 150, 700, 350),
        (150, 400, 400, 600),
        (600, 450, 900, 650),
    ]
    
    for box_rect in boxes_to_draw:
        draw.rectangle(box_rect, outline="red", width=4)
        # Add some content
        cx = (box_rect[0] + box_rect[2]) // 2
        cy = (box_rect[1] + box_rect[3]) // 2
        draw.text((cx - 20, cy), f"Box", fill="black")
    
    # Detect all boxes
    detected = detect_red_boxes(img, min_area=500)
    
    print(f"Drew {len(boxes_to_draw)} boxes, detected {len(detected)} boxes")
    
    for idx, box in enumerate(detected):
        print(f"Box #{idx}: {box.width}x{box.height} at ({box.x}, {box.y}) - area: {box.area}")
    
    # Visualize
    viz = visualize_detections(img, detected, show_labels=True)
    
    return img, detected, viz


def example_tolerance_tuning():
    """Example 4: Adjust detection parameters for different red shades."""
    print("\n" + "=" * 60)
    print("Example 4: Tolerance Tuning")
    print("=" * 60)
    
    # Create image with different red shades
    img = Image.new("RGB", (900, 300), color="white")
    draw = ImageDraw.Draw(img)
    
    red_shades = [
        ((255, 0, 0), "Pure Red"),
        ((200, 50, 50), "Light Red"),
        ((150, 0, 0), "Dark Red"),
    ]
    
    x_offset = 50
    for (color, name) in red_shades:
        draw.rectangle((x_offset, 50, x_offset + 200, 250), outline=color, width=5)
        draw.text((x_offset + 50, 20), name, fill="black")
        x_offset += 280
    
    # Detect with default parameters
    boxes_default = detect_red_boxes(img, min_area=500)
    print(f"Default parameters detected: {len(boxes_default)} boxes")
    
    # Detect with relaxed parameters (lower saturation threshold)
    boxes_relaxed = detect_red_boxes(
        img,
        min_area=500,
        red_lower_1=(0, 40, 30),  # Lower saturation/value thresholds
        red_lower_2=(170, 40, 30),
    )
    print(f"Relaxed parameters detected: {len(boxes_relaxed)} boxes")
    
    return img, boxes_default, boxes_relaxed


def example_save_and_load():
    """Example 5: Save and load images."""
    print("\n" + "=" * 60)
    print("Example 5: File I/O")
    print("=" * 60)
    
    # Create and save test image
    test_img = create_test_image_with_red_box()
    temp_path = Path("temp_test_screenshot.png")
    test_img.save(temp_path)
    print(f"Saved test image to {temp_path}")
    
    # Load and detect from file path
    boxes = detect_red_boxes(str(temp_path), min_area=500)
    print(f"Detected {len(boxes)} box(es) from file")
    
    if boxes:
        # Extract and save cropped content
        cropped = extract_box_content(str(temp_path), boxes[0])
        crop_path = Path("temp_extracted_content.png")
        cropped.save(crop_path)
        print(f"Saved extracted content to {crop_path}")
        
        # Save visualization
        viz = visualize_detections(str(temp_path), boxes)
        viz_path = Path("temp_visualization.png")
        viz.save(viz_path)
        print(f"Saved visualization to {viz_path}")
    
    print("\nNote: Temporary files created (temp_*.png)")
    print("You can delete them manually if needed.")
    
    return temp_path


def run_all_examples():
    """Run all examples."""
    print("\n" + "=" * 60)
    print("RED BOX DETECTOR - Usage Examples")
    print("=" * 60 + "\n")
    
    example_basic_detection()
    example_extract_content()
    example_multiple_boxes()
    example_tolerance_tuning()
    example_save_and_load()
    
    print("\n" + "=" * 60)
    print("All examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    run_all_examples()
