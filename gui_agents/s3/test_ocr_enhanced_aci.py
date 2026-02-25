"""
Test script demonstrating OCR-enhanced element matching for LegacyACI.

This script shows how to use the new OCR capabilities to find and interact
with GUI elements by text, improving accuracy for specific applications.
"""

import sys
import os
import logging
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from agents.grounding import LegacyACI
from PIL import Image
import pyautogui

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

logger = logging.getLogger(__name__)


def test_ocr_element_extraction():
    """Test 1: Extract and list all visible text elements"""
    print("=" * 50)
    print("Test 1: OCR Element Extraction")
    print("=" * 50)
    
    # Initialize ACI
    screen_width, screen_height = pyautogui.size()
    aci = LegacyACI(width=screen_width, height=screen_height)
    
    # Take a screenshot
    screenshot = pyautogui.screenshot()
    aci.assign_screenshot({"screenshot": screenshot, "original_screenshot": screenshot})
    
    # Extract OCR elements
    elements = aci.get_ocr_elements()
    # print(elements)
    
    print(f"\nFound {len(elements)} text elements on screen:")
    for i, elem in enumerate(elements[:10]):  # Show first 10
        print(f"  {i+1}. '{elem.text}' at {elem.center} (confidence: {elem.confidence:.2f})")
    
    if len(elements) > 10:
        print(f"  ... and {len(elements) - 10} more")
    
    return aci, elements


def test_find_by_text(aci):
    """Test 2: Find specific elements by text"""
    print("\n" + "=" * 50)
    print("Test 2: Find Elements by Text")
    print("=" * 50)
    
    # Example searches - adjust these based on what's visible on your screen
    search_queries = [
        "File",
        "Edit", 
        "View",
        "Save",
        "References",
        "Reference",
    ]
    
    for query in search_queries:
        coords = aci.find_element_by_text(query, fuzzy=True)
        if coords:
            print(f"✓ Found '{query}' at relative position {coords}")
        else:
            print(f"✗ '{query}' not found")


def test_list_visible_texts(aci):
    """Test 3: List all visible texts in different regions"""
    print("\n" + "=" * 50)
    print("Test 3: List Visible Texts by Region")
    print("=" * 50)
    
    # Define regions (relative coordinates)
    regions = {
        "Top Bar": (0.0, 0.0, 1.0, 0.1),      # Top 10% of screen
        "Left Panel": (0.0, 0.0, 0.2, 1.0),   # Left 20% of screen
        "Center": (0.3, 0.3, 0.4, 0.4),       # Center region
    }
    
    for region_name, region_coords in regions.items():
        texts = aci.list_visible_texts(region=region_coords, min_confidence=0.6)
        print(f"\n{region_name} region contains {len(texts)} text elements:")
        for item in texts[:5]:  # Show first 5
            print(f"  - '{item['text']}' (conf: {item['confidence']:.2f})")
        if len(texts) > 5:
            print(f"  ... and {len(texts) - 5} more")


def test_click_by_text(aci):
    """Test 4: Simulate clicking elements by text (dry run)"""
    print("\n" + "=" * 50)
    print("Test 4: Click by Text (Dry Run)")
    print("=" * 50)
    
    # Example: Try to find and prepare to click "File" menu
    # Note: This won't actually click, just shows the command that would be executed
    
    test_queries = ["File", "Edit", "Save"]
    
    for query in test_queries:
        print(f"\nAttempting to click '{query}'...")
        result = aci.click_by_text(query, fuzzy=True, min_confidence=0.6)
        
        if result and result.get('result'):
            print(f"  Would execute: {result['result'][:100]}...")
            print(f"  Annotation: {result.get('annotation', 'N/A')}")
        else:
            print(f"  Could not locate '{query}'")


def test_app_specific_workflow():
    """Test 5: Demonstrate application-specific workflow"""
    print("\n" + "=" * 50)
    print("Test 5: Application-Specific Workflow Example")
    print("=" * 50)
    
    screen_width, screen_height = pyautogui.size()
    aci = LegacyACI(width=screen_width, height=screen_height)
    
    # Take fresh screenshot
    screenshot = pyautogui.screenshot()
    aci.assign_screenshot({"screenshot": screenshot})
    
    # Example: Workflow for a text editor
    print("\nSimulating text editor workflow:")
    
    workflow_steps = [
        ("File", "Find File menu"),
        ("Save", "Find Save button"),
        ("Edit", "Find Edit menu"),
        ("Undo", "Find Undo option"),
    ]
    
    for text_query, description in workflow_steps:
        print(f"\nStep: {description}")
        coords = aci.find_element_by_text(text_query, fuzzy=True, min_confidence=0.5)
        
        if coords:
            print(f"  ✓ Located '{text_query}' at {coords}")
            # In real usage, you would call: aci.click(coords)
        else:
            print(f"  ✗ Could not find '{text_query}' - would try alternative method")


def demonstrate_ocr_caching():
    """Test 6: Demonstrate OCR caching performance"""
    print("\n" + "=" * 50)
    print("Test 6: OCR Caching Performance")
    print("=" * 50)
    
    import time
    
    screen_width, screen_height = pyautogui.size()
    aci = LegacyACI(width=screen_width, height=screen_height)
    
    screenshot = pyautogui.screenshot()
    aci.assign_screenshot({"screenshot": screenshot})
    
    # First extraction (no cache)
    start = time.time()
    elements1 = aci.get_ocr_elements(force_refresh=True)
    time1 = time.time() - start
    print(f"First OCR extraction: {len(elements1)} elements in {time1:.3f}s")
    
    # Second extraction (with cache)
    start = time.time()
    elements2 = aci.get_ocr_elements(force_refresh=False)
    time2 = time.time() - start
    print(f"Cached OCR retrieval: {len(elements2)} elements in {time2:.3f}s")
    print(f"Speed improvement: {time1-time2:.1f}s faster with cache")


def main():
    """Run all tests"""
    print("\n" + "=" * 70)
    print(" OCR-Enhanced LegacyACI Test Suite")
    print("=" * 70)
    
    try:
        # Test 1: Basic OCR extraction
        aci, elements = test_ocr_element_extraction()
        
        if not elements:
            logger.warning("\nNo OCR elements found. Tests may not work properly.")
            logger.warning("Make sure you have some text visible on screen.")
            return
        
        # Test 2: Find by text
        test_find_by_text(aci)
        
        # Test 3: List by region
        test_list_visible_texts(aci)
        
        # Test 4: Click simulation
        test_click_by_text(aci)
        
        # Test 5: Application workflow
        test_app_specific_workflow()
        
        # Test 6: Caching performance
        demonstrate_ocr_caching()
        
        print("\n" + "=" * 70)
        print(" All tests completed!")
        print("=" * 70)
        
    except Exception as e:
        logger.error(f"\nTest failed with error: {e}", exc_info=True)
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
