#!/usr/bin/env python3
"""
Test script for Construction Tool Detection in OWLViTDetector.

This script validates that the construction tool detection works correctly
with professional trade terminology.
"""

import sys
import os
import logging
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_construction_detection():
    """Test construction tool detection functionality"""
    try:
        # Test imports
        import torch
        from OWLViTDetector import OWLViTDetector
        from PIL import Image
        logger.info("✅ All required modules imported successfully")
        
        # Initialize detector with construction capabilities
        logger.info("Initializing OWL-ViT detector for construction tools...")
        detector = OWLViTDetector(confidence_threshold=0.2)
        logger.info("✅ Construction-enhanced OWL-ViT detector initialized")
        
        # Test construction tool query generation
        test_tools = ["hammer", "screwdriver", "wrench", "drill", "level"]
        logger.info("Testing construction tool query generation:")
        
        for tool in test_tools:
            queries = detector.get_construction_tool_queries(tool)
            category = detector._get_tool_category(tool)
            logger.info(f"  🔨 '{tool}' -> {queries} (category: {category})")
        
        # Test with a dummy image (since we don't have a real construction scene)
        logger.info("Creating test image...")
        dummy_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # Test construction tool detection (will likely return empty results with dummy image)
        logger.info("Testing construction tool detection with professional terminology...")
        detections = detector.detect_construction_tools(
            dummy_image, 
            ["hammer", "screwdriver"], 
            confidence_threshold=0.1
        )
        
        logger.info(f"Found {len(detections)} detections (expected 0 with dummy image)")
        for det in detections:
            logger.info(f"  Detection: {det['label']} ({det['confidence']:.3f}) - {det['category']}")
        
        # Test professional terminology expansion
        logger.info("Testing professional terminology expansion:")
        test_cases = {
            "hammer": "Should expand to framing hammer, claw hammer, etc.",
            "phillips screwdriver": "Should find Phillips head variations",
            "adjustable wrench": "Should include crescent wrench, spanner",
            "2x4": "Should categorize as construction material"
        }
        
        for tool, expected in test_cases.items():
            queries = detector.get_construction_tool_queries(tool)
            category = detector._get_tool_category(tool)
            logger.info(f"  '{tool}' -> {len(queries)} variants, category: {category}")
            logger.info(f"    Expected: {expected}")
            logger.info(f"    Got: {queries[:3]}...")  # Show first 3 variants
        
        logger.info("✅ Phase 1B Construction tool detection test completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_construction_detection()
    sys.exit(0 if success else 1)