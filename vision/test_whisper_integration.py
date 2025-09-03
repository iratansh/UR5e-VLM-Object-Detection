#!/usr/bin/env python3
"""
Test script for Whisper ASR integration in SpeechCommandProcessor.

This script validates that the Whisper integration works correctly
for construction site speech recognition.
"""

import sys
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_whisper_integration():
    """Test basic Whisper ASR functionality"""
    try:
        # Test imports
        import whisper
        import sounddevice as sd
        import numpy as np
        logger.info("✅ All required modules imported successfully")
        
        # Test Whisper model loading
        logger.info("Loading Whisper base model...")
        model = whisper.load_model("base")
        logger.info("✅ Whisper model loaded successfully")
        
        # Test audio device availability
        devices = sd.query_devices()
        logger.info(f"✅ Found {len(devices)} audio devices")
        
        # Test SpeechCommandProcessor import
        from SpeechCommandProcessor import SpeechCommandProcessor
        logger.info("✅ SpeechCommandProcessor imported successfully")
        
        # Initialize processor with Whisper
        logger.info("Initializing SpeechCommandProcessor with Whisper...")
        processor = SpeechCommandProcessor(whisper_model="tiny")  # Use tiny model for testing
        logger.info("✅ SpeechCommandProcessor initialized successfully")
        
        # Test command processing
        test_commands = [
            "pick up the hammer",
            "get the red screwdriver", 
            "grasp the adjustable wrench"
        ]
        
        for cmd in test_commands:
            result = processor.process_command(cmd)
            if result:
                logger.info(f"✅ Command '{cmd}' -> Intent: {result['intent']}, Target: {result['target']}")
            else:
                logger.warning(f"⚠️ Command '{cmd}' could not be processed")
        
        # Clean up
        processor.cleanup()
        logger.info("✅ Phase 1A Whisper integration test completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_whisper_integration()
    sys.exit(0 if success else 1)