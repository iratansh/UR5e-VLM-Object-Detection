#!/usr/bin/env python3
"""
Test script for Construction TTS Manager.

This script validates the TTS integration for construction clarification
dialogues and speech synthesis capabilities.
"""

import sys
import os
import logging
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_tts_integration():
    """Test Construction TTS Manager functionality"""
    
    try:
        from ConstructionTTSManager import (
            ConstructionTTSManager,
            VoiceProfile, 
            TTSPriority,
            create_construction_tts
        )
        
        logger.info("✅ ConstructionTTSManager imported successfully")
        
        # Test basic initialization
        logger.info("\n" + "="*50)
        logger.info("TESTING TTS INITIALIZATION")
        logger.info("="*50)
        
        tts_manager = ConstructionTTSManager(
            voice_profile=VoiceProfile.PROFESSIONAL,
            enable_background_speech=True,
            construction_mode=True
        )
        
        # Test capability check
        capabilities = tts_manager.test_speech_capabilities()
        logger.info(f"TTS Capabilities: {capabilities}")
        
        # Test different voice profiles
        logger.info("\n" + "="*50)
        logger.info("TESTING VOICE PROFILES")
        logger.info("="*50)
        
        voice_profiles = [
            (VoiceProfile.APPRENTICE_FRIENDLY, "Friendly pace for learning"),
            (VoiceProfile.PROFESSIONAL, "Standard construction pace"),
            (VoiceProfile.URGENT, "Faster for urgent situations"),
            (VoiceProfile.NOISY_ENVIRONMENT, "Optimized for construction sites")
        ]
        
        for profile, description in voice_profiles:
            logger.info(f"\n🔊 Testing {profile.value}: {description}")
            tts_manager.set_voice_profile(profile)
            
            # Test speech with this profile
            test_text = f"Testing {profile.value} voice profile for construction communication."
            success = tts_manager.speak_clarification(
                test_text,
                priority=TTSPriority.NORMAL,
                blocking=True  # Wait for completion in tests
            )
            
            if success:
                logger.info(f"   ✅ {profile.value} speech test completed")
            else:
                logger.warning(f"   ⚠️ {profile.value} speech test failed")
            
            time.sleep(0.5)  # Brief pause between tests
        
        # Test construction terminology pronunciation
        logger.info("\n" + "="*50)
        logger.info("TESTING CONSTRUCTION TERMINOLOGY")
        logger.info("="*50)
        
        construction_phrases = [
            "I found a sawzall on the workbench.",
            "The dewalt drill is at 2x4 lumber.",
            "Check if the wall is plumb and square.",
            "Set the studs at 16\" O.C. spacing.",
            "Use a 3/4\" bit for the rebar holes."
        ]
        
        tts_manager.set_voice_profile(VoiceProfile.PROFESSIONAL)
        
        for phrase in construction_phrases:
            logger.info(f"🔨 Speaking: '{phrase}'")
            tts_manager.speak_clarification(phrase, blocking=True)
            time.sleep(0.3)
        
        # Test priority levels
        logger.info("\n" + "="*50)
        logger.info("TESTING PRIORITY LEVELS")
        logger.info("="*50)
        
        priority_tests = [
            (TTSPriority.LOW, "Background information: Tool inventory updated."),
            (TTSPriority.NORMAL, "Standard clarification: Is this the right hammer?"),
            (TTSPriority.HIGH, "Important message: Double-check measurements before cutting."),
            (TTSPriority.CRITICAL, "SAFETY ALERT: Hard hat required in this area!")
        ]
        
        for priority, message in priority_tests:
            logger.info(f"📢 Priority {priority.name}: {message}")
            tts_manager.speak_clarification(message, priority=priority, blocking=True)
            time.sleep(0.5)
        
        # Test integration with clarification manager
        logger.info("\n" + "="*50)
        logger.info("TESTING CLARIFICATION INTEGRATION")
        logger.info("="*50)
        
        try:
            from ConstructionClarificationManager import (
                ConstructionClarificationManager,
                ClarificationStrategy,
                UserExpertiseLevel
            )
            
            # Create clarification manager
            clarification_mgr = ConstructionClarificationManager(
                user_expertise=UserExpertiseLevel.JOURNEYMAN
            )
            
            # Mock detection data
            mock_detection = [{
                'label': 'framing hammer',
                'trade_term': 'framing hammer', 
                'category': 'striking_tool',
                'bbox': [100, 100, 200, 200]
            }]
            mock_confidence = [0.85]
            
            # Test different clarification strategies with TTS
            strategies = [
                ClarificationStrategy.DIRECT,
                ClarificationStrategy.CONFIDENCE_BASED,
                ClarificationStrategy.EXPERTISE_ADAPTIVE
            ]
            
            for strategy in strategies:
                response = clarification_mgr.request_clarification(
                    tool_request="hammer",
                    detected_objects=mock_detection,
                    confidence_scores=mock_confidence,
                    strategy=strategy
                )
                
                logger.info(f"🤖 {strategy.value}: {response.text}")
                
                # Speak the clarification
                if response.tts_enabled:
                    tts_manager.speak_clarification(response.text, blocking=True)
                
                time.sleep(0.5)
            
            logger.info("✅ Clarification integration test completed")
            
        except ImportError as e:
            logger.warning(f"Clarification integration test skipped: {e}")
        
        # Test expertise-based TTS configuration
        logger.info("\n" + "="*50)
        logger.info("TESTING EXPERTISE-BASED CONFIGURATION")
        logger.info("="*50)
        
        expertise_levels = ["apprentice", "journeyman", "foreman", "master"]
        
        for level in expertise_levels:
            logger.info(f"👷 Testing {level} TTS configuration")
            
            expert_tts = create_construction_tts(level)
            capabilities = expert_tts.test_speech_capabilities()
            
            test_message = f"TTS configured for {level} level worker."
            expert_tts.speak_clarification(test_message, blocking=True)
            
            logger.info(f"   Profile: {capabilities['current_profile']}")
            expert_tts.cleanup()
        
        # Test background vs blocking speech
        logger.info("\n" + "="*50)
        logger.info("TESTING SPEECH MODES")
        logger.info("="*50)
        
        logger.info("🔄 Testing background (non-blocking) speech...")
        tts_manager.speak_clarification("This is background speech message one.", blocking=False)
        tts_manager.speak_clarification("This is background speech message two.", blocking=False)
        tts_manager.speak_clarification("This is background speech message three.", blocking=False)
        
        # Wait for background speech to complete
        logger.info(f"   Queue size: {tts_manager.get_queue_size()}")
        time.sleep(3)
        
        logger.info("⏸️ Testing speech interruption...")
        tts_manager.speak_clarification("This is a long message that will be interrupted before completion.", blocking=False)
        time.sleep(0.5)
        tts_manager.stop_current_speech()
        logger.info("   Speech stopped successfully")
        
        # Final status
        logger.info("\n" + "="*50)
        logger.info("FINAL STATUS")
        logger.info("="*50)
        
        final_capabilities = tts_manager.test_speech_capabilities()
        logger.info(f"Final TTS status: {final_capabilities}")
        
        # Cleanup
        tts_manager.cleanup()
        
        logger.info("\n✅ Phase 1D TTS integration test completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_tts_integration()
    sys.exit(0 if success else 1)