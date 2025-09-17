#!/usr/bin/env python3
"""
Integration Testing for Construction HRI Research Phases.

This module provides comprehensive testing for all implemented construction
HRI research phases, including:
- Phase 1: Speech Processing Stack (Whisper ASR + Haystack NLP + Coqui TTS)
- Phase 2: Construction-Specific Object Detection
- Phase 3: Five Clarification Strategy System  
- Phase 5: Context Memory and History Awareness
- Phase 6: Professional Identity-Aware Communication
- Phase 7: Graduated Uncertainty Expression

Tests validate implementation completeness and integration between components.
"""

import logging
import unittest
import numpy as np
import time
from typing import List, Dict, Any
import tempfile
import cv2

# Import all construction modules for testing
try:
    from SpeechCommandProcessor import SpeechCommandProcessor
    from ConstructionHaystackNLP import ConstructionHaystackNLP, ConstructionIntent, ConstructionEntity
    from ConstructionTTSManager import ConstructionTTSManager, VoiceProfile, TTSPriority
    from ConstructionClarificationManager import (
        ConstructionClarificationManager, ClarificationStrategy, 
        UserExpertiseLevel, ClarificationResponse
    )
    from OWLViTDetector import OWLViTDetector
    IMPORTS_AVAILABLE = True
except ImportError as e:
    logging.error(f"Failed to import construction modules: {e}")
    IMPORTS_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class TestPhase1SpeechProcessing(unittest.TestCase):
    """Test Phase 1: Speech Processing Stack Integration"""
    
    @classmethod
    def setUpClass(cls):
        if not IMPORTS_AVAILABLE:
            cls.skipTest(cls, "Required modules not available")
        cls.speech_processor = SpeechCommandProcessor(whisper_model="tiny")
        cls.haystack_nlp = ConstructionHaystackNLP()
        cls.tts_manager = ConstructionTTSManager()
    
    def test_whisper_asr_integration(self):
        """Test Whisper ASR is properly integrated"""
        self.assertIsNotNone(self.speech_processor.whisper_model)
        self.assertEqual(self.speech_processor.sample_rate, 16000)
    
    def test_haystack_nlp_intent_recognition(self):
        """Test Haystack NLP intent recognition for construction commands"""
        
        test_commands = [
            ("pick up the framing hammer", "pickup_tool"),
            ("find the Phillips screwdriver", "find_tool"),  
            ("place the drill on the workbench", "place_tool"),
            ("go to the toolbox", "move_to_location"),
            ("yes that's correct", "confirm_action"),
            ("stop the current action", "cancel_action")
        ]
        
        for command, expected_intent in test_commands:
            result = self.haystack_nlp.parse_construction_command(command)
            self.assertIsNotNone(result)
            self.assertEqual(result.intent, expected_intent, 
                           f"Command '{command}' should have intent '{expected_intent}', got '{result.intent}'")
    
    def test_haystack_entity_extraction(self):
        """Test entity extraction from construction commands"""
        
        command = "pick up the red framing hammer from the toolbox"
        result = self.haystack_nlp.parse_construction_command(command)
        
        entities = {e['entity']: e['value'] for e in result.entities}
        self.assertIn('tool_name', entities)
        self.assertIn('color', entities)
        self.assertIn('location', entities)
    
    def test_speech_command_processing_with_haystack(self):
        """Test SpeechCommandProcessor integration with Haystack NLP"""
        
        command = "get me the adjustable wrench"
        processed = self.speech_processor.process_command(command)
        
        self.assertIsNotNone(processed)
        self.assertEqual(processed['intent'], 'pickup_tool')
        self.assertIn('wrench', processed['target'].lower())
        self.assertGreater(processed['confidence'], 0.0)
    
    def test_tts_construction_speech(self):
        """Test TTS with construction terminology"""
        
        test_phrases = [
            "I found a framing hammer on the workbench",
            "Looking for a Phillips head screwdriver", 
            "I'm 80% confident this is the right tool"
        ]
        
        for phrase in test_phrases:
            success = self.tts_manager.speak_clarification(phrase, blocking=False)
            self.assertTrue(success)
        
        # Test construction pronunciations
        processed = self.tts_manager._apply_construction_pronunciations("Get the sawzall from the truck")
        self.assertIn("saw-zall", processed)

class TestPhase2ConstructionObjectDetection(unittest.TestCase):
    """Test Phase 2: Construction-Specific Object Detection"""
    
    @classmethod
    def setUpClass(cls):
        if not IMPORTS_AVAILABLE:
            cls.skipTest(cls, "Required modules not available")
        cls.detector = OWLViTDetector(confidence_threshold=0.1)  # Lower threshold for testing
        
        # Create test image (simple colored rectangles representing tools)
        cls.test_image = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.rectangle(cls.test_image, (50, 50), (150, 100), (255, 0, 0), -1)  # Blue "hammer"
        cv2.rectangle(cls.test_image, (200, 150), (300, 200), (0, 255, 0), -1)  # Green "screwdriver"
        cv2.rectangle(cls.test_image, (400, 300), (500, 350), (0, 0, 255), -1)  # Red "wrench"
    
    def test_construction_tool_queries(self):
        """Test construction tool query generation"""
        
        hammer_queries = self.detector.get_construction_tool_queries("hammer")
        self.assertIn("framing hammer", hammer_queries)
        self.assertIn("claw hammer", hammer_queries)
        
        screwdriver_queries = self.detector.get_construction_tool_queries("screwdriver")
        self.assertIn("Phillips head screwdriver", hammer_queries or screwdriver_queries)
    
    def test_construction_tool_categorization(self):
        """Test tool categorization system"""
        
        categories = {
            "hammer": "striking_tool",
            "screwdriver": "turning_tool", 
            "wrench": "gripping_tool",
            "drill": "power_tool",
            "tape measure": "measuring_tool",
            "saw": "cutting_tool"
        }
        
        for tool, expected_category in categories.items():
            category = self.detector._get_tool_category(tool)
            self.assertEqual(category, expected_category)
    
    def test_professional_terminology_mapping(self):
        """Test professional construction terminology is used"""
        
        # Test that professional terms are included in knowledge base
        self.assertIn("framing hammer", str(self.detector.construction_tools))
        self.assertIn("Phillips head screwdriver", str(self.detector.construction_tools))
        self.assertIn("adjustable wrench", str(self.detector.construction_tools))
        self.assertIn("skill saw", str(self.detector.construction_tools))

class TestPhase3ClarificationStrategies(unittest.TestCase):  
    """Test Phase 3: Five Clarification Strategy System"""
    
    @classmethod
    def setUpClass(cls):
        if not IMPORTS_AVAILABLE:
            cls.skipTest(cls, "Required modules not available")
        cls.clarification_manager = ConstructionClarificationManager()
    
    def test_all_five_strategies_implemented(self):
        """Test that all five clarification strategies are implemented"""
        
        test_objects = [{"label": "hammer", "trade_term": "framing hammer"}]
        test_confidences = [0.75]
        
        strategies = [
            ClarificationStrategy.DIRECT,
            ClarificationStrategy.HISTORY_AWARE,
            ClarificationStrategy.CONFIDENCE_BASED, 
            ClarificationStrategy.OPTIONS_BASED,
            ClarificationStrategy.EXPERTISE_ADAPTIVE
        ]
        
        for strategy in strategies:
            response = self.clarification_manager.request_clarification(
                "get me a hammer", test_objects, test_confidences, strategy
            )
            self.assertIsInstance(response, ClarificationResponse)
            self.assertEqual(response.strategy, strategy)
            self.assertTrue(len(response.text) > 0)
    
    def test_expertise_level_adaptation(self):
        """Test expertise-level adaptive responses"""
        
        test_objects = [{"label": "hammer", "trade_term": "framing hammer"}]
        test_confidences = [0.8]
        
        expertise_levels = [
            UserExpertiseLevel.APPRENTICE,
            UserExpertiseLevel.JOURNEYMAN,
            UserExpertiseLevel.FOREMAN,
            UserExpertiseLevel.MASTER
        ]
        
        responses = []
        for level in expertise_levels:
            self.clarification_manager.update_user_expertise(level)
            response = self.clarification_manager.request_clarification(
                "get me a hammer", test_objects, test_confidences, 
                ClarificationStrategy.EXPERTISE_ADAPTIVE
            )
            responses.append(response.text)
        
        # Verify responses are different for different expertise levels
        self.assertEqual(len(set(responses)), len(responses), 
                        "Responses should be different for different expertise levels")
    
    def test_construction_jargon_integration(self):
        """Test construction jargon database integration"""
        
        jargon = self.clarification_manager.construction_jargon
        self.assertIn("framing hammer", jargon)
        self.assertIn("speed square", jargon)
        self.assertIn("sawzall", jargon)
    
    def test_task_memory_system(self):
        """Test shared task memory for history-aware responses"""
        
        # Add task to memory
        self.clarification_manager.update_task_memory(
            "framing hammer", "pickup", True, strategy_used="direct"
        )
        
        # Check memory was updated
        self.assertEqual(len(self.clarification_manager.task_memory), 1)
        
        # Test history-aware response references previous task
        test_objects = [{"label": "hammer", "trade_term": "claw hammer"}]
        response = self.clarification_manager.request_clarification(
            "get me a hammer", test_objects, [0.7], 
            ClarificationStrategy.HISTORY_AWARE
        )
        
        # Should reference "framing hammer" from history
        self.assertIn("framing hammer", response.text)

class TestPhase5ContextMemory(unittest.TestCase):
    """Test Phase 5: Context Memory and History Awareness"""
    
    @classmethod  
    def setUpClass(cls):
        if not IMPORTS_AVAILABLE:
            cls.skipTest(cls, "Required modules not available")
        cls.clarification_manager = ConstructionClarificationManager(memory_size=5)
    
    def test_transactive_memory_implementation(self):
        """Test transactive memory system for shared task history"""
        
        # Simulate task sequence
        tasks = [
            ("framing hammer", "pickup"),
            ("Phillips screwdriver", "pickup"),
            ("measuring tape", "find")
        ]
        
        for tool, action in tasks:
            self.clarification_manager.update_task_memory(tool, action, True)
        
        self.assertEqual(len(self.clarification_manager.task_memory), 3)
        
        # Test "Like the X from earlier" functionality
        test_objects = [{"label": "screwdriver", "trade_term": "flathead screwdriver"}]
        response = self.clarification_manager.request_clarification(
            "find a screwdriver", test_objects, [0.6],
            ClarificationStrategy.HISTORY_AWARE
        )
        
        # Should reference Phillips screwdriver from history
        self.assertTrue("Phillips" in response.text or "screwdriver" in response.text)

class TestPhase67ProfessionalCommunication(unittest.TestCase):
    """Test Phase 6 & 7: Professional Communication & Uncertainty Expression"""
    
    @classmethod
    def setUpClass(cls):
        if not IMPORTS_AVAILABLE:
            cls.skipTest(cls, "Required modules not available") 
        cls.clarification_manager = ConstructionClarificationManager()
    
    def test_graduated_uncertainty_expression(self):
        """Test percentage-based confidence expression"""
        
        test_objects = [{"label": "hammer", "trade_term": "framing hammer"}]
        confidence_levels = [0.9, 0.7, 0.4]  # High, medium, low
        
        for confidence in confidence_levels:
            response = self.clarification_manager.request_clarification(
                "get me a hammer", test_objects, [confidence],
                ClarificationStrategy.CONFIDENCE_BASED
            )
            
            # Should contain percentage expression
            self.assertTrue(any(char.isdigit() for char in response.text))
            self.assertIn("%", response.text)
    
    def test_professional_trade_terminology(self):
        """Test use of professional construction trade terminology"""
        
        # Test that responses use proper trade terms
        test_objects = [{"label": "hammer", "trade_term": "framing hammer"}]
        response = self.clarification_manager.request_clarification(
            "get me a hammer", test_objects, [0.8]
        )
        
        # Should use trade term, not generic term
        self.assertIn("framing hammer", response.text)
    
    def test_measure_twice_cut_once_culture(self):
        """Test alignment with construction's verification culture"""
        
        # Low confidence should trigger verification behavior
        test_objects = [{"label": "saw", "trade_term": "circular saw"}]  
        response = self.clarification_manager.request_clarification(
            "get me a saw", test_objects, [0.3],
            ClarificationStrategy.CONFIDENCE_BASED
        )
        
        # Should suggest double-checking or verification
        verification_words = ["double", "check", "sure", "confirm", "verify"]
        self.assertTrue(any(word in response.text.lower() for word in verification_words))

class TestIntegrationCompleteness(unittest.TestCase):
    """Test overall system integration and completeness"""
    
    @classmethod
    def setUpClass(cls):
        if not IMPORTS_AVAILABLE:
            cls.skipTest(cls, "Required modules not available")
        cls.speech_processor = SpeechCommandProcessor()
        cls.clarification_manager = ConstructionClarificationManager()
        cls.tts_manager = ConstructionTTSManager()
        cls.detector = OWLViTDetector()
    
    def test_end_to_end_construction_workflow(self):
        """Test complete construction HRI workflow integration"""
        
        # 1. Process voice command
        command = "pick up the framing hammer" 
        parsed_command = self.speech_processor.process_command(command)
        self.assertIsNotNone(parsed_command)
        
        # 2. Extract tool queries for object detection
        queries = self.speech_processor.parse_object_query(command)
        self.assertTrue(len(queries) > 0)
        
        # 3. Simulate object detection results
        mock_detections = [{"label": "framing hammer", "trade_term": "framing hammer"}]
        mock_confidences = [0.75]
        
        # 4. Generate clarification if needed
        clarification = self.clarification_manager.request_clarification(
            command, mock_detections, mock_confidences
        )
        self.assertIsInstance(clarification, ClarificationResponse)
        
        # 5. Speak clarification via TTS
        tts_success = self.tts_manager.speak_clarification(
            clarification.text, TTSPriority.NORMAL, blocking=False
        )
        self.assertTrue(tts_success)
        
        # 6. Update task memory for future context
        self.clarification_manager.update_task_memory(
            "framing hammer", "pickup", True
        )
        
        self.assertEqual(len(self.clarification_manager.task_memory), 1)
    
    def test_all_phases_implemented(self):
        """Verify all required phases are fully implemented"""
        
        # Phase 1: Speech Processing Stack
        self.assertIsNotNone(self.speech_processor.whisper_model)  # Whisper ASR
        self.assertIsNotNone(self.speech_processor.haystack_nlp)  # Haystack NLP  
        self.assertIsNotNone(self.tts_manager.tts_engine)          # Coqui TTS (or fallback)
        
        # Phase 2: Construction Object Detection
        self.assertTrue(len(self.detector.construction_tools) > 0)
        
        # Phase 3: Five Clarification Strategies
        strategies = [s.value for s in ClarificationStrategy]
        self.assertEqual(len(strategies), 5)
        
        # Phase 5: Context Memory
        self.assertIsNotNone(self.clarification_manager.task_memory)
        
        # Phase 6: Professional Identity-Aware Communication  
        self.assertTrue(len(self.clarification_manager.construction_jargon) > 0)
        
        # Phase 7: Graduated Uncertainty Expression
        test_response = self.clarification_manager._confidence_based_clarification(
            "test", [{"label": "tool"}], [0.5]
        )
        self.assertIn("%", test_response.text)

def run_construction_integration_tests():
    """Run all construction HRI integration tests"""
    
    if not IMPORTS_AVAILABLE:
        print("❌ Required modules not available - skipping tests")
        return False
    
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_classes = [
        TestPhase1SpeechProcessing,
        TestPhase2ConstructionObjectDetection, 
        TestPhase3ClarificationStrategies,
        TestPhase5ContextMemory,
        TestPhase67ProfessionalCommunication,
        TestIntegrationCompleteness
    ]
    
    for test_class in test_classes:
        test_suite.addTest(unittest.makeSuite(test_class))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print summary
    print(f"\n{'='*60}")
    print("CONSTRUCTION HRI PHASE INTEGRATION TEST SUMMARY")
    print(f"{'='*60}")
    print(f"Tests Run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success Rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    
    if result.failures:
        print(f"\n❌ FAILURES:")
        for test, traceback in result.failures:
            print(f"  - {test}: {traceback.split('AssertionError:')[-1].strip()}")
    
    if result.errors:
        print(f"\n❌ ERRORS:")
        for test, traceback in result.errors:
            print(f"  - {test}: {traceback.split('Exception:')[-1].strip()}")
    
    return len(result.failures) == 0 and len(result.errors) == 0

if __name__ == "__main__":
    run_construction_integration_tests()