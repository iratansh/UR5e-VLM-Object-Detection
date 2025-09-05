#!/usr/bin/env python3
"""
Construction Rasa NLP Manager for Intent Recognition and Entity Extraction.

This module replaces the spaCy + transformers pipeline in SpeechCommandProcessor
with a dedicated Rasa NLP framework optimized for construction voice commands.

Provides:
- Construction-specific intent classification
- Entity extraction for tools, locations, and actions
- Domain-specific training data management
- Integration with existing speech processing pipeline
"""

import logging
import json
import os
import tempfile
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import yaml

# Rasa imports - graceful fallback if not available
try:
    from rasa.core.agent import Agent
    from rasa.core.interpreter import RasaNLUInterpreter
    from rasa.shared.nlu.training_data.training_data import TrainingData
    from rasa.shared.nlu.training_data.formats.rasa_yaml import RasaYAMLReader
    from rasa.model_training import train_nlu
    RASA_AVAILABLE = True
except ImportError:
    RASA_AVAILABLE = False
    logging.warning("Rasa not available - using mock NLP implementation")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ConstructionIntent(Enum):
    """Construction-specific intents for voice commands"""
    PICKUP_TOOL = "pickup_tool"
    PLACE_TOOL = "place_tool" 
    FIND_TOOL = "find_tool"
    MOVE_TO_LOCATION = "move_to_location"
    DESCRIBE_OBJECT = "describe_object"
    CONFIRM_ACTION = "confirm_action"
    CANCEL_ACTION = "cancel_action"
    REQUEST_HELP = "request_help"
    SAFETY_ALERT = "safety_alert"
    STATUS_CHECK = "status_check"

class ConstructionEntity(Enum):
    """Construction domain entities"""
    TOOL_NAME = "tool_name"
    TOOL_TYPE = "tool_type" 
    LOCATION = "location"
    MATERIAL = "material"
    MEASUREMENT = "measurement"
    COLOR = "color"
    SIZE = "size"
    URGENCY = "urgency"
    PERSON = "person"
    SAFETY_CONCERN = "safety_concern"

@dataclass
class NLUResult:
    """Result from NLP processing"""
    intent: str
    confidence: float
    entities: List[Dict[str, Any]]
    text: str
    metadata: Dict[str, Any]

class ConstructionRasaNLP:
    """
    Rasa-based NLP manager for construction voice commands.
    
    Replaces the spaCy + transformers pipeline with construction-specific
    intent recognition and entity extraction using Rasa NLU.
    
    Parameters
    ----------
    model_path : str, optional
        Path to trained Rasa model directory
    training_data_path : str, optional
        Path to construction training data
    confidence_threshold : float, optional
        Intent confidence threshold, by default 0.6
        
    Attributes
    ----------
    interpreter : RasaNLUInterpreter or None
        Rasa NLU interpreter instance
    training_data : Dict
        Construction-specific training examples
    model_path : str
        Path to the trained model
    """
    
    def __init__(self,
                 model_path: Optional[str] = None,
                 training_data_path: Optional[str] = None, 
                 confidence_threshold: float = 0.6):
        
        self.logger = logging.getLogger(__name__)
        self.confidence_threshold = confidence_threshold
        self.model_path = model_path
        self.training_data_path = training_data_path
        self.interpreter = None
        
        # Initialize construction training data
        self.training_data = self._create_construction_training_data()
        
        # Initialize Rasa NLU
        if RASA_AVAILABLE:
            self._initialize_rasa_nlu()
        else:
            self.logger.warning("Rasa not available - using mock NLP")
            
        self.logger.info("✅ Construction Rasa NLP Manager initialized")

    def _create_construction_training_data(self) -> Dict[str, Any]:
        """Create comprehensive construction training data"""
        
        return {
            "version": "3.1",
            "nlu": [
                # PICKUP_TOOL intent examples
                {
                    "intent": "pickup_tool",
                    "examples": [
                        "pick up the [hammer](tool_name)",
                        "get me the [framing hammer](tool_name)",
                        "grab that [Phillips screwdriver](tool_name)",
                        "bring me a [level](tool_name)",
                        "fetch the [red](color) [drill](tool_name)",
                        "hand me the [adjustable wrench](tool_name)",
                        "I need the [measuring tape](tool_name)",
                        "get the [blue](color) [hammer](tool_name) from the [toolbox](location)",
                        "pick up that [skill saw](tool_name) on the [workbench](location)",
                        "grab the [16 oz](size) [framing hammer](tool_name)"
                    ]
                },
                
                # FIND_TOOL intent examples  
                {
                    "intent": "find_tool",
                    "examples": [
                        "find the [hammer](tool_name)",
                        "where is the [screwdriver](tool_name)",
                        "locate a [level](tool_name)",
                        "search for the [drill](tool_name)",
                        "look for the [measuring tape](tool_name)",
                        "can you see a [wrench](tool_name)",
                        "is there a [saw](tool_name) around here",
                        "find me a [Phillips head](tool_type) [screwdriver](tool_name)",
                        "look for a [small](size) [hammer](tool_name)",
                        "where's the [red](color) [toolbox](location)"
                    ]
                },
                
                # PLACE_TOOL intent examples
                {
                    "intent": "place_tool", 
                    "examples": [
                        "put down the [hammer](tool_name)",
                        "place the [drill](tool_name) on the [workbench](location)",
                        "set the [level](tool_name) down",
                        "drop off the [screwdriver](tool_name)",
                        "return the [measuring tape](tool_name) to the [toolbox](location)",
                        "put the [wrench](tool_name) back",
                        "place it on the [table](location)",
                        "set that down on the [floor](location)",
                        "put the [saw](tool_name) in the [truck](location)"
                    ]
                },
                
                # MOVE_TO_LOCATION intent examples
                {
                    "intent": "move_to_location",
                    "examples": [
                        "go to the [workbench](location)",
                        "move to the [toolbox](location)", 
                        "head over to the [truck](location)",
                        "navigate to [home position](location)",
                        "go to the [left side](location)",
                        "move [closer](location)",
                        "back up",
                        "go to the [gang box](location)",
                        "move to the [job site](location)",
                        "position yourself near the [materials](location)"
                    ]
                },
                
                # DESCRIBE_OBJECT intent examples
                {
                    "intent": "describe_object",
                    "examples": [
                        "what do you see",
                        "describe what's there", 
                        "what tools are available",
                        "tell me what you found",
                        "what's on the [workbench](location)",
                        "describe the [hammer](tool_name)",
                        "what size is that [wrench](tool_name)",
                        "what color is the [drill](tool_name)",
                        "how many [screwdrivers](tool_name) are there",
                        "what's in the [toolbox](location)"
                    ]
                },
                
                # CONFIRM_ACTION intent examples
                {
                    "intent": "confirm_action",
                    "examples": [
                        "yes", "yeah", "yep", "correct", "right", "that's right",
                        "affirmative", "go ahead", "do it", "proceed", "continue",
                        "that's the one", "perfect", "exactly", "good", "ok", "okay"
                    ]
                },
                
                # CANCEL_ACTION intent examples  
                {
                    "intent": "cancel_action",
                    "examples": [
                        "no", "nope", "stop", "cancel", "abort", "wrong", "incorrect",
                        "that's not right", "not that one", "different tool", "hold on",
                        "wait", "pause", "never mind", "forget it", "not now"
                    ]
                },
                
                # SAFETY_ALERT intent examples
                {
                    "intent": "safety_alert", 
                    "examples": [
                        "watch out", "be careful", "safety concern", "danger",
                        "stop immediately", "[hard hat](safety_concern) required",
                        "[safety glasses](safety_concern) needed", "hazard alert",
                        "emergency stop", "clear the area", "[fall hazard](safety_concern)",
                        "[electrical hazard](safety_concern)", "hot surface"
                    ]
                },
                
                # STATUS_CHECK intent examples
                {
                    "intent": "status_check",
                    "examples": [
                        "what's your status", "are you ready", "how's it going",
                        "what are you doing", "current task", "system status", 
                        "battery level", "are you working", "progress report",
                        "all good", "any problems", "operational status"
                    ]
                }
            ]
        }

    def _initialize_rasa_nlu(self):
        """Initialize Rasa NLU interpreter"""
        
        if not RASA_AVAILABLE:
            return
            
        try:
            # Create temporary training data file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.yml', delete=False) as f:
                yaml.dump(self.training_data, f, default_flow_style=False)
                training_file = f.name
            
            # Train model if no existing model path
            if not self.model_path or not os.path.exists(self.model_path):
                self.logger.info("🤖 Training Rasa NLU model for construction domain...")
                
                # Create config for NLU pipeline
                config = {
                    "language": "en",
                    "pipeline": [
                        {"name": "WhitespaceTokenizer"},
                        {"name": "RegexFeaturizer"},
                        {"name": "LexicalSyntacticFeaturizer"},
                        {"name": "CountVectorsFeaturizer"},
                        {"name": "CountVectorsFeaturizer",
                         "analyzer": "char_wb",
                         "min_ngram": 1, 
                         "max_ngram": 4},
                        {"name": "DIETClassifier",
                         "epochs": 100,
                         "constrain_similarities": True},
                        {"name": "EntitySynonymMapper"},
                        {"name": "ResponseSelector",
                         "epochs": 100,
                         "constrain_similarities": True}
                    ]
                }
                
                # Write config file
                with tempfile.NamedTemporaryFile(mode='w', suffix='.yml', delete=False) as f:
                    yaml.dump(config, f, default_flow_style=False)
                    config_file = f.name
                
                # Train the model
                self.model_path = train_nlu(
                    config=config_file,
                    nlu_data=training_file,
                    output_dir=tempfile.mkdtemp(),
                    fixed_model_name="construction_nlu",
                    persist_nlu_training_data=True
                )
                
                self.logger.info(f"✅ Model trained and saved to: {self.model_path}")
                
                # Cleanup temp files
                os.unlink(training_file)
                os.unlink(config_file)
            
            # Load the interpreter
            self.interpreter = RasaNLUInterpreter(self.model_path)
            self.logger.info("✅ Rasa NLU interpreter loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Rasa NLU: {e}")
            self.interpreter = None

    def parse_construction_command(self, text: str) -> NLUResult:
        """
        Parse construction voice command using Rasa NLU.
        
        Parameters
        ----------
        text : str
            Voice command text to parse
            
        Returns
        -------
        NLUResult
            Parsed intent, entities, and metadata
        """
        
        if not text.strip():
            return NLUResult(
                intent="unknown",
                confidence=0.0,
                entities=[],
                text=text,
                metadata={"error": "Empty input"}
            )
        
        if self.interpreter and RASA_AVAILABLE:
            try:
                result = self.interpreter.parse(text)
                
                intent = result.get('intent', {}).get('name', 'unknown')
                confidence = result.get('intent', {}).get('confidence', 0.0)
                entities = result.get('entities', [])
                
                return NLUResult(
                    intent=intent,
                    confidence=confidence,
                    entities=entities,
                    text=text,
                    metadata={
                        'rasa_result': result,
                        'processing_time': result.get('processing_time', 0)
                    }
                )
                
            except Exception as e:
                self.logger.error(f"Rasa parsing failed: {e}")
                return self._fallback_parsing(text)
        else:
            # Mock implementation when Rasa not available
            return self._mock_parsing(text)

    def _fallback_parsing(self, text: str) -> NLUResult:
        """Fallback parsing when Rasa fails"""
        
        # Simple rule-based fallback
        text_lower = text.lower()
        
        # Intent classification
        if any(word in text_lower for word in ['pick up', 'get', 'grab', 'bring', 'fetch', 'hand me']):
            intent = "pickup_tool"
        elif any(word in text_lower for word in ['find', 'where', 'locate', 'search', 'look for']):
            intent = "find_tool"  
        elif any(word in text_lower for word in ['put down', 'place', 'set down', 'drop off', 'return']):
            intent = "place_tool"
        elif any(word in text_lower for word in ['go to', 'move to', 'navigate', 'head']):
            intent = "move_to_location"
        elif any(word in text_lower for word in ['yes', 'yeah', 'correct', 'right', 'ok', 'okay']):
            intent = "confirm_action"
        elif any(word in text_lower for word in ['no', 'stop', 'cancel', 'wrong', 'wait']):
            intent = "cancel_action"
        else:
            intent = "unknown"
        
        # Simple entity extraction
        entities = []
        tools = ['hammer', 'screwdriver', 'wrench', 'drill', 'saw', 'level', 'square', 'tape']
        colors = ['red', 'blue', 'green', 'yellow', 'black', 'white']
        locations = ['toolbox', 'workbench', 'truck', 'floor', 'table']
        
        for tool in tools:
            if tool in text_lower:
                entities.append({
                    'entity': 'tool_name',
                    'value': tool,
                    'confidence': 0.8
                })
        
        for color in colors:
            if color in text_lower:
                entities.append({
                    'entity': 'color', 
                    'value': color,
                    'confidence': 0.8
                })
                
        for location in locations:
            if location in text_lower:
                entities.append({
                    'entity': 'location',
                    'value': location,
                    'confidence': 0.8
                })
        
        confidence = 0.7 if intent != "unknown" else 0.3
        
        return NLUResult(
            intent=intent,
            confidence=confidence,
            entities=entities,
            text=text,
            metadata={'fallback_parsing': True}
        )

    def _mock_parsing(self, text: str) -> NLUResult:
        """Mock parsing when Rasa not available"""
        
        self.logger.info(f"🤖 [MOCK RASA] Parsing: '{text}'")
        return self._fallback_parsing(text)

    def extract_tool_references(self, text: str) -> List[str]:
        """
        Extract construction tool references from text.
        
        Parameters
        ----------
        text : str
            Input text to analyze
            
        Returns
        -------
        List[str]
            List of tool names found in text
        """
        
        result = self.parse_construction_command(text)
        
        tool_references = []
        for entity in result.entities:
            if entity.get('entity') in ['tool_name', 'tool_type']:
                tool_references.append(entity.get('value'))
        
        return tool_references

    def get_intent_confidence(self, text: str) -> Tuple[str, float]:
        """
        Get intent and confidence for construction command.
        
        Parameters
        ----------
        text : str
            Voice command text
            
        Returns
        -------
        Tuple[str, float]
            Intent name and confidence score
        """
        
        result = self.parse_construction_command(text)
        return result.intent, result.confidence

    def is_valid_construction_command(self, text: str) -> bool:
        """
        Check if text represents a valid construction command.
        
        Parameters
        ----------
        text : str
            Text to validate
            
        Returns
        -------
        bool
            True if valid construction command, False otherwise
        """
        
        intent, confidence = self.get_intent_confidence(text)
        return confidence >= self.confidence_threshold and intent != "unknown"

    def get_supported_intents(self) -> List[str]:
        """Get list of supported construction intents"""
        return [intent.value for intent in ConstructionIntent]

    def get_supported_entities(self) -> List[str]:
        """Get list of supported construction entities"""
        return [entity.value for entity in ConstructionEntity]

    def update_training_data(self, new_examples: List[Dict[str, Any]]):
        """
        Add new training examples to improve model performance.
        
        Parameters
        ----------
        new_examples : List[Dict[str, Any]]
            New training examples with intent and entities
        """
        
        if not new_examples:
            return
            
        # Add to existing training data
        self.training_data['nlu'].extend(new_examples)
        
        self.logger.info(f"📚 Added {len(new_examples)} new training examples")
        
        # Retrain model if Rasa available
        if RASA_AVAILABLE:
            self.logger.info("🔄 Retraining model with new examples...")
            self._initialize_rasa_nlu()

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current NLU model"""
        
        return {
            'rasa_available': RASA_AVAILABLE,
            'model_path': self.model_path,
            'confidence_threshold': self.confidence_threshold,
            'training_examples': len(self.training_data.get('nlu', [])),
            'supported_intents': len(self.get_supported_intents()),
            'supported_entities': len(self.get_supported_entities()),
            'interpreter_loaded': self.interpreter is not None
        }

    def cleanup(self):
        """Clean up Rasa NLP resources"""
        
        if self.interpreter:
            try:
                # Cleanup interpreter resources
                del self.interpreter
                self.interpreter = None
            except Exception as e:
                self.logger.error(f"Error cleaning up Rasa interpreter: {e}")
        
        self.logger.info("🤖 Rasa NLP Manager cleaned up")