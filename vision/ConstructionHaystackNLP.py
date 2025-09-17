#!/usr/bin/env python3
"""
Construction Haystack NLP Manager for Intent Recognition and Entity Extraction.

This module provides Haystack NLP framework for SpeechCommandProcessor
for construction voice commands processing. Haystack provides modern NLP capabilities
with excellent RAG integration and Python 3.10 compatibility.

Provides:
- Construction-specific intent classification using Haystack components
- Entity extraction for tools, locations, and actions
- Question answering capabilities for construction queries
- Integration with existing ChromaDB RAG system
- Modern semantic NLP capabilities for construction voice commands
"""

import logging
import json
import os
import tempfile
import warnings
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum

# Suppress known compatibility warnings
warnings.filterwarnings("ignore", message="pkg_resources is deprecated")

# Haystack imports - graceful fallback if not available
try:
    from haystack import Pipeline, Document
    from haystack.components.embedders import SentenceTransformersTextEmbedder
    from haystack.document_stores.in_memory import InMemoryDocumentStore
    from haystack.components.retrievers.in_memory import InMemoryBM25Retriever
    HAYSTACK_AVAILABLE = True
    logging.info("✅ Haystack NLP successfully imported")
except ImportError as e:
    HAYSTACK_AVAILABLE = False
    logging.warning(f"Haystack not available - using mock NLP implementation: {e}")
except Exception as e:
    HAYSTACK_AVAILABLE = False
    logging.warning(f"Haystack import error - using mock NLP implementation: {e}")

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

class ConstructionHaystackNLP:
    """
    Haystack-based NLP manager for construction voice commands.
    
    Provides modern Haystack framework for construction voice commands with
    intent recognition, entity extraction, and RAG integration.
    
    Parameters
    ----------
    model_name : str, optional
        Sentence transformers model for embeddings, by default 'all-mpnet-base-v2'
    confidence_threshold : float, optional
        Intent confidence threshold, by default 0.6
        
    Attributes
    ----------
    pipeline : Pipeline or None
        Haystack Pipeline instance
    document_store : InMemoryDocumentStore
        Document store for construction knowledge
    embedder : SentenceTransformersTextEmbedder
        Text embedder for semantic similarity
    """
    
    def __init__(self,
                 model_name: str = "all-mpnet-base-v2",
                 confidence_threshold: float = 0.6):
        
        self.logger = logging.getLogger(__name__)
        self.confidence_threshold = confidence_threshold
        self.model_name = model_name
        self.pipeline = None
        self.document_store = None
        self.embedder = None
        
        # Initialize construction training examples
        self.training_examples = self._create_construction_training_data()
        
        # Initialize Haystack NLP
        if HAYSTACK_AVAILABLE:
            self.logger.info("✅ Haystack available - initializing NLP pipeline")
            self._initialize_haystack_pipeline()
        else:
            self.logger.warning("Haystack not available - using enhanced mock NLP")
            
        self.logger.info("✅ Construction Haystack NLP Manager initialized")

    def _create_construction_training_data(self) -> List[Dict[str, Any]]:
        """Create comprehensive construction training examples"""
        
        return [
            # PICKUP_TOOL intent examples
            {"text": "pick up the hammer", "intent": "pickup_tool", "entities": [{"entity": "tool_name", "value": "hammer"}]},
            {"text": "get me the framing hammer", "intent": "pickup_tool", "entities": [{"entity": "tool_name", "value": "framing hammer"}]},
            {"text": "grab that Phillips screwdriver", "intent": "pickup_tool", "entities": [{"entity": "tool_name", "value": "Phillips screwdriver"}]},
            {"text": "bring me a level", "intent": "pickup_tool", "entities": [{"entity": "tool_name", "value": "level"}]},
            {"text": "fetch the red drill", "intent": "pickup_tool", "entities": [{"entity": "tool_name", "value": "drill"}, {"entity": "color", "value": "red"}]},
            
            # FIND_TOOL intent examples  
            {"text": "find the hammer", "intent": "find_tool", "entities": [{"entity": "tool_name", "value": "hammer"}]},
            {"text": "where is the screwdriver", "intent": "find_tool", "entities": [{"entity": "tool_name", "value": "screwdriver"}]},
            {"text": "locate a level", "intent": "find_tool", "entities": [{"entity": "tool_name", "value": "level"}]},
            {"text": "search for the drill", "intent": "find_tool", "entities": [{"entity": "tool_name", "value": "drill"}]},
            {"text": "look for the measuring tape", "intent": "find_tool", "entities": [{"entity": "tool_name", "value": "measuring tape"}]},
            
            # PLACE_TOOL intent examples
            {"text": "put down the hammer", "intent": "place_tool", "entities": [{"entity": "tool_name", "value": "hammer"}]},
            {"text": "place the drill on the workbench", "intent": "place_tool", "entities": [{"entity": "tool_name", "value": "drill"}, {"entity": "location", "value": "workbench"}]},
            {"text": "set the level down", "intent": "place_tool", "entities": [{"entity": "tool_name", "value": "level"}]},
            {"text": "return the measuring tape to the toolbox", "intent": "place_tool", "entities": [{"entity": "tool_name", "value": "measuring tape"}, {"entity": "location", "value": "toolbox"}]},
            
            # MOVE_TO_LOCATION intent examples
            {"text": "go to the workbench", "intent": "move_to_location", "entities": [{"entity": "location", "value": "workbench"}]},
            {"text": "move to the toolbox", "intent": "move_to_location", "entities": [{"entity": "location", "value": "toolbox"}]},
            {"text": "head over to the truck", "intent": "move_to_location", "entities": [{"entity": "location", "value": "truck"}]},
            {"text": "navigate to home position", "intent": "move_to_location", "entities": [{"entity": "location", "value": "home position"}]},
            
            # DESCRIBE_OBJECT intent examples
            {"text": "what do you see", "intent": "describe_object", "entities": []},
            {"text": "describe what's there", "intent": "describe_object", "entities": []},
            {"text": "what tools are available", "intent": "describe_object", "entities": []},
            {"text": "tell me what you found", "intent": "describe_object", "entities": []},
            {"text": "what's on the workbench", "intent": "describe_object", "entities": [{"entity": "location", "value": "workbench"}]},
            
            # CONFIRM_ACTION intent examples
            {"text": "yes", "intent": "confirm_action", "entities": []},
            {"text": "yeah", "intent": "confirm_action", "entities": []},
            {"text": "correct", "intent": "confirm_action", "entities": []},
            {"text": "that's right", "intent": "confirm_action", "entities": []},
            {"text": "go ahead", "intent": "confirm_action", "entities": []},
            
            # CANCEL_ACTION intent examples  
            {"text": "no", "intent": "cancel_action", "entities": []},
            {"text": "stop", "intent": "cancel_action", "entities": []},
            {"text": "cancel", "intent": "cancel_action", "entities": []},
            {"text": "wrong", "intent": "cancel_action", "entities": []},
            {"text": "that's not right", "intent": "cancel_action", "entities": []},
            
            # SAFETY_ALERT intent examples
            {"text": "watch out", "intent": "safety_alert", "entities": []},
            {"text": "be careful", "intent": "safety_alert", "entities": []},
            {"text": "safety concern", "intent": "safety_alert", "entities": []},
            {"text": "danger", "intent": "safety_alert", "entities": []},
            {"text": "hard hat required", "intent": "safety_alert", "entities": [{"entity": "safety_concern", "value": "hard hat"}]},
            
            # STATUS_CHECK intent examples
            {"text": "what's your status", "intent": "status_check", "entities": []},
            {"text": "are you ready", "intent": "status_check", "entities": []},
            {"text": "how's it going", "intent": "status_check", "entities": []},
            {"text": "what are you doing", "intent": "status_check", "entities": []},
            {"text": "system status", "intent": "status_check", "entities": []}
        ]

    def _initialize_haystack_pipeline(self):
        """Initialize Haystack NLP pipeline for construction commands"""
        
        if not HAYSTACK_AVAILABLE:
            return
            
        try:
            # Initialize document store for construction knowledge
            self.document_store = InMemoryDocumentStore()
            
            # Create documents from training examples for semantic matching
            documents = []
            for example in self.training_examples:
                doc = Document(
                    content=example["text"],
                    meta={
                        "intent": example["intent"],
                        "entities": example["entities"],
                        "example_type": "training"
                    }
                )
                documents.append(doc)
            
            # Add construction domain knowledge documents
            construction_docs = [
                Document(content="A hammer is a tool used for hitting nails into wood or other materials", meta={"type": "tool_definition"}),
                Document(content="A screwdriver is used for turning screws with slotted or cross-shaped heads", meta={"type": "tool_definition"}),
                Document(content="A level is used to determine if a surface is horizontal or vertical", meta={"type": "tool_definition"}),
                Document(content="A drill is used for making holes in various materials", meta={"type": "tool_definition"}),
                Document(content="A measuring tape is used for measuring distances and dimensions", meta={"type": "tool_definition"}),
                Document(content="The workbench is the main work surface for construction tasks", meta={"type": "location_definition"}),
                Document(content="The toolbox stores and organizes construction tools", meta={"type": "location_definition"}),
            ]
            
            documents.extend(construction_docs)
            self.document_store.write_documents(documents)
            
            # Initialize text embedder
            self.embedder = SentenceTransformersTextEmbedder(model=self.model_name)
            
            # Create semantic retrieval pipeline
            self.retriever = InMemoryBM25Retriever(document_store=self.document_store)
            
            self.logger.info(f"✅ Haystack pipeline initialized with {len(documents)} documents")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Haystack pipeline: {e}")
            self.document_store = None
            self.embedder = None

    def parse_construction_command(self, text: str) -> NLUResult:
        """
        Parse construction voice command using Haystack NLP.
        
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
        
        if HAYSTACK_AVAILABLE and self.document_store:
            try:
                return self._haystack_parsing(text)
            except Exception as e:
                self.logger.error(f"Haystack parsing failed: {e}")
                return self._fallback_parsing(text)
        else:
            return self._mock_parsing(text)

    def _haystack_parsing(self, text: str) -> NLUResult:
        """Parse using Haystack semantic similarity"""
        
        # Retrieve most similar training examples
        results = self.retriever.run(query=text, top_k=5)
        
        if not results["documents"]:
            return self._fallback_parsing(text)
        
        # Find best matching intent based on semantic similarity
        best_match = results["documents"][0]
        intent = best_match.meta.get("intent", "unknown")
        confidence = min(best_match.score * 2, 1.0) if hasattr(best_match, 'score') else 0.8
        
        # Extract entities using pattern matching enhanced with semantic context
        entities = self._extract_entities_semantic(text, best_match.meta.get("entities", []))
        
        return NLUResult(
            intent=intent,
            confidence=confidence,
            entities=entities,
            text=text,
            metadata={
                "method": "haystack_semantic",
                "best_match": best_match.content,
                "match_score": getattr(best_match, 'score', 0.8)
            }
        )

    def _extract_entities_semantic(self, text: str, reference_entities: List[Dict]) -> List[Dict[str, Any]]:
        """Enhanced entity extraction using semantic context"""
        
        entities = []
        text_lower = text.lower()
        
        # Construction tools dictionary with variations
        tools_dict = {
            'hammer': ['hammer', 'framing hammer', 'claw hammer', 'ball peen'],
            'screwdriver': ['screwdriver', 'phillips screwdriver', 'flathead', 'phillips head'],
            'drill': ['drill', 'power drill', 'cordless drill', 'drill driver'],
            'level': ['level', 'spirit level', 'bubble level'],
            'wrench': ['wrench', 'adjustable wrench', 'socket wrench', 'box wrench'],
            'saw': ['saw', 'circular saw', 'hand saw', 'skill saw', 'reciprocating saw'],
            'tape': ['measuring tape', 'tape measure', 'tape', 'ruler'],
            'square': ['square', 'speed square', 'framing square', 'try square']
        }
        
        colors = ['red', 'blue', 'green', 'yellow', 'black', 'white', 'orange', 'gray']
        locations = ['workbench', 'toolbox', 'truck', 'floor', 'table', 'gang box', 'home position']
        sizes = ['small', 'large', 'big', '16 oz', '20 oz', '12 inch', '6 inch']
        
        # Extract tools with semantic matching
        for tool_base, variations in tools_dict.items():
            for variation in variations:
                if variation in text_lower:
                    entities.append({
                        'entity': 'tool_name',
                        'value': tool_base,
                        'confidence': 0.9,
                        'raw_value': variation
                    })
                    break
        
        # Extract other entities
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
        
        for size in sizes:
            if size in text_lower:
                entities.append({
                    'entity': 'size',
                    'value': size,
                    'confidence': 0.7
                })
        
        return entities

    def _fallback_parsing(self, text: str) -> NLUResult:
        """Fallback parsing when Haystack fails"""
        
        text_lower = text.lower()
        
        # Intent classification with improved patterns
        intent_patterns = {
            'pickup_tool': ['pick up', 'get', 'grab', 'bring', 'fetch', 'hand me', 'give me'],
            'find_tool': ['find', 'where', 'locate', 'search', 'look for', 'can you see'],
            'place_tool': ['put down', 'place', 'set down', 'drop off', 'return', 'put back'],
            'move_to_location': ['go to', 'move to', 'navigate', 'head', 'position'],
            'describe_object': ['what do you see', 'describe', 'what tools', 'tell me what'],
            'confirm_action': ['yes', 'yeah', 'correct', 'right', 'ok', 'okay', 'go ahead'],
            'cancel_action': ['no', 'stop', 'cancel', 'wrong', 'wait', 'hold on'],
            'safety_alert': ['watch out', 'careful', 'danger', 'safety', 'hazard'],
            'status_check': ['status', 'ready', 'doing', 'working', 'progress']
        }
        
        intent = "unknown"
        max_confidence = 0.0
        
        for intent_name, patterns in intent_patterns.items():
            for pattern in patterns:
                if pattern in text_lower:
                    confidence = 0.8 if len(pattern.split()) > 1 else 0.7
                    if confidence > max_confidence:
                        intent = intent_name
                        max_confidence = confidence
        
        # Extract entities
        entities = self._extract_entities_semantic(text, [])
        
        return NLUResult(
            intent=intent,
            confidence=max_confidence if intent != "unknown" else 0.3,
            entities=entities,
            text=text,
            metadata={'method': 'fallback_parsing'}
        )

    def _mock_parsing(self, text: str) -> NLUResult:
        """Mock parsing when Haystack not available"""
        
        self.logger.info(f"🤖 [MOCK HAYSTACK] Parsing: '{text}'")
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

    def add_training_examples(self, new_examples: List[Dict[str, Any]]):
        """
        Add new training examples to improve model performance.
        
        Parameters
        ----------
        new_examples : List[Dict[str, Any]]
            New training examples with intent and entities
        """
        
        if not new_examples:
            return
            
        self.training_examples.extend(new_examples)
        
        if HAYSTACK_AVAILABLE and self.document_store:
            # Add new documents to the store
            documents = []
            for example in new_examples:
                doc = Document(
                    content=example["text"],
                    meta={
                        "intent": example["intent"],
                        "entities": example.get("entities", []),
                        "example_type": "user_added"
                    }
                )
                documents.append(doc)
            
            self.document_store.write_documents(documents)
            
        self.logger.info(f"📚 Added {len(new_examples)} new training examples")

    def answer_construction_question(self, question: str) -> str:
        """
        Answer construction-related questions using RAG.
        
        Parameters
        ----------
        question : str
            Construction question to answer
            
        Returns
        -------
        str
            Answer based on construction knowledge
        """
        
        if not HAYSTACK_AVAILABLE or not self.document_store:
            return "Construction knowledge system not available"
        
        try:
            # Retrieve relevant construction documents
            results = self.retriever.run(query=question, top_k=3)
            
            if not results["documents"]:
                return "I don't have specific information about that construction topic"
            
            # Extract relevant information
            relevant_info = []
            for doc in results["documents"]:
                if doc.meta.get("type", "").endswith("_definition"):
                    relevant_info.append(doc.content)
            
            if relevant_info:
                return " ".join(relevant_info[:2])  # Return top 2 most relevant
            else:
                return "Let me help you with that construction task"
                
        except Exception as e:
            self.logger.error(f"Error answering construction question: {e}")
            return "Unable to process construction question"

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current NLU model"""
        
        return {
            'haystack_available': HAYSTACK_AVAILABLE,
            'model_name': self.model_name,
            'confidence_threshold': self.confidence_threshold,
            'training_examples': len(self.training_examples),
            'supported_intents': len(self.get_supported_intents()),
            'supported_entities': len(self.get_supported_entities()),
            'document_store_loaded': self.document_store is not None,
            'pipeline_initialized': self.embedder is not None
        }

    def cleanup(self):
        """Clean up Haystack NLP resources"""
        
        if self.document_store:
            try:
                self.document_store = None
            except Exception as e:
                self.logger.error(f"Error cleaning up document store: {e}")
        
        if self.embedder:
            try:
                del self.embedder
                self.embedder = None
            except Exception as e:
                self.logger.error(f"Error cleaning up embedder: {e}")
        
        self.logger.info("🤖 Haystack NLP Manager cleaned up")

# Alias for backward compatibility
ConstructionRasaNLP = ConstructionHaystackNLP