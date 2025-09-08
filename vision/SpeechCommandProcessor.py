"""
Speech Command Processing for Robot Control.

This module provides natural language command processing for robot control:
- Command intent recognition
- Object reference resolution
- Action parameter extraction
- Command validation and safety checks

The processor enables natural interaction with the robot system
through voice commands like "pick up the red cup" or "move to home position".
"""

import logging
import queue
import time
from typing import List, Tuple, Optional, Dict
import threading
import numpy as np
import re
import warnings
import tempfile
import wave
import io
import sys

# Suppress known compatibility warnings
warnings.filterwarnings("ignore", message=".*TypedStorage is deprecated.*")
warnings.filterwarnings("ignore", message=".*Model.*was trained with spaCy.*")
warnings.filterwarnings("ignore", message=".*weights.*were not initialized.*")

import spacy
from transformers import pipeline
import whisper
import sounddevice as sd

logging.basicConfig(level=logging.INFO)

class SpeechCommandProcessor:
    """
    Natural language command processor for robot control.
    
    This class processes voice commands to extract robot control intents,
    target objects, and action parameters using NLP techniques.
    
    Parameters
    ----------
    model_name : str, optional
        Name of the language model to use, by default "bert-base-uncased"
    use_gpu : bool, optional
        Whether to use GPU acceleration, by default False
    language : str, optional
        Language for speech recognition, by default "en-US"
    timeout : float, optional
        Recognition timeout in seconds, by default 5.0
    whisper_model : str, optional
        Whisper model size to use, by default "base"
        
    Attributes
    ----------
    nlp : spacy.Language
        SpaCy NLP pipeline
    intent_classifier : pipeline
        Transformer pipeline for intent classification
    command_history : List[Dict]
        History of processed commands
    logger : logging.Logger
        Logger for logging messages
    language : str
        Language for speech recognition
    timeout : float
        Recognition timeout in seconds
    whisper_model : whisper.Whisper
        Whisper ASR model for speech recognition
    sample_rate : int
        Audio sample rate for recording (16kHz)
    command_queue : queue.Queue
        Queue for storing recognized commands
    listening : bool
        Flag indicating whether the processor is listening
    listen_thread : threading.Thread
        Thread for background listening
    command_patterns : Dict[str, str]
        Dictionary of command patterns
    """
    
    def __init__(self, model_name: str = "bert-base-uncased", use_gpu: bool = False, language: str = "en-US", timeout: float = 5.0, whisper_model: str = "base"):
        self.logger = logging.getLogger(__name__)
        self.language = language
        self.timeout = timeout
        self.sample_rate = 16000  # 16kHz for Whisper
        self.command_queue = queue.Queue()
        self.listening = False
        self.listen_thread = None
        
        # Command patterns
        self.command_patterns = {
            'pickup': r'pick\s+up\s+(?:the\s+)?(.+)',
            'grasp': r'grasp\s+(?:the\s+)?(.+)',
            'get': r'get\s+(?:the\s+)?(.+)'
        }
        
        # Load Whisper model
        try:
            self.logger.info(f"Loading Whisper {whisper_model} model...")
            self.whisper_model = whisper.load_model(whisper_model)
            self.logger.info("✅ Whisper model loaded successfully")
        except Exception as e:
            self.logger.error(f"Failed to load Whisper model: {e}")
            raise
        
        # Initialize Rasa NLP for construction-specific intent recognition
        try:
            from ConstructionRasaNLP import ConstructionRasaNLP
            self.rasa_nlp = ConstructionRasaNLP(
                confidence_threshold=0.6
            )
            self.logger.info("✅ Rasa NLP initialized for construction commands")
        except ImportError as e:
            self.logger.warning(f"Rasa NLP not available: {e}")
            self.rasa_nlp = None
        
        # Fallback to spaCy + transformers if Rasa fails
        try:
            # Try to load the spaCy model
            try:
                self.nlp = spacy.load("en_core_web_sm")
                self.logger.info("Loaded spaCy model: en_core_web_sm as fallback")
            except OSError:
                # If model not found, try to download it
                self.logger.warning("spaCy model not found. Attempting to download...")
                import subprocess
                subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
                self.nlp = spacy.load("en_core_web_sm")
                self.logger.info("Downloaded and loaded spaCy model: en_core_web_sm")
            
            try:
                # Suppress transformers warnings during pipeline creation
                import transformers
                transformers.logging.set_verbosity_error()
                self.intent_classifier = pipeline("text-classification", model=model_name, device=-1)  # Force CPU to avoid GPU warnings
                self.logger.info(f"Loaded intent classifier: {model_name} as fallback")
            except Exception as e:
                self.logger.warning(f"Could not load intent classifier: {e}")
                self.intent_classifier = None
            
        except Exception as e:
            self.logger.error(f"Error initializing NLP components: {e}")
            # Fallback to minimal functionality
            self.nlp = None
            self.intent_classifier = None
        
        self.command_history = []
        # Precompiled patterns for construction lingo normalization
        self._dimension_lumber_pattern = re.compile(r"(\b(?:one|two|three|four|five|six|seven|eight|nine|ten|\d)\s*(?:x|by|\-)?\s*(?:one|two|three|four|five|six|seven|eight|nine|ten|\d)\b)")
        self._word_to_num = {
            'one': '1', 'two': '2', 'three': '3', 'four': '4', 'five': '5',
            'six': '6', 'seven': '7', 'eight': '8', 'nine': '9', 'ten': '10'
        }
        self._fraction_pattern = re.compile(r"(\b(?:one|two|three|four|five|six|seven|eight|nine|ten|\d)\s*(?:/|\s+over\s+)\s*(?:one|two|three|four|five|six|seven|eight|nine|ten|\d)\b)")
        self._multi_word_materials = {
            'three quarter inch plywood': '3/4 inch plywood',
            'half inch plywood': '1/2 inch plywood',
            'quarter inch plywood': '1/4 inch plywood'
        }

    def normalize_construction_lingo(self, text: str) -> str:
        """Normalize construction site lingo into canonical detection-friendly forms.

        Examples:
        - "two by four" -> "2x4"
        - "2 by 4" -> "2x4"
        - "three quarter inch plywood" -> "3/4 inch plywood"
        - "grab the two-by-four next to the saw" -> "grab the 2x4 next to the saw"

        This helps downstream OWL-ViT query expansion and RAG retrieval.
        """
        if not text:
            return text
        original = text
        lowered = text.lower()

        # Multi-word material standardization (e.g., thickness + material)
        for phrase, canonical in self._multi_word_materials.items():
            if phrase in lowered:
                lowered = lowered.replace(phrase, canonical)

        # Dimension lumber normalization (two by four => 2x4)
        def _normalize_dimension(match: re.Match) -> str:
            token = match.group(1)
            parts = re.split(r"x|by|\-", token)
            if len(parts) == 2:
                a, b = parts
                a = a.strip()
                b = b.strip()
                a_num = self._word_to_num.get(a, a)
                b_num = self._word_to_num.get(b, b)
                if a_num.isdigit() and b_num.isdigit():
                    return f"{a_num}x{b_num}"
            return token

        lowered = self._dimension_lumber_pattern.sub(_normalize_dimension, lowered)

        # Fractions (one over two -> 1/2)
        def _normalize_fraction(m: re.Match) -> str:
            frac = m.group(1)
            parts = re.split(r"/|over", frac)
            if len(parts) == 2:
                a = parts[0].strip()
                b = parts[1].strip()
                a_num = self._word_to_num.get(a, a)
                b_num = self._word_to_num.get(b, b)
                if a_num.isdigit() and b_num.isdigit():
                    return f"{a_num}/{b_num}"
            return frac

        lowered = self._fraction_pattern.sub(_normalize_fraction, lowered)

        # Return with original casing preserved only if unchanged
        return lowered
        
    def _record_audio(self, duration: float = None) -> np.ndarray:
        """
        Record audio using sounddevice for Whisper processing.
        
        Parameters
        ----------
        duration : float, optional
            Recording duration in seconds, by default timeout value
            
        Returns
        -------
        np.ndarray
            Audio data as numpy array
            
        Notes
        -----
        Records at 16kHz sample rate optimized for Whisper ASR
        """
        if duration is None:
            duration = self.timeout
        
        try:
            self.logger.info("🎤 Recording audio...")
            audio_data = sd.rec(
                int(duration * self.sample_rate),
                samplerate=self.sample_rate,
                channels=1,
                dtype=np.float32
            )
            sd.wait()  # Wait for recording to finish
            return audio_data.flatten()
        except Exception as e:
            self.logger.error(f"Audio recording failed: {e}")
            return np.array([])
    
    def start_listening(self):
        """
        Start listening for voice commands.
        
        Notes
        -----
        Non-blocking operation, use get_command() to retrieve results
        """
        if not self.listening:
            self.listening = True
            self.listen_thread = threading.Thread(target=self._listen_loop, daemon=True)
            self.listen_thread.start()
            self.logger.info("Started listening for voice commands")
    
    def stop_listening(self):
        """
        Stop listening for voice commands.
        """
        self.listening = False
        if self.listen_thread:
            self.listen_thread.join(timeout=1.0)
        self.logger.info("Stopped listening")
    
    def _listen_loop(self):
        """Background listening loop using Whisper ASR"""
        while self.listening:
            try:
                # Record audio
                audio_data = self._record_audio()
                
                if len(audio_data) > 0:
                    # Transcribe with Whisper
                    try:
                        result = self.whisper_model.transcribe(
                            audio_data,
                            language=self.language.split('-')[0] if '-' in self.language else self.language,
                            fp16=False  # Use FP32 for compatibility
                        )
                        
                        command = result['text'].strip()
                        if command:
                            self.logger.info(f"🗣️ Whisper recognized: {command}")
                            self.command_queue.put(command.lower())
                        else:
                            self.logger.debug("No speech detected")
                            
                    except Exception as e:
                        self.logger.error(f"Whisper transcription error: {e}")
                else:
                    self.logger.debug("No audio recorded")
                    
            except Exception as e:
                self.logger.error(f"Listening error: {e}")
                time.sleep(0.1)
    
    def get_command(self) -> Optional[str]:
        """
        Get the latest voice command.
        
        Returns
        -------
        Optional[str]
            Recognized command text, or None if recognition failed
            
        Notes
        -----
        Uses Whisper ASR for robust speech transcription, optimized for construction site environments
        """
        try:
            return self.command_queue.get_nowait()
        except queue.Empty:
            return None
    
    def parse_object_query(self, command: str) -> List[str]:
        """
        Extract object queries from command text using Rasa NLP.
        
        Parameters
        ----------
        command : str
            Command text to parse
            
        Returns
        -------
        List[str]
            List of object queries extracted from command
            
        Examples
        --------
        >>> parse_object_query("pick up the framing hammer")
        ['framing hammer']
        """
        if not command:
            return []
        
        queries = []
        
        # Primary: Use Rasa NLP for tool extraction
        if self.rasa_nlp:
            try:
                tool_references = self.rasa_nlp.extract_tool_references(command)
                if tool_references:
                    queries.extend(tool_references)
                    return queries
            except Exception as e:
                self.logger.error(f"Rasa tool extraction failed: {e}")
        
        # Fallback: Use original pattern matching
        for pattern in self.command_patterns.values():
            match = re.search(pattern, command)
            if match:
                object_name = match.group(1).strip()
                queries.append(object_name)
                break
                
        return queries

    def process_command(self, command: str) -> Optional[Dict]:
        """
        Process a natural language command using Rasa NLP.
        
        Parameters
        ----------
        command : str
            Natural language command string
            
        Returns
        -------
        Optional[Dict]
            Dictionary containing:
            - intent: Command intent (e.g., "pickup_tool", "find_tool")
            - target: Target object or location
            - parameters: Additional command parameters
            - entities: Extracted entities with metadata
            - confidence: Intent confidence score
            Or None if command cannot be processed
            
        Notes
        -----
        Processing steps:
        1. Use Rasa NLP for construction-specific intent recognition
        2. Extract entities (tools, locations, attributes)
        3. Fallback to spaCy/regex if Rasa unavailable
        4. Validate command safety
        """
        if not command:
            return None
        
        # Normalize construction lingo early so NLU & detection get canonical forms
        normalized_command = self.normalize_construction_lingo(command)

        command_info = {
            'intent': None,
            'target': None,
            'parameters': {},
            'entities': [],
            'confidence': 0.0,
            'raw_command': command,
            'normalized_command': normalized_command
        }
        
        # Primary: Use Rasa NLP for construction commands
        if self.rasa_nlp:
            try:
                # Feed normalized form to Rasa if it differs (fallback to raw if needed)
                parse_input = normalized_command if normalized_command != command else command
                nlu_result = self.rasa_nlp.parse_construction_command(parse_input)
                
                command_info['intent'] = nlu_result.intent
                command_info['confidence'] = nlu_result.confidence
                command_info['entities'] = nlu_result.entities
                
                # Extract target from entities
                for entity in nlu_result.entities:
                    if entity.get('entity') == 'tool_name':
                        command_info['target'] = entity.get('value')
                        break
                    elif entity.get('entity') == 'location':
                        if not command_info['target']:  # Location as fallback target
                            command_info['target'] = entity.get('value')
                
                # Extract parameters from other entities
                for entity in nlu_result.entities:
                    entity_type = entity.get('entity')
                    entity_value = entity.get('value')
                    
                    if entity_type in ['color', 'size', 'material', 'urgency']:
                        command_info['parameters'][entity_type] = entity_value
                    elif entity_type == 'location':
                        command_info['parameters']['location'] = entity_value
                
                self.logger.info(f"🤖 Rasa NLP: {nlu_result.intent} ({nlu_result.confidence:.2f}) -> {command_info['target']}")
                
                if command_info['intent'] and command_info['intent'] != 'unknown':
                    self.command_history.append(command_info)
                    return command_info
                    
            except Exception as e:
                self.logger.error(f"Rasa NLP processing failed: {e}")
        
        # Fallback: Use original spaCy + regex approach
        if self.nlp:
            try:
                doc = self.nlp(normalized_command.lower())
                
                # Pattern matching fallback
                for intent, pattern in self.command_patterns.items():
                    match = re.search(pattern, normalized_command.lower())
                    if match:
                        command_info['intent'] = intent
                        command_info['target'] = match.group(1).strip()
                        command_info['confidence'] = 0.7
                        break
                
                # Complex NLP extraction fallback
                if not command_info['intent']:
                    verbs = [token.lemma_ for token in doc if token.pos_ == "VERB"]
                    if verbs:
                        command_info['intent'] = verbs[0]
                        command_info['confidence'] = 0.5
                    
                    nouns = [chunk.text for chunk in doc.noun_chunks]
                    if nouns:
                        command_info['target'] = nouns[-1]  # Last noun chunk is often the target
                
                if command_info['intent'] and command_info['target']:
                    self.command_history.append(command_info)
                    return command_info
                    
            except Exception as e:
                self.logger.error(f"Fallback NLP processing failed: {e}")
        
        # Final fallback: basic pattern matching
        for intent, pattern in self.command_patterns.items():
            match = re.search(pattern, normalized_command.lower())
            if match:
                command_info['intent'] = intent
                command_info['target'] = match.group(1).strip()
                command_info['confidence'] = 0.3
                self.command_history.append(command_info)
                return command_info
        
        return None

    def extract_object_reference(self, command: str) -> Optional[Dict]:
        """
        Extract object reference from command.
        
        Parameters
        ----------
        command : str
            Natural language command
            
        Returns
        -------
        Optional[Dict]
            Dictionary containing:
            - object: Main object name
            - attributes: List of object attributes (color, size, etc.)
            - spatial: Spatial reference (e.g., "on the left")
            Or None if no object reference found
        """
        if not command:
            return None
        
        doc = self.nlp(command.lower())
        
        result = {
            'object': None,
            'attributes': [],
            'spatial': None
        }
        
        # Find noun chunks that might refer to objects
        for chunk in doc.noun_chunks:
            # Skip pronouns and determiners
            if chunk.root.pos_ == 'NOUN' or chunk.root.pos_ == 'PROPN':
                # Found a potential object reference
                result['object'] = chunk.root.text
                
                for token in chunk:
                    if token.pos_ == 'ADJ':
                        result['attributes'].append(token.text)
        
        # Look for spatial references
        spatial_patterns = [
            r'on the (left|right|top|bottom)',
            r'(left|right|top|bottom) side',
            r'in the (center|middle|front|back)',
            r'(near|next to|beside) the'
        ]
        
        for pattern in spatial_patterns:
            match = re.search(pattern, command.lower())
            if match:
                result['spatial'] = match.group(0)
                break
        
        if result['object']:
            return result
        
        return None

    def validate_command(self, command_info: Dict) -> bool:
        """
        Validate command for safety and executability.
        
        Parameters
        ----------
        command_info : Dict
            Parsed command information
            
        Returns
        -------
        bool
            True if command is valid and safe, False otherwise
            
        Notes
        -----
        Checks:
        - Command intent is supported
        - Target object is specified
        - No unsafe operations requested
        """
        if not command_info:
            return False
        
        supported_intents = list(self.command_patterns.keys()) + ['move', 'stop', 'release']
        if command_info['intent'] not in supported_intents:
            self.logger.warning(f"Unsupported intent: {command_info['intent']}")
            return False
        
        # Check if target is specified for actions that need it
        if command_info['intent'] in ['pickup', 'grasp', 'get'] and not command_info['target']:
            self.logger.warning(f"No target specified for {command_info['intent']} action")
            return False
        
        # Check for unsafe commands (could be expanded)
        unsafe_targets = ['human', 'person', 'face', 'hand', 'arm']
        if command_info.get('target') and any(word in command_info['target'].lower() for word in unsafe_targets):
            self.logger.warning(f"Unsafe target detected: {command_info['target']}")
            return False
        
        return True

    def get_command_history(self) -> List[Dict]:
        """
        Get history of processed commands.
        
        Returns
        -------
        List[Dict]
            List of processed command information dictionaries
        """
        return self.command_history

    def cleanup(self):
        """
        Clean up resources used by the speech processor.
        
        This method ensures proper cleanup of all resources,
        including stopping the listening thread and releasing Whisper model.
        """
        self.stop_listening()
        
        # Clear Whisper model from memory if needed
        if hasattr(self, 'whisper_model'):
            try:
                del self.whisper_model
            except Exception as e:
                logging.error(f"Error cleaning up Whisper model: {e}")
        
        logging.info("Speech command processor cleaned up")