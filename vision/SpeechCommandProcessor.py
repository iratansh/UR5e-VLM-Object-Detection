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
import speech_recognition as sr
import numpy as np
import re
import spacy
from transformers import pipeline
import sys

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
    recognizer : sr.Recognizer
        Speech recognition object
    microphone : sr.Microphone
        Microphone object for speech input
    command_queue : queue.Queue
        Queue for storing recognized commands
    listening : bool
        Flag indicating whether the processor is listening
    listen_thread : threading.Thread
        Thread for background listening
    command_patterns : Dict[str, str]
        Dictionary of command patterns
    """
    
    def __init__(self, model_name: str = "bert-base-uncased", use_gpu: bool = False, language: str = "en-US", timeout: float = 5.0):
        self.logger = logging.getLogger(__name__)
        self.language = language
        self.timeout = timeout
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        self.command_queue = queue.Queue()
        self.listening = False
        self.listen_thread = None
        
        # Command patterns
        self.command_patterns = {
            'pickup': r'pick\s+up\s+(?:the\s+)?(.+)',
            'grasp': r'grasp\s+(?:the\s+)?(.+)',
            'get': r'get\s+(?:the\s+)?(.+)'
        }
        
        # Calibrate microphone
        self._calibrate_noise()
        
        try:
            # Try to load the spaCy model
            try:
                self.nlp = spacy.load("en_core_web_sm")
                self.logger.info("Loaded spaCy model: en_core_web_sm")
            except OSError:
                # If model not found, try to download it
                self.logger.warning("spaCy model not found. Attempting to download...")
                import subprocess
                subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
                self.nlp = spacy.load("en_core_web_sm")
                self.logger.info("Downloaded and loaded spaCy model: en_core_web_sm")
            
            try:
                self.intent_classifier = pipeline("text-classification", model=model_name, use_gpu=use_gpu)
                self.logger.info(f"Loaded intent classifier: {model_name}")
            except Exception as e:
                self.logger.warning(f"Could not load intent classifier: {e}")
                self.intent_classifier = None
            
        except Exception as e:
            self.logger.error(f"Error initializing NLP components: {e}")
            # Fallback to minimal functionality
            self.nlp = None
            self.intent_classifier = None
        
        self.command_history = []
        
    def _calibrate_noise(self):
        """
        Calibrate microphone for ambient noise levels.
        
        Notes
        -----
        Uses energy threshold adjustment to filter background noise
        """
        try:
            with self.microphone as source:
                self.logger.info("Calibrating for ambient noise...")
                self.recognizer.adjust_for_ambient_noise(source, duration=1)
                self.logger.info("✅ Noise calibration complete")
        except Exception as e:
            self.logger.error(f"Failed to calibrate noise: {e}")
    
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
        """Background listening loop"""
        while self.listening:
            try:
                with self.microphone as source:
                    # Listen for audio with timeout
                    self.audio = self.recognizer.listen(source, timeout=self.timeout)
                    self.logger.info("Listening for commands...")
                
                # Recognize speech
                try:
                    command = self.recognizer.recognize_google(self.audio, language=self.language)
                    self.logger.info(f"Recognized command: {command}")
                    self.command_queue.put(command.lower())
                except sr.UnknownValueError:
                    self.logger.warning("Could not understand audio")
                except sr.RequestError as e:
                    self.logger.error(f"Recognition service error: {e}")
                    
            except sr.WaitTimeoutError:
                pass 
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
        Uses Google Speech Recognition API for transcription
        """
        try:
            return self.command_queue.get_nowait()
        except queue.Empty:
            return None
    
    def parse_object_query(self, command: str) -> List[str]:
        """
        Extract object queries from command text.
        
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
        >>> parse_object_query("pick up the water bottle")
        ['water bottle']
        """
        if not command:
            return []
            
        queries = []
        for pattern in self.command_patterns.values():
            match = re.search(pattern, command)
            if match:
                object_name = match.group(1).strip()
                queries.append(object_name)
                break
                
        return queries

    def process_command(self, command: str) -> Optional[Dict]:
        """
        Process a natural language command.
        
        Parameters
        ----------
        command : str
            Natural language command string
            
        Returns
        -------
        Optional[Dict]
            Dictionary containing:
            - intent: Command intent (e.g., "pick", "move")
            - target: Target object or location
            - parameters: Additional command parameters
            Or None if command cannot be processed
            
        Notes
        -----
        Processing steps:
        1. Tokenize and parse command
        2. Classify command intent
        3. Extract target and parameters
        4. Validate command safety
        """
        if not command:
            return None
        
        doc = self.nlp(command.lower())
        
        command_info = {
            'intent': None,
            'target': None,
            'parameters': {}
        }
        
        for intent, pattern in self.command_patterns.items():
            match = re.search(pattern, command.lower())
            if match:
                command_info['intent'] = intent
                command_info['target'] = match.group(1).strip()
                break
        
        # If no pattern matched, try more complex NLP extraction
        if not command_info['intent']:
            verbs = [token.lemma_ for token in doc if token.pos_ == "VERB"]
            if verbs:
                command_info['intent'] = verbs[0]
            
            nouns = [chunk.text for chunk in doc.noun_chunks]
            if nouns:
                command_info['target'] = nouns[-1]  # Last noun chunk is often the target
        
        if command_info['intent'] and command_info['target']:
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
        including stopping the listening thread.
        """
        self.stop_listening()
        
        # Release microphone resources if needed
        if hasattr(self, 'microphone') and hasattr(self.microphone, 'close'):
            try:
                self.microphone.close()
            except Exception as e:
                logging.error(f"Error closing microphone: {e}")
        
        logging.info("Speech command processor cleaned up")