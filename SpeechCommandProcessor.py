import logging
import queue
import time
from typing import List, Tuple, Optional
import threading
import speech_recognition as sr
import numpy as np

logging.basicConfig(level=logging.INFO)

class SpeechCommandProcessor:
    """Process voice commands for object detection"""
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        self.command_queue = queue.Queue()
        self.listening = False
        self.listen_thread = None
        
        # Calibrate microphone
        try:
            with self.microphone as source:
                self.recognizer.adjust_for_ambient_noise(source)
            self.logger.info("Microphone calibrated")
        except Exception as e:
            self.logger.warning(f"Microphone setup failed: {e}")
    
    def start_listening(self):
        """Start listening for voice commands in background thread"""
        if not self.listening:
            self.listening = True
            self.listen_thread = threading.Thread(target=self._listen_loop, daemon=True)
            self.listen_thread.start()
            self.logger.info("Started listening for voice commands")
    
    def stop_listening(self):
        """Stop listening for voice commands"""
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
                    audio = self.recognizer.listen(source, timeout=1.0, phrase_time_limit=3.0)
                
                # Recognize speech
                try:
                    command = self.recognizer.recognize_google(audio, language='en-US')
                    print(f"Input speech: '{command}'") 
                    self.command_queue.put(command.lower())
                    self.logger.info(f"🗣️ Voice command: '{command}'")
                except sr.UnknownValueError:
                    pass  # No speech detected
                except sr.RequestError as e:
                    self.logger.warning(f"Speech recognition error: {e}")
                    
            except sr.WaitTimeoutError:
                pass  # Timeout, continue listening
            except Exception as e:
                self.logger.error(f"Listening error: {e}")
                time.sleep(0.1)
    
    def get_command(self) -> Optional[str]:
        """Get latest voice command if available"""
        try:
            return self.command_queue.get_nowait()
        except queue.Empty:
            return None
    
    def parse_object_query(self, command: str) -> List[str]:
        """Parse voice command into object queries"""
        # Simple parsing - can be enhanced with NLP
        trigger_words = ['find', 'get', 'pick up', 'grab', 'locate', 'detect']
        
        if any(word in command for word in trigger_words):
            # Extract object descriptions
            queries = []
            
            # Common object mappings
            object_map = {
                'bottle': ['bottle', 'water bottle', 'plastic bottle'],
                'cup': ['cup', 'mug', 'coffee cup'],
                'phone': ['phone', 'smartphone', 'cell phone', 'mobile'],
                'book': ['book', 'notebook'],
                'apple': ['apple', 'red apple'],
                'mouse': ['mouse', 'computer mouse'],
                'remote': ['remote', 'remote control'],
                'keys': ['keys', 'car keys']
            }
            
            for key, variations in object_map.items():
                if any(var in command for var in variations):
                    queries.extend(variations[:2])  
    
            return queries if queries else ['graspable object']
        
        return []