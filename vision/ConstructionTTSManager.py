#!/usr/bin/env python3
"""
Construction TTS Manager for Robot Speech Synthesis.

This module provides text-to-speech capabilities for construction clarification
dialogues, enabling spoken robot responses for hands-free interaction in
construction environments.

Integrates with the ConstructionClarificationManager to provide:
- Adaptive speech rate for construction site noise
- Professional construction terminology pronunciation
- Context-aware voice parameters
- Background/foreground speech modes
"""

import logging
import threading
import time
from typing import Optional, Dict, Any, List
from enum import Enum
import queue

# Try to import pyttsx3, fall back to mock if not available
try:
    import pyttsx3
    PYTTSX3_AVAILABLE = True
except ImportError:
    PYTTSX3_AVAILABLE = False
    logging.warning("pyttsx3 not available on macOS - TTS will use mock implementation (expected)")

# Check CoquiTTS availability without importing (Python 3.9 compatibility issue)
try:
    import importlib.util
    tts_spec = importlib.util.find_spec("TTS.api")
    COQUI_TTS_AVAILABLE = tts_spec is not None
except ImportError:
    COQUI_TTS_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class VoiceProfile(Enum):
    """Voice profiles for different construction contexts"""
    APPRENTICE_FRIENDLY = "apprentice_friendly"    # Slower, clearer speech
    PROFESSIONAL = "professional"                  # Standard construction pace
    URGENT = "urgent"                             # Faster for time-critical tasks
    NOISY_ENVIRONMENT = "noisy_environment"       # Louder, slower for construction sites

class TTSPriority(Enum):
    """Priority levels for TTS messages"""
    LOW = 1       # Background information
    NORMAL = 2    # Standard clarifications 
    HIGH = 3      # Important safety or urgent messages
    CRITICAL = 4  # Emergency or critical safety alerts

class ConstructionTTSManager:
    """
    Text-to-Speech manager optimized for construction HRI.
    
    Provides intelligent speech synthesis with construction-specific
    optimizations for noisy environments and professional terminology.
    
    Parameters
    ----------
    voice_profile : VoiceProfile, optional
        Default voice configuration, by default PROFESSIONAL
    enable_background_speech : bool, optional
        Allow background/non-blocking speech, by default True
    construction_mode : bool, optional
        Enable construction site optimizations, by default True
        
    Attributes
    ----------
    tts_engine : pyttsx3.Engine or None
        TTS engine instance (None if pyttsx3 unavailable)
    voice_profile : VoiceProfile
        Current voice configuration
    speech_queue : queue.Queue
        Queue for managing speech requests
    is_speaking : bool
        Flag indicating if TTS is currently active
    speech_thread : threading.Thread
        Background thread for speech processing
    """
    
    def __init__(self, 
                 voice_profile: VoiceProfile = VoiceProfile.PROFESSIONAL,
                 enable_background_speech: bool = True,
                 construction_mode: bool = True,
                 use_coqui: bool = True):
        
        self.logger = logging.getLogger(__name__)
        self.voice_profile = voice_profile
        self.enable_background_speech = enable_background_speech
        self.construction_mode = construction_mode
        self.use_coqui = use_coqui
        
        # Speech management
        self.speech_queue = queue.Queue()
        self.is_speaking = False
        self.speech_thread = None
        self.stop_speech = False
        
        # TTS engines
        self.tts_engine = None
        self.coqui_tts = None
        self.system_tts_available = False
        
        # Construction-specific pronunciations
        self.construction_pronunciations = {
            # Tool names with proper pronunciation
            "sawzall": "saw-zall",
            "skilsaw": "skill-saw", 
            "dewalt": "dee-walt",
            "milwaukee": "mil-wah-key",
            "makita": "mah-key-tah",
            "ryobi": "rye-oh-bee",
            
            # Construction terms
            "plumb": "plum",
            "square": "skware", 
            "rebar": "ree-bar",
            "drywall": "dry-wall",
            "2x4": "two by four",
            "2x6": "two by six",
            "2x8": "two by eight",
            "2x10": "two by ten",
            "2x12": "two by twelve",
            
            # Measurements
            "16\" O.C.": "sixteen inches on center",
            "24\" O.C.": "twenty four inches on center",
            "3/4\"": "three quarter inch",
            "1/2\"": "half inch",
            "1/4\"": "quarter inch"
        }
        
        # Initialize TTS engine
        self._initialize_tts_engine()
        
        # Start background speech processing if enabled
        if self.enable_background_speech:
            self._start_speech_thread()
        
        self.logger.info(f"✅ Construction TTS Manager initialized")
        self.logger.info(f"   Voice profile: {voice_profile.value}")
        self.logger.info(f"   Background speech: {enable_background_speech}")
        self.logger.info(f"   Construction mode: {construction_mode}")

    def _initialize_tts_engine(self):
        """Initialize the TTS engine with construction-optimized settings"""
        
        if self.use_coqui and COQUI_TTS_AVAILABLE:
            try:
                # Import TTS only when needed to avoid Python 3.9 compatibility issues
                from TTS.api import TTS
                # Initialize CoquiTTS with a compatible model for Python 3.9
                self.coqui_tts = TTS("tts_models/en/ljspeech/tacotron2-DDC", progress_bar=False, gpu=False)
                self.logger.info("✅ CoquiTTS engine initialized successfully")
                return
            except Exception as e:
                self.logger.error(f"Failed to initialize CoquiTTS engine: {e}")
                self.logger.info("Falling back to system TTS...")
                self.coqui_tts = None
        
        # Try system TTS (macOS) as fallback
        try:
            import subprocess
            result = subprocess.run(['say', '-v', '?'], capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                self.system_tts_available = True
                self.logger.info("✅ macOS system TTS available as fallback")
                return
        except:
            pass
        
        if not PYTTSX3_AVAILABLE:
            self.logger.warning("TTS engine not available on macOS - using mock implementation (expected)")
            self.tts_engine = None
            self.system_tts_available = False
            return

        try:
            self.tts_engine = pyttsx3.init()
            
            # Configure voice settings based on profile
            self._apply_voice_profile(self.voice_profile)
            
            self.logger.info("✅ TTS engine initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize TTS engine: {e}")
            self.tts_engine = None

    def _apply_voice_profile(self, profile: VoiceProfile):
        """Apply voice configuration based on profile"""
        
        if not self.tts_engine:
            return
            
        try:
            voices = self.tts_engine.getProperty('voices')
            
            if profile == VoiceProfile.APPRENTICE_FRIENDLY:
                # Slower, clearer speech for learning
                self.tts_engine.setProperty('rate', 160)  # Slower
                self.tts_engine.setProperty('volume', 0.9)
                # Prefer female voice if available (often perceived as more patient)
                for voice in voices:
                    if 'female' in voice.name.lower() or 'woman' in voice.name.lower():
                        self.tts_engine.setProperty('voice', voice.id)
                        break
                        
            elif profile == VoiceProfile.PROFESSIONAL:
                # Standard professional pace
                self.tts_engine.setProperty('rate', 180)
                self.tts_engine.setProperty('volume', 0.8)
                # Prefer male voice for construction authority
                for voice in voices:
                    if 'male' in voice.name.lower() or 'man' in voice.name.lower():
                        self.tts_engine.setProperty('voice', voice.id)
                        break
                        
            elif profile == VoiceProfile.URGENT:
                # Faster pace for time-critical situations
                self.tts_engine.setProperty('rate', 220)
                self.tts_engine.setProperty('volume', 1.0)
                
            elif profile == VoiceProfile.NOISY_ENVIRONMENT:
                # Optimized for construction site noise
                self.tts_engine.setProperty('rate', 140)  # Slower for clarity
                self.tts_engine.setProperty('volume', 1.0)  # Louder
            
            self.logger.info(f"Applied voice profile: {profile.value}")
            
        except Exception as e:
            self.logger.error(f"Failed to apply voice profile: {e}")

    def _start_speech_thread(self):
        """Start background thread for speech processing"""
        
        self.stop_speech = False
        self.speech_thread = threading.Thread(target=self._speech_processor, daemon=True)
        self.speech_thread.start()
        self.logger.info("🔊 Background speech thread started")

    def _speech_processor(self):
        """Background thread for processing speech queue"""
        
        while not self.stop_speech:
            try:
                # Get speech request with timeout
                speech_request = self.speech_queue.get(timeout=1.0)
                
                # Process the speech request
                self._speak_text(speech_request)
                
                # Mark task as done
                self.speech_queue.task_done()
                
            except queue.Empty:
                continue  # No speech requests, keep waiting
            except Exception as e:
                self.logger.error(f"Error in speech processor: {e}")

    def speak_clarification(self, text: str, 
                          priority: TTSPriority = TTSPriority.NORMAL,
                          voice_profile: Optional[VoiceProfile] = None,
                          blocking: bool = False) -> bool:
        """
        Speak a clarification message with construction optimizations.
        
        Parameters
        ----------
        text : str
            Text to speak
        priority : TTSPriority, optional
            Message priority level, by default NORMAL
        voice_profile : VoiceProfile, optional
            Override voice profile for this message
        blocking : bool, optional
            Whether to block until speech completes, by default False
            
        Returns
        -------
        bool
            True if speech was initiated successfully, False otherwise
            
        Examples
        --------
        >>> tts.speak_clarification("I found a framing hammer. Is that correct?")
        >>> tts.speak_clarification("SAFETY ALERT: Hard hat required", priority=TTSPriority.CRITICAL)
        """
        
        if not text.strip():
            return False
        
        # Apply construction pronunciation improvements
        processed_text = self._apply_construction_pronunciations(text)
        
        # Create speech request
        speech_request = {
            'text': processed_text,
            'original_text': text,
            'priority': priority,
            'voice_profile': voice_profile or self.voice_profile,
            'timestamp': time.time(),
            'blocking': blocking
        }
        
        if self.enable_background_speech and not blocking:
            # Add to queue for background processing
            try:
                # Clear queue if critical priority
                if priority == TTSPriority.CRITICAL:
                    self._clear_speech_queue()
                
                self.speech_queue.put(speech_request)
                self.logger.info(f"🔊 Queued speech ({priority.name}): '{text}'")
                return True
                
            except Exception as e:
                self.logger.error(f"Failed to queue speech: {e}")
                return False
        else:
            # Immediate/blocking speech
            return self._speak_text(speech_request)

    def _speak_text(self, speech_request: Dict[str, Any]) -> bool:
        """Internal method to actually speak text"""
        
        text = speech_request['text']
        voice_profile = speech_request.get('voice_profile')
        
        # CoquiTTS implementation
        if self.coqui_tts is not None:
            try:
                self.is_speaking = True
                
                # Generate audio with CoquiTTS
                import tempfile
                import os
                import subprocess
                
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                    wav_path = tmp_file.name
                
                # Generate speech
                self.coqui_tts.tts_to_file(text=text, file_path=wav_path)
                
                # Play audio file (macOS compatible)
                def play_audio():
                    try:
                        subprocess.run(['afplay', wav_path], check=True)
                    except subprocess.CalledProcessError as e:
                        self.logger.error(f"Failed to play CoquiTTS audio: {e}")
                    finally:
                        try:
                            os.unlink(wav_path)
                        except:
                            pass
                        self.is_speaking = False
                
                if speech_request.get('blocking', False):
                    play_audio()
                else:
                    speech_thread = threading.Thread(target=play_audio, daemon=True)
                    speech_thread.start()
                
                self.logger.info(f"🔊 [CoquiTTS] Speaking: '{speech_request['original_text']}'")
                return True
                
            except Exception as e:
                self.logger.error(f"CoquiTTS speech failed: {e}")
                return False
        
        # System TTS implementation (macOS 'say' command)
        if self.system_tts_available and not self.tts_engine:
            try:
                import subprocess
                self.is_speaking = True
                
                def system_speak():
                    try:
                        subprocess.run(['say', text], check=True)
                    except subprocess.CalledProcessError as e:
                        self.logger.error(f"System TTS failed: {e}")
                    finally:
                        self.is_speaking = False
                
                if speech_request.get('blocking', False):
                    system_speak()
                else:
                    speech_thread = threading.Thread(target=system_speak, daemon=True)
                    speech_thread.start()
                
                self.logger.info(f"🔊 [SYSTEM TTS] Speaking: '{speech_request['original_text']}'")
                return True
                
            except Exception as e:
                self.logger.error(f"System TTS failed: {e}")
                return False

        # Original pyttsx3 implementation
        if not self.tts_engine:
            # Mock implementation for when TTS not available
            self.logger.info(f"🔊 [MOCK TTS] Speaking: '{text}'")
            time.sleep(len(text) * 0.05)  # Simulate speech duration
            return True

        try:
            self.is_speaking = True
            
            # Apply voice profile if specified
            if voice_profile and voice_profile != self.voice_profile:
                self._apply_voice_profile(voice_profile)
            
            # Speak the text
            self.tts_engine.say(text)
            
            # Wait for speech to complete if blocking
            if speech_request.get('blocking', False):
                self.tts_engine.runAndWait()
            else:
                # Non-blocking speech
                def run_speech():
                    try:
                        self.tts_engine.runAndWait()
                    finally:
                        self.is_speaking = False
                
                speech_thread = threading.Thread(target=run_speech, daemon=True)
                speech_thread.start()
            
            self.logger.info(f"🔊 Speaking: '{speech_request['original_text']}'")
            return True
            
        except Exception as e:
            self.logger.error(f"TTS speech failed: {e}")
            return False
        finally:
            if speech_request.get('blocking', False):
                self.is_speaking = False

    def _apply_construction_pronunciations(self, text: str) -> str:
        """Apply construction-specific pronunciation improvements"""
        
        if not self.construction_mode:
            return text
        
        processed_text = text
        
        # Apply construction terminology pronunciations
        for term, pronunciation in self.construction_pronunciations.items():
            # Case-insensitive replacement
            processed_text = processed_text.replace(term, pronunciation)
            processed_text = processed_text.replace(term.title(), pronunciation.title())
            processed_text = processed_text.replace(term.upper(), pronunciation.upper())
        
        # Add pauses after tool names for clarity
        tool_names = ['hammer', 'screwdriver', 'wrench', 'drill', 'saw', 'level', 'square']
        for tool in tool_names:
            processed_text = processed_text.replace(f' {tool} ', f' {tool}, ')
        
        return processed_text

    def _clear_speech_queue(self):
        """Clear pending speech requests (used for critical messages)"""
        
        cleared_count = 0
        while True:
            try:
                self.speech_queue.get_nowait()
                cleared_count += 1
            except queue.Empty:
                break
        
        if cleared_count > 0:
            self.logger.info(f"🔊 Cleared {cleared_count} pending speech requests for critical message")

    def stop_current_speech(self):
        """Stop any currently playing speech"""
        
        if self.tts_engine and self.is_speaking:
            try:
                self.tts_engine.stop()
                self.is_speaking = False
                self.logger.info("🔊 Stopped current speech")
            except Exception as e:
                self.logger.error(f"Failed to stop speech: {e}")

    def set_voice_profile(self, profile: VoiceProfile):
        """Update the default voice profile"""
        
        old_profile = self.voice_profile
        self.voice_profile = profile
        self._apply_voice_profile(profile)
        
        self.logger.info(f"🔊 Voice profile updated: {old_profile.value} -> {profile.value}")

    def is_speech_active(self) -> bool:
        """Check if TTS is currently speaking"""
        return self.is_speaking

    def get_queue_size(self) -> int:
        """Get number of pending speech requests"""
        return self.speech_queue.qsize()

    def test_speech_capabilities(self) -> Dict[str, Any]:
        """Test TTS capabilities and return status information"""
        
        results = {
            'engine_available': self.tts_engine is not None or self.system_tts_available or self.coqui_tts is not None,
            'coqui_tts_available': self.coqui_tts is not None,
            'system_tts_available': self.system_tts_available,
            'pyttsx3_available': self.tts_engine is not None,
            'background_speech_enabled': self.enable_background_speech,
            'current_profile': self.voice_profile.value,
            'construction_mode': self.construction_mode,
            'queue_size': self.get_queue_size(),
            'is_speaking': self.is_speaking
        }
        
        if self.tts_engine:
            try:
                voices = self.tts_engine.getProperty('voices')
                results['available_voices'] = len(voices) if voices else 0
                results['current_rate'] = self.tts_engine.getProperty('rate')
                results['current_volume'] = self.tts_engine.getProperty('volume')
            except Exception as e:
                results['voice_info_error'] = str(e)
        
        return results

    def cleanup(self):
        """Clean up TTS resources"""
        
        # Stop speech thread
        if self.speech_thread and self.speech_thread.is_alive():
            self.stop_speech = True
            self.speech_thread.join(timeout=2.0)
        
        # Stop any current speech
        self.stop_current_speech()
        
        # Clean up TTS engine
        if self.tts_engine:
            try:
                self.tts_engine.stop()
            except Exception as e:
                self.logger.error(f"Error cleaning up TTS engine: {e}")
        
        self.logger.info("🔊 TTS Manager cleaned up")

# Integration helper function
def create_construction_tts(expertise_level: str = "journeyman") -> ConstructionTTSManager:
    """
    Create TTS manager configured for construction expertise level.
    
    Parameters
    ----------
    expertise_level : str
        User expertise level: apprentice, journeyman, foreman, master
        
    Returns
    -------
    ConstructionTTSManager
        Configured TTS manager
    """
    
    if expertise_level.lower() == "apprentice":
        profile = VoiceProfile.APPRENTICE_FRIENDLY
    elif expertise_level.lower() in ["foreman", "master"]:
        profile = VoiceProfile.PROFESSIONAL
    else:  # journeyman or default
        profile = VoiceProfile.PROFESSIONAL
    
    return ConstructionTTSManager(
        voice_profile=profile,
        enable_background_speech=True,
        construction_mode=True
    )