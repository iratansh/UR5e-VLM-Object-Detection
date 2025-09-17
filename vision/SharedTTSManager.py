#!/usr/bin/env python3
"""
Shared TTS Manager Singleton for Construction HRI System.

This module provides a singleton pattern for TTS management to prevent
repeated loading of CoquiTTS models across multiple system components.
"""

import logging
import threading
from typing import Optional, Any

class SharedTTSManager:
    """
    Singleton TTS manager for construction HRI system.
    
    Ensures only one instance of CoquiTTS models are loaded
    across all system components to optimize performance and memory usage.
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if not hasattr(self, '_initialized') or not self._initialized:
            self.logger = logging.getLogger(__name__)
            self.tts_manager = None
            self._model_cache = {}  # Cache for Tacotron2/vocoder models
            self._initialization_lock = threading.Lock()
            self._initialized = True
            self.logger.info("🔄 SharedTTS singleton with model caching created")
    
    def get_tts_manager(self, voice_profile=None, enable_background_speech=True, 
                       construction_mode=True, use_coqui=True, force_reload=False):
        """
        Get or create the shared TTS manager instance.
        
        Parameters
        ----------
        voice_profile : VoiceProfile, optional
            Voice profile to use for TTS
        enable_background_speech : bool, optional
            Enable background speech processing, by default True
        construction_mode : bool, optional
            Enable construction-specific pronunciations, by default True
        use_coqui : bool, optional
            Use CoquiTTS neural synthesis, by default True
        force_reload : bool, optional
            Force reload of TTS manager, by default False
            
        Returns
        -------
        ConstructionTTSManager
            Shared TTS manager instance
        """
        if self.tts_manager is None or force_reload:
            with self._initialization_lock:
                # Double-check pattern for thread safety
                if self.tts_manager is None or force_reload:
                    try:
                        from ConstructionTTSManager import ConstructionTTSManager, VoiceProfile
                        
                        # Use provided voice profile or default to PROFESSIONAL
                        if voice_profile is None:
                            voice_profile = VoiceProfile.PROFESSIONAL
                        
                        self.logger.info("🔄 Initializing TTS with model caching...")
                        self.tts_manager = ConstructionTTSManager(
                            voice_profile=voice_profile,
                            enable_background_speech=enable_background_speech,
                            construction_mode=construction_mode,
                            use_coqui=use_coqui
                        )
                        
                        # Cache models to prevent reloading
                        if hasattr(self.tts_manager, 'coqui_tts') and self.tts_manager.coqui_tts:
                            self._model_cache['coqui_tts'] = self.tts_manager.coqui_tts
                            self.logger.info("💾 Cached CoquiTTS models for reuse")
                        
                        # Log actual TTS engine being used
                        tts_engine = "Unknown"
                        if hasattr(self.tts_manager, 'coqui_tts') and self.tts_manager.coqui_tts:
                            tts_engine = "CoquiTTS (Neural)"
                        elif hasattr(self.tts_manager, 'system_tts_available') and self.tts_manager.system_tts_available:
                            tts_engine = "System TTS"
                        elif hasattr(self.tts_manager, 'tts_engine') and self.tts_manager.tts_engine:
                            tts_engine = "pyttsx3"
                        else:
                            tts_engine = "Mock TTS"
                        
                        self.logger.info(f"✅ Shared TTS manager initialized: {tts_engine}")
                        
                    except ImportError as e:
                        self.logger.error(f"Failed to load ConstructionTTSManager: {e}")
                        return None
                
        return self.tts_manager
    
    def cleanup(self):
        """Clean up shared resources."""
        if self.tts_manager:
            try:
                # Stop background speech thread if it exists
                if hasattr(self.tts_manager, 'background_speech_active'):
                    self.tts_manager.background_speech_active = False
                
                # Cleanup CoquiTTS if loaded
                if hasattr(self.tts_manager, 'coqui_tts') and self.tts_manager.coqui_tts:
                    self.tts_manager.coqui_tts = None
                
                self.logger.info("🧹 Shared TTS manager cleaned up")
            except Exception as e:
                self.logger.error(f"Error cleaning up TTS manager: {e}")
        
        # Reset singleton state for testing
        SharedTTSManager._instance = None
    
    @classmethod
    def reset_singleton(cls):
        """Reset singleton for testing purposes."""
        with cls._lock:
            if cls._instance:
                cls._instance.cleanup()
            cls._instance = None


def get_shared_tts_manager(**kwargs) -> Optional[Any]:
    """
    Convenience function to get shared TTS manager.
    
    Parameters
    ----------
    **kwargs
        Arguments passed to get_tts_manager()
    
    Returns
    -------
    Optional[ConstructionTTSManager]
        Shared TTS manager instance or None if unavailable
    """
    manager = SharedTTSManager()
    return manager.get_tts_manager(**kwargs)