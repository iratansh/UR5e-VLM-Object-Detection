#!/usr/bin/env python3
"""
Shared RAG Manager Singleton for Construction HRI System.

This module provides a singleton pattern for RAG management to prevent
repeated loading of models and databases across multiple system components.
"""

import logging
import tempfile
import threading
from typing import Optional, Dict, Any

class SharedRAGManager:
    """
    Singleton RAG manager for construction HRI system.
    
    Ensures only one instance of RAG models and databases are loaded
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
            self.rag_manager = None
            self.temp_dir = None
            self._initialized = True
            self.logger.info("🔄 SharedRAGManager singleton created")
    
    def get_rag_manager(self, force_reload: bool = False):
        """
        Get or create the shared RAG manager instance.
        
        Parameters
        ----------
        force_reload : bool, optional
            Force reload of RAG manager, by default False
            
        Returns
        -------
        EnhancedConstructionRAG
            Shared RAG manager instance
        """
        if self.rag_manager is None or force_reload:
            try:
                from EnhancedConstructionRAG import EnhancedConstructionRAG
                
                # Create persistent temp directory if needed
                if self.temp_dir is None:
                    self.temp_dir = tempfile.mkdtemp(prefix="construction_rag_")
                    self.logger.info(f"📁 Created persistent RAG directory: {self.temp_dir}")
                
                self.rag_manager = EnhancedConstructionRAG(db_path=self.temp_dir)
                self.logger.info("✅ Shared RAG manager initialized")
                
            except ImportError as e:
                self.logger.error(f"Failed to load EnhancedConstructionRAG: {e}")
                return None
                
        return self.rag_manager
    
    def cleanup(self):
        """Clean up shared resources."""
        if self.rag_manager:
            try:
                self.rag_manager.cleanup()
                self.logger.info("🧹 Shared RAG manager cleaned up")
            except Exception as e:
                self.logger.error(f"Error cleaning up RAG manager: {e}")
        
        if self.temp_dir:
            import shutil
            try:
                shutil.rmtree(self.temp_dir)
                self.logger.info(f"🗑️  Removed temp directory: {self.temp_dir}")
            except Exception as e:
                self.logger.error(f"Error removing temp directory: {e}")
        
        # Reset singleton state for testing
        SharedRAGManager._instance = None
    
    @classmethod
    def reset_singleton(cls):
        """Reset singleton for testing purposes."""
        with cls._lock:
            if cls._instance:
                cls._instance.cleanup()
            cls._instance = None


def get_shared_rag_manager() -> Optional[Any]:
    """
    Convenience function to get shared RAG manager.
    
    Returns
    -------
    Optional[EnhancedConstructionRAG]
        Shared RAG manager instance or None if unavailable
    """
    manager = SharedRAGManager()
    return manager.get_rag_manager()