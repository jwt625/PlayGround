"""
Transcription Service Abstraction

This module provides a base class for transcription services and implementations
for both Whisper (local) and AssemblyAI (cloud) transcription providers.
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, Callable
import os
import time
from datetime import datetime


class TranscriptionResult:
    """Standardized transcription result"""
    
    def __init__(self, text: str, confidence: float = 1.0, is_final: bool = True, 
                 language: str = "unknown", processing_time: float = 0.0, 
                 segments: list = None, metadata: dict = None):
        self.text = text
        self.confidence = confidence
        self.is_final = is_final
        self.language = language
        self.processing_time = processing_time
        self.segments = segments or []
        self.metadata = metadata or {}
        self.timestamp = datetime.now().isoformat()
    
    def to_dict(self):
        """Convert to dictionary for JSON serialization"""
        return {
            'text': self.text,
            'confidence': self.confidence,
            'is_final': self.is_final,
            'language': self.language,
            'processing_time': self.processing_time,
            'segments': self.segments,
            'metadata': self.metadata,
            'timestamp': self.timestamp
        }


class TranscriptionService(ABC):
    """Abstract base class for transcription services"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.is_initialized = False
        self.stats = {
            'total_processed': 0,
            'total_processing_time': 0.0,
            'average_processing_time': 0.0
        }
    
    @abstractmethod
    def initialize(self) -> bool:
        """Initialize the transcription service"""
        pass
    
    @abstractmethod
    def transcribe_audio_data(self, audio_data: bytes, source: str = 'microphone') -> Optional[TranscriptionResult]:
        """Transcribe audio data (bytes) and return result"""
        pass
    
    @abstractmethod
    def transcribe_file(self, file_path: str) -> Optional[TranscriptionResult]:
        """Transcribe an audio file and return result"""
        pass
    
    @abstractmethod
    def supports_streaming(self) -> bool:
        """Return True if this service supports real-time streaming"""
        pass
    
    def start_streaming(self, callback: Callable[[TranscriptionResult], None], 
                       sample_rate: int = 16000) -> bool:
        """Start streaming transcription (if supported)"""
        if not self.supports_streaming():
            raise NotImplementedError("This transcription service does not support streaming")
        return False
    
    def stop_streaming(self):
        """Stop streaming transcription (if supported)"""
        if not self.supports_streaming():
            raise NotImplementedError("This transcription service does not support streaming")
    
    def cleanup(self):
        """Clean up resources"""
        pass
    
    def get_stats(self) -> Dict[str, Any]:
        """Get transcription statistics"""
        if self.stats['total_processed'] > 0:
            self.stats['average_processing_time'] = (
                self.stats['total_processing_time'] / self.stats['total_processed']
            )
        return self.stats.copy()
    
    def _update_stats(self, processing_time: float):
        """Update internal statistics"""
        self.stats['total_processed'] += 1
        self.stats['total_processing_time'] += processing_time


class TranscriptionServiceFactory:
    """Factory for creating transcription services"""
    
    @staticmethod
    def create_service(provider: str, config: Dict[str, Any] = None) -> TranscriptionService:
        """Create a transcription service instance"""
        if provider.lower() == 'whisper':
            try:
                from .whisper_transcription_service import WhisperTranscriptionService
            except ImportError:
                from whisper_transcription_service import WhisperTranscriptionService
            return WhisperTranscriptionService(config)
        elif provider.lower() == 'assemblyai':
            try:
                from .assemblyai_transcription_service import AssemblyAITranscriptionService
            except ImportError:
                from assemblyai_transcription_service import AssemblyAITranscriptionService
            return AssemblyAITranscriptionService(config)
        else:
            raise ValueError(f"Unknown transcription provider: {provider}")
    
    @staticmethod
    def get_available_providers() -> list:
        """Get list of available transcription providers"""
        return ['whisper', 'assemblyai']


def get_transcription_service(provider: str = None, config: Dict[str, Any] = None) -> TranscriptionService:
    """
    Convenience function to get a transcription service
    
    Args:
        provider: 'whisper' or 'assemblyai'. If None, will try to determine from config or environment
        config: Configuration dictionary
    
    Returns:
        TranscriptionService instance
    """
    if not provider:
        # Try to determine provider from config or environment
        if config and 'provider' in config:
            provider = config['provider']
        else:
            provider = os.getenv('TRANSCRIPTION_PROVIDER', 'whisper')
    
    service = TranscriptionServiceFactory.create_service(provider, config)
    
    if not service.initialize():
        raise RuntimeError(f"Failed to initialize {provider} transcription service")
    
    return service
