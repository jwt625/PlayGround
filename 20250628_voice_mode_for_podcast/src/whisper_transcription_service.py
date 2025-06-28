"""
Whisper Transcription Service Implementation

This module provides Whisper-based transcription using the existing transcript_processor logic
but adapted to the new transcription service interface.
"""

import os
import time
import tempfile
import numpy as np
from typing import Optional, Dict, Any, Callable
import logging

import whisper
from pydub import AudioSegment

try:
    from .transcription_service import TranscriptionService, TranscriptionResult
except ImportError:
    from transcription_service import TranscriptionService, TranscriptionResult


class WhisperTranscriptionService(TranscriptionService):
    """Whisper implementation of transcription service"""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.model = None
        self.model_name = self.config.get('model_name', 'small')
        self.language = self.config.get('language', 'en')
        
        # Deduplication tracking (from original implementation)
        self.recent_transcripts = []
        self.max_recent_transcripts = 5
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
    
    def initialize(self) -> bool:
        """Initialize the Whisper model"""
        try:
            self.logger.info(f"Loading Whisper {self.model_name} model...")
            start_time = time.time()
            self.model = whisper.load_model(self.model_name)
            load_time = time.time() - start_time
            self.logger.info(f"✅ Whisper {self.model_name} model loaded in {load_time:.2f}s")
            self.is_initialized = True
            return True
        except Exception as e:
            self.logger.error(f"❌ Failed to load Whisper model: {e}")
            return False
    
    def transcribe_audio_data(self, audio_data: bytes, source: str = 'microphone') -> Optional[TranscriptionResult]:
        """Transcribe audio data using Whisper"""
        if not self.is_initialized or not audio_data:
            return None
        
        try:
            # Convert audio data to temporary file
            temp_file = self._audio_data_to_temp_file(audio_data)
            if not temp_file:
                return None
            
            # Transcribe with Whisper
            start_time = time.time()
            result = self.model.transcribe(
                temp_file,
                language=self.language,
                task="transcribe",
                fp16=False,  # Use fp32 for better compatibility
                verbose=False
            )
            processing_time = time.time() - start_time
            
            # Clean up temp file
            try:
                os.unlink(temp_file)
            except:
                pass
            
            # Extract transcript text
            text = result.get('text', '').strip()
            
            if not text:
                self.logger.debug("⚠️  Empty transcript result")
                return None
            
            # Smart deduplication
            cleaned_text = self._deduplicate_text(text)
            if not cleaned_text:
                self.logger.debug("🔄 Duplicate content filtered out")
                return None
            
            text = cleaned_text
            
            # Update statistics
            self._update_stats(processing_time)
            
            # Calculate confidence (Whisper doesn't provide this directly)
            confidence = self._estimate_confidence(result)
            
            self.logger.info(f"🎤 Whisper result: '{text}' (confidence: {confidence:.2f})")
            
            return TranscriptionResult(
                text=text,
                confidence=confidence,
                is_final=True,
                language=result.get('language', 'unknown'),
                processing_time=processing_time,
                segments=result.get('segments', []),
                metadata={
                    'source': source,
                    'provider': 'whisper',
                    'model_name': self.model_name
                }
            )
        
        except Exception as e:
            self.logger.error(f"Error processing audio chunk: {e}")
            return None
    
    def transcribe_file(self, file_path: str) -> Optional[TranscriptionResult]:
        """Transcribe an audio file using Whisper"""
        if not self.is_initialized:
            return None
        
        try:
            self.logger.info(f"Processing audio file with Whisper: {file_path}")
            start_time = time.time()
            
            result = self.model.transcribe(
                file_path,
                language=self.language,
                task="transcribe",
                fp16=False,
                verbose=True
            )
            
            processing_time = time.time() - start_time
            self._update_stats(processing_time)
            
            return TranscriptionResult(
                text=result.get('text', '').strip(),
                confidence=self._estimate_confidence(result),
                is_final=True,
                language=result.get('language', 'unknown'),
                processing_time=processing_time,
                segments=result.get('segments', []),
                metadata={
                    'provider': 'whisper',
                    'model_name': self.model_name,
                    'file_path': file_path
                }
            )
        
        except Exception as e:
            self.logger.error(f"Error processing audio file {file_path}: {e}")
            return None
    
    def supports_streaming(self) -> bool:
        """Whisper does not support real-time streaming"""
        return False
    
    def cleanup(self):
        """Clean up resources"""
        self.model = None
    
    def _audio_data_to_temp_file(self, audio_data: bytes) -> Optional[str]:
        """Convert audio data to temporary WAV file (from original implementation)"""
        try:
            # Create temporary file
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                temp_file.write(audio_data)
                temp_file_path = temp_file.name
            
            # Verify the file was created and has content
            if os.path.exists(temp_file_path) and os.path.getsize(temp_file_path) > 0:
                return temp_file_path
            else:
                self.logger.warning("Created temp file is empty or doesn't exist")
                return None
                
        except Exception as e:
            self.logger.error(f"Error creating temp file: {e}")
            return None
    
    def _estimate_confidence(self, result: dict) -> float:
        """Estimate confidence from Whisper result (from original implementation)"""
        try:
            segments = result.get('segments', [])
            if not segments:
                return 0.5  # Default confidence
            
            # Calculate average confidence based on segment characteristics
            total_confidence = 0.0
            total_duration = 0.0
            
            for segment in segments:
                duration = segment.get('end', 0) - segment.get('start', 0)
                if duration > 0:
                    # Higher confidence for longer segments with more tokens
                    tokens = segment.get('tokens', [])
                    token_confidence = min(1.0, len(tokens) / 10.0)  # Normalize by expected tokens
                    
                    # Penalize very short segments
                    duration_confidence = min(1.0, duration / 0.5)  # Normalize by 0.5 seconds
                    
                    segment_confidence = (token_confidence + duration_confidence) / 2.0
                    total_confidence += segment_confidence * duration
                    total_duration += duration
            
            if total_duration > 0:
                return min(1.0, total_confidence / total_duration)
            else:
                return 0.5
                
        except Exception as e:
            self.logger.warning(f"Error estimating confidence: {e}")
            return 0.5
    
    def _deduplicate_text(self, text: str) -> Optional[str]:
        """Smart deduplication to avoid repeated transcripts (from original implementation)"""
        if not text or not text.strip():
            return None
        
        text = text.strip()
        
        # Check against recent transcripts
        for recent_text in self.recent_transcripts:
            # Exact match
            if text == recent_text:
                return None
            
            # Substring check (avoid partial repeats)
            if len(text) > 10 and (text in recent_text or recent_text in text):
                return None
            
            # Similar text check (simple word overlap)
            text_words = set(text.lower().split())
            recent_words = set(recent_text.lower().split())
            if len(text_words) > 2 and len(recent_words) > 2:
                overlap = len(text_words.intersection(recent_words))
                similarity = overlap / min(len(text_words), len(recent_words))
                if similarity > 0.8:  # 80% word overlap
                    return None
        
        # Add to recent transcripts
        self.recent_transcripts.append(text)
        if len(self.recent_transcripts) > self.max_recent_transcripts:
            self.recent_transcripts.pop(0)
        
        return text
