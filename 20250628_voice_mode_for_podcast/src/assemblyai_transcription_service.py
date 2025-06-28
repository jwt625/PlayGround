"""
AssemblyAI Transcription Service Implementation

This module provides AssemblyAI-based transcription with both file and streaming support.
"""

import os
import time
import tempfile
import threading
import wave
import struct
from typing import Optional, Dict, Any, Callable
import logging

import assemblyai as aai

try:
    from .transcription_service import TranscriptionService, TranscriptionResult
except ImportError:
    from transcription_service import TranscriptionService, TranscriptionResult


class AssemblyAITranscriptionService(TranscriptionService):
    """AssemblyAI implementation of transcription service"""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.api_key = None
        self.transcriber = None
        self.realtime_transcriber = None
        self.streaming_callback = None
        self.is_streaming = False

        # Default configuration
        self.speech_model = self.config.get('speech_model', aai.SpeechModel.best)
        self.sample_rate = self.config.get('sample_rate', 16000)

        # Setup logging
        self.logger = logging.getLogger(__name__)
    
    def initialize(self) -> bool:
        """Initialize the AssemblyAI service"""
        try:
            # Get API key from config, environment, or .env file
            self.api_key = self.config.get('api_key')
            if not self.api_key:
                self.api_key = os.getenv('API_KEY')
            if not self.api_key:
                # Try to load from .env file
                try:
                    from dotenv import load_dotenv
                    load_dotenv()
                    self.api_key = os.getenv('API_KEY')
                except ImportError:
                    pass
            
            if not self.api_key:
                self.logger.error("No AssemblyAI API key found. Set API_KEY environment variable or pass in config.")
                return False
            
            # Set the API key
            aai.settings.api_key = self.api_key
            
            # Create transcriber for file-based transcription
            config = aai.TranscriptionConfig(speech_model=self.speech_model)
            self.transcriber = aai.Transcriber(config=config)
            
            self.is_initialized = True
            self.logger.info("✅ AssemblyAI transcription service initialized")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize AssemblyAI service: {e}")
            return False
    
    def transcribe_audio_data(self, audio_data: bytes, source: str = 'microphone') -> Optional[TranscriptionResult]:
        """
        For chunk-based transcription, we'll use streaming mode instead of file-based.
        This method will return None and rely on the streaming callback for results.
        """
        if not self.is_initialized:
            return None

        # For AssemblyAI, we prefer streaming over chunk-based file transcription
        # The actual transcription happens in the streaming mode
        self.logger.debug(f"Received audio chunk of {len(audio_data)} bytes from {source}")

        # Return None - transcription results come through streaming callback
        return None
    
    def transcribe_file(self, file_path: str) -> Optional[TranscriptionResult]:
        """Transcribe an audio file using AssemblyAI"""
        if not self.is_initialized:
            return None
        
        try:
            start_time = time.time()
            
            self.logger.info(f"Processing audio file with AssemblyAI: {file_path}")
            
            # Transcribe the file
            transcript = self.transcriber.transcribe(file_path)
            
            if transcript.status == "error":
                self.logger.error(f"AssemblyAI transcription failed: {transcript.error}")
                return None
            
            processing_time = time.time() - start_time
            self._update_stats(processing_time)
            
            # Extract text and confidence
            text = transcript.text or ""
            confidence = transcript.confidence or 0.0
            
            return TranscriptionResult(
                text=text.strip(),
                confidence=confidence,
                is_final=True,
                language=getattr(transcript, 'language_code', 'unknown') or "unknown",
                processing_time=processing_time,
                segments=[],
                metadata={
                    'provider': 'assemblyai',
                    'speech_model': str(self.speech_model),
                    'file_path': file_path
                }
            )
            
        except Exception as e:
            self.logger.error(f"Error processing audio file {file_path}: {e}")
            return None
    
    def supports_streaming(self) -> bool:
        """AssemblyAI supports real-time streaming (if account has paid features)"""
        # Note: Streaming requires a paid account with credit card
        # For free accounts, we'll fall back to file-based transcription
        return False  # Disable streaming for now due to paid requirement
    
    def start_streaming(self, callback: Callable[[TranscriptionResult], None],
                       sample_rate: int = 16000) -> bool:
        """Start streaming transcription with AssemblyAI"""
        if not self.is_initialized:
            return False

        if self.is_streaming:
            self.logger.warning("Streaming already active")
            return False

        try:
            self.streaming_callback = callback
            self.sample_rate = sample_rate

            # Create realtime transcriber with event handlers
            self.realtime_transcriber = aai.RealtimeTranscriber(
                sample_rate=sample_rate,
                on_data=self._on_data,
                on_error=self._on_error,
                on_open=self._on_open,
                on_close=self._on_close,
                word_boost=['um', 'uh']  # Optional word boost
            )

            # Connect to streaming service
            self.realtime_transcriber.connect()

            # Start streaming from microphone in background thread
            def stream_microphone():
                try:
                    self.logger.info("🎤 Starting AssemblyAI microphone stream...")
                    microphone_stream = aai.extras.MicrophoneStream(sample_rate=sample_rate)
                    self.realtime_transcriber.stream(microphone_stream)
                except Exception as e:
                    self.logger.error(f"Error in microphone streaming: {e}")
                    self.is_streaming = False

            # Start streaming in background thread
            self.streaming_thread = threading.Thread(target=stream_microphone, daemon=True)
            self.streaming_thread.start()

            self.is_streaming = True
            self.logger.info("✅ AssemblyAI streaming started")
            return True

        except Exception as e:
            self.logger.error(f"Failed to start AssemblyAI streaming: {e}")
            return False
    
    def stop_streaming(self):
        """Stop streaming transcription"""
        if self.realtime_transcriber and self.is_streaming:
            try:
                self.realtime_transcriber.close()
                self.is_streaming = False
                self.logger.info("🛑 AssemblyAI streaming stopped")
            except Exception as e:
                self.logger.error(f"Error stopping streaming: {e}")

    def cleanup(self):
        """Clean up resources"""
        self.stop_streaming()
        self.realtime_transcriber = None
        self.transcriber = None
    
    # Streaming event handlers
    def _on_open(self, session_opened: aai.RealtimeSessionOpened):
        """Handle streaming session start"""
        self.logger.info(f"🌊 AssemblyAI streaming session started: {session_opened.session_id}")

    def _on_data(self, transcript: aai.RealtimeTranscript):
        """Handle streaming transcription data"""
        self.logger.info(f"📝 Received transcript: '{transcript.text}' (type: {type(transcript).__name__})")

        if self.streaming_callback and transcript.text:
            result = TranscriptionResult(
                text=transcript.text,
                confidence=transcript.confidence if hasattr(transcript, 'confidence') else 1.0,
                is_final=isinstance(transcript, aai.RealtimeFinalTranscript),
                language="unknown",
                processing_time=0.0,
                segments=[],
                metadata={
                    'provider': 'assemblyai',
                    'streaming': True,
                    'is_final': isinstance(transcript, aai.RealtimeFinalTranscript),
                    'message_type': type(transcript).__name__
                }
            )
            self.logger.info(f"📤 Calling streaming callback with: '{result.text}'")
            self.streaming_callback(result)
        else:
            self.logger.warning(f"⚠️ No callback or empty text: callback={self.streaming_callback is not None}, text='{transcript.text}'")

    def _on_close(self):
        """Handle streaming session close"""
        self.logger.info("🛑 AssemblyAI streaming session closed")
        self.is_streaming = False

    def _on_error(self, error: aai.RealtimeError):
        """Handle streaming errors"""
        self.logger.error(f"❌ AssemblyAI streaming error: {error}")
        self.is_streaming = False

    def _audio_data_to_wav_file(self, audio_data: bytes) -> str:
        """Convert raw audio data to a proper WAV file"""
        try:
            # Create temporary WAV file
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                temp_file_path = temp_file.name

            # Audio parameters (matching the audio capture settings)
            sample_rate = 16000  # Sample rate from audio capture
            channels = 1  # Mono
            sample_width = 2  # 16-bit audio (2 bytes per sample)

            # Create WAV file with proper headers
            with wave.open(temp_file_path, 'wb') as wav_file:
                wav_file.setnchannels(channels)
                wav_file.setsampwidth(sample_width)
                wav_file.setframerate(sample_rate)
                wav_file.writeframes(audio_data)

            self.logger.debug(f"Created WAV file: {temp_file_path}, size: {len(audio_data)} bytes")
            return temp_file_path

        except Exception as e:
            self.logger.error(f"Error creating WAV file: {e}")
            return None
