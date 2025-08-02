"""Service layer for interacting with Voxtral backend."""

import base64
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional

from voxtral import VoxtralClient, VoxtralConfig
from voxtral.types import AudioInput, TranscriptionRequest as VoxtralTranscriptionRequest, AudioUnderstandingRequest as VoxtralAudioUnderstandingRequest
from voxtral.exceptions import VoxtralError

from ..core.config import config
from ..core.logging import get_logger
from ..models.audio import (
    AudioFile,
    TranscriptionRequest,
    TranscriptionResponse,
    AudioUnderstandingRequest,
    AudioUnderstandingResponse,
    BatchTranscriptionRequest,
    BatchTranscriptionResponse,
    BatchTranscriptionResult,
)

logger = get_logger(__name__)


class VoxtralService:
    """Service for handling Voxtral model interactions."""
    
    def __init__(self) -> None:
        """Initialize the Voxtral service."""
        self.voxtral_config = VoxtralConfig(
            server_host=config.voxtral_host,
            server_port=config.voxtral_port,
            request_timeout=config.voxtral_timeout,
        )
        self.client = VoxtralClient(self.voxtral_config)
        self._model_name: Optional[str] = None
    
    async def get_model_name(self) -> str:
        """Get the model name from the Voxtral backend."""
        if self._model_name is None:
            self._model_name = await self.client.get_model_name()
        return self._model_name
    
    async def health_check(self) -> bool:
        """Check if the Voxtral backend is healthy."""
        try:
            return await self.client.health_check()
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False
    
    def _save_audio_to_temp_file(self, audio_data: str, format: str) -> Path:
        """Save base64 audio data to a temporary file."""
        try:
            # Decode base64 audio data
            audio_bytes = base64.b64decode(audio_data)
            
            # Create temporary file with appropriate extension
            suffix = f".{format}"
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
            temp_file.write(audio_bytes)
            temp_file.close()
            
            return Path(temp_file.name)
        except Exception as e:
            raise VoxtralError(f"Failed to save audio data to temporary file: {e}")
    
    async def transcribe_audio(self, request: TranscriptionRequest) -> TranscriptionResponse:
        """Transcribe audio to text."""
        start_time = time.time()
        temp_file = None
        
        try:
            # Save audio data to temporary file
            temp_file = self._save_audio_to_temp_file(request.audio_file, request.format)
            
            # Create Voxtral request
            audio_input = AudioInput(path=temp_file, language=request.language)
            voxtral_request = VoxtralTranscriptionRequest(
                audio=audio_input,
                language=request.language,
                temperature=request.temperature,
            )
            
            # Perform transcription
            response = await self.client.transcribe(voxtral_request)
            
            processing_time_ms = int((time.time() - start_time) * 1000)
            
            return TranscriptionResponse(
                transcription=response.content,
                language_detected=response.language,
                confidence=response.confidence,
                processing_time_ms=processing_time_ms,
                audio_duration_seconds=None,  # TODO: Extract from audio file
            )
        
        finally:
            # Clean up temporary file
            if temp_file and temp_file.exists():
                temp_file.unlink()
    
    async def understand_audio(self, request: AudioUnderstandingRequest) -> AudioUnderstandingResponse:
        """Understand and answer questions about audio content."""
        start_time = time.time()
        temp_files: List[Path] = []
        
        try:
            # Save all audio files to temporary files
            audio_inputs = []
            for audio_file in request.audio_files:
                temp_file = self._save_audio_to_temp_file(audio_file.data, audio_file.format)
                temp_files.append(temp_file)
                audio_inputs.append(AudioInput(path=temp_file))
            
            # Create Voxtral request
            voxtral_request = VoxtralAudioUnderstandingRequest(
                audio_files=audio_inputs,
                question=request.question,
                temperature=request.temperature,
                top_p=request.top_p,
                max_tokens=request.max_tokens,
            )
            
            # Perform audio understanding
            response = await self.client.understand_audio(voxtral_request)
            
            processing_time_ms = int((time.time() - start_time) * 1000)
            
            # Extract token usage
            token_usage = response.usage or {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            }
            
            return AudioUnderstandingResponse(
                answer=response.content,
                audio_count=len(request.audio_files),
                processing_time_ms=processing_time_ms,
                token_usage=token_usage,
            )
        
        finally:
            # Clean up temporary files
            for temp_file in temp_files:
                if temp_file.exists():
                    temp_file.unlink()
    
    async def batch_transcribe(self, request: BatchTranscriptionRequest) -> BatchTranscriptionResponse:
        """Transcribe multiple audio files."""
        start_time = time.time()
        results = []
        
        for i, audio_file in enumerate(request.audio_files):
            file_id = audio_file.id or f"file_{i}"
            
            try:
                # Create individual transcription request
                transcription_request = TranscriptionRequest(
                    audio_file=audio_file.data,
                    format=audio_file.format,
                    language=request.language,
                    temperature=request.temperature,
                )
                
                # Transcribe the audio
                response = await self.transcribe_audio(transcription_request)
                
                results.append(BatchTranscriptionResult(
                    id=file_id,
                    transcription=response.transcription,
                    success=True,
                ))
            
            except Exception as e:
                logger.error(f"Failed to transcribe audio file {file_id}: {e}")
                results.append(BatchTranscriptionResult(
                    id=file_id,
                    error=str(e),
                    success=False,
                ))
        
        total_processing_time_ms = int((time.time() - start_time) * 1000)
        
        return BatchTranscriptionResponse(
            results=results,
            total_processing_time_ms=total_processing_time_ms,
        )


# Global service instance
voxtral_service = VoxtralService()
