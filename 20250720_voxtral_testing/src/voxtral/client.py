"""Main client for interacting with Voxtral Mini 3B model."""

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional

from openai import OpenAI
from mistral_common.protocol.instruct.messages import (
    TextChunk, 
    AudioChunk, 
    UserMessage, 
    AssistantMessage,
    RawAudio
)
from mistral_common.protocol.transcription.request import TranscriptionRequest as MistralTranscriptionRequest
from mistral_common.audio import Audio
from huggingface_hub import hf_hub_download

from .config import VoxtralConfig
from .types import (
    AudioInput,
    TranscriptionRequest,
    AudioUnderstandingRequest,
    TranscriptionResponse,
    AudioUnderstandingResponse,
)
from .exceptions import (
    VoxtralError,
    VoxtralServerError,
    VoxtralAudioError,
    VoxtralTimeoutError,
)


logger = logging.getLogger(__name__)


class VoxtralClient:
    """Client for interacting with Voxtral Mini 3B model via vLLM server."""
    
    def __init__(self, config: Optional[VoxtralConfig] = None) -> None:
        """Initialize the Voxtral client.
        
        Args:
            config: Configuration object. If None, loads from environment.
        """
        self.config = config or VoxtralConfig.from_env()
        self._client: Optional[OpenAI] = None
        self._model_name: Optional[str] = None
        
    @property
    def client(self) -> OpenAI:
        """Get or create the OpenAI client."""
        if self._client is None:
            self._client = OpenAI(
                api_key=self.config.api_key,
                base_url=self.config.base_url,
                timeout=self.config.request_timeout,
            )
        return self._client
    
    async def get_model_name(self) -> str:
        """Get the model name from the server."""
        if self._model_name is None:
            try:
                models = self.client.models.list()
                if not models.data:
                    raise VoxtralServerError("No models available on server")
                self._model_name = models.data[0].id
                logger.info(f"Using model: {self._model_name}")
            except Exception as e:
                raise VoxtralServerError(f"Failed to get model name: {e}") from e
        return self._model_name
    
    def _audio_input_to_chunk(self, audio_input: AudioInput) -> AudioChunk:
        """Convert AudioInput to AudioChunk for mistral-common."""
        try:
            if str(audio_input.path).startswith(("http://", "https://")):
                # Handle remote URLs - download to temp file
                # For now, assume it's a huggingface hub file
                if "huggingface.co" in str(audio_input.path):
                    # Extract repo and filename from URL
                    # This is a simplified approach - in production, use proper URL parsing
                    parts = str(audio_input.path).split("/")
                    if "resolve" in parts:
                        repo_idx = parts.index("resolve") - 1
                        repo_id = "/".join(parts[repo_idx-1:repo_idx+1])
                        filename = parts[-1]
                        file_path = hf_hub_download(repo_id, filename, repo_type="dataset")
                    else:
                        raise VoxtralAudioError(f"Unsupported URL format: {audio_input.path}")
                else:
                    raise VoxtralAudioError(f"Unsupported remote URL: {audio_input.path}")
            else:
                file_path = str(audio_input.path)
            
            audio = Audio.from_file(file_path, strict=False)
            return AudioChunk.from_audio(audio)
        except Exception as e:
            raise VoxtralAudioError(f"Failed to process audio file {audio_input.path}: {e}") from e

    async def transcribe(self, request: TranscriptionRequest) -> TranscriptionResponse:
        """Transcribe audio to text.

        Args:
            request: Transcription request with audio and parameters.

        Returns:
            Transcription response with text and metadata.

        Raises:
            VoxtralServerError: If the server request fails.
            VoxtralAudioError: If audio processing fails.
        """
        try:
            model_name = await self.get_model_name()

            # Convert audio to RawAudio for transcription
            if str(request.audio.path).startswith(("http://", "https://")):
                if "huggingface.co" in str(request.audio.path):
                    parts = str(request.audio.path).split("/")
                    if "resolve" in parts:
                        repo_idx = parts.index("resolve") - 1
                        repo_id = "/".join(parts[repo_idx-1:repo_idx+1])
                        filename = parts[-1]
                        file_path = hf_hub_download(repo_id, filename, repo_type="dataset")
                    else:
                        raise VoxtralAudioError(f"Unsupported URL format: {request.audio.path}")
                else:
                    raise VoxtralAudioError(f"Unsupported remote URL: {request.audio.path}")
            else:
                file_path = str(request.audio.path)

            audio = Audio.from_file(file_path, strict=False)
            raw_audio = RawAudio.from_audio(audio)

            # Create transcription request
            transcription_req = MistralTranscriptionRequest(
                model=model_name,
                audio=raw_audio,
                language=request.language or "en",
                temperature=request.temperature,
            ).to_openai(exclude=("top_p", "seed"))

            start_time = time.time()
            response = self.client.audio.transcriptions.create(**transcription_req)
            processing_time = time.time() - start_time

            return TranscriptionResponse(
                content=response.text,
                model=model_name,
                language=request.language,
                metadata={"processing_time": processing_time}
            )

        except Exception as e:
            if isinstance(e, (VoxtralError, VoxtralAudioError)):
                raise
            raise VoxtralServerError(f"Transcription failed: {e}") from e

    async def understand_audio(self, request: AudioUnderstandingRequest) -> AudioUnderstandingResponse:
        """Understand and answer questions about audio content.

        Args:
            request: Audio understanding request with audio files and question.

        Returns:
            Audio understanding response with answer and metadata.

        Raises:
            VoxtralServerError: If the server request fails.
            VoxtralAudioError: If audio processing fails.
        """
        try:
            model_name = await self.get_model_name()

            # Convert audio files to chunks
            audio_chunks = []
            for audio_input in request.audio_files:
                chunk = self._audio_input_to_chunk(audio_input)
                audio_chunks.append(chunk)

            # Create text chunk for the question
            text_chunk = TextChunk(text=request.question)

            # Combine audio chunks and text into user message
            content = audio_chunks + [text_chunk]
            user_msg = UserMessage(content=content).to_openai()

            start_time = time.time()
            response = self.client.chat.completions.create(
                model=model_name,
                messages=[user_msg],
                temperature=request.temperature,
                top_p=request.top_p,
                max_tokens=request.max_tokens,
            )
            processing_time = time.time() - start_time

            content = response.choices[0].message.content or ""

            return AudioUnderstandingResponse(
                content=content,
                model=model_name,
                audio_count=len(request.audio_files),
                processing_time=processing_time,
                usage=response.usage.model_dump() if response.usage else None,
            )

        except Exception as e:
            if isinstance(e, (VoxtralError, VoxtralAudioError)):
                raise
            raise VoxtralServerError(f"Audio understanding failed: {e}") from e

    async def health_check(self) -> bool:
        """Check if the server is healthy and responsive.

        Returns:
            True if server is healthy, False otherwise.
        """
        try:
            models = self.client.models.list()
            return len(models.data) > 0
        except Exception as e:
            logger.warning(f"Health check failed: {e}")
            return False
