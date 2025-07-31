"""Voxtral Mini 3B testing and development package."""

from .client import VoxtralClient
from .config import VoxtralConfig
from .exceptions import (
    VoxtralError,
    VoxtralServerError,
    VoxtralConfigError,
    VoxtralAudioError,
    VoxtralTimeoutError,
)
from .types import (
    AudioInput,
    TranscriptionRequest,
    AudioUnderstandingRequest,
    TranscriptionResponse,
    AudioUnderstandingResponse,
)

__version__ = "0.1.0"
__all__ = [
    "VoxtralClient",
    "VoxtralConfig",
    "VoxtralError",
    "VoxtralServerError",
    "VoxtralConfigError",
    "VoxtralAudioError",
    "VoxtralTimeoutError",
    "AudioInput",
    "TranscriptionRequest",
    "AudioUnderstandingRequest",
    "TranscriptionResponse",
    "AudioUnderstandingResponse",
]
