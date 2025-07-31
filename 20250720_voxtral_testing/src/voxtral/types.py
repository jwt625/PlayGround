"""Type definitions for Voxtral package."""

from typing import Any, Dict, List, Literal, Optional, Union
from pathlib import Path
from pydantic import BaseModel, Field, validator


AudioLanguage = Literal[
    "en", "es", "fr", "pt", "hi", "de", "nl", "it"
]

AudioFormat = Literal["mp3", "wav", "flac", "m4a", "ogg"]


class AudioInput(BaseModel):
    """Represents an audio input for processing."""
    
    path: Union[str, Path]
    format: Optional[AudioFormat] = None
    language: Optional[AudioLanguage] = None
    
    @validator("path")
    def validate_path(cls, v: Union[str, Path]) -> Union[str, Path]:
        """Validate that the audio path exists or is a valid URL."""
        path_str = str(v)

        # If it's a URL, return as string (don't convert to Path)
        if path_str.startswith(("http://", "https://")):
            return path_str

        # If it's a local path, check if it exists
        path = Path(v)
        if not path.exists():
            raise ValueError(f"Audio file not found: {path}")
        return path
    
    @validator("format", pre=True, always=True)
    def infer_format(cls, v: Optional[str], values: Dict[str, Any]) -> Optional[str]:
        """Infer format from file extension if not provided."""
        if v is not None:
            return v
        
        path = values.get("path")
        if path:
            suffix = Path(path).suffix.lower().lstrip(".")
            if suffix in ["mp3", "wav", "flac", "m4a", "ogg"]:
                return suffix
        return None


class TranscriptionRequest(BaseModel):
    """Request for audio transcription."""
    
    audio: AudioInput
    language: Optional[AudioLanguage] = None
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    
    class Config:
        arbitrary_types_allowed = True


class AudioUnderstandingRequest(BaseModel):
    """Request for audio understanding/Q&A."""
    
    audio_files: List[AudioInput]
    question: str
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    top_p: float = Field(default=0.95, ge=0.0, le=1.0)
    max_tokens: int = Field(default=500, gt=0)
    
    class Config:
        arbitrary_types_allowed = True
    
    @validator("audio_files")
    def validate_audio_files(cls, v: List[AudioInput]) -> List[AudioInput]:
        """Validate that at least one audio file is provided."""
        if not v:
            raise ValueError("At least one audio file must be provided")
        return v


class VoxtralResponse(BaseModel):
    """Base response from Voxtral."""
    
    content: str
    model: str
    usage: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None


class TranscriptionResponse(VoxtralResponse):
    """Response from transcription request."""
    
    language: Optional[str] = None
    confidence: Optional[float] = None


class AudioUnderstandingResponse(VoxtralResponse):
    """Response from audio understanding request."""
    
    audio_count: int
    processing_time: Optional[float] = None
