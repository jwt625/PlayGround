"""FastAPI application configuration."""

import os
from typing import List
from pydantic import BaseModel, Field


class AppConfig(BaseModel):
    """Configuration for the FastAPI application."""
    
    # Application settings
    app_name: str = Field(default="Voxtral Transcription API")
    app_version: str = Field(default="0.1.0")
    debug: bool = Field(default=False)
    
    # Server settings
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8080, gt=0, le=65535)
    
    # CORS settings
    allowed_origins: List[str] = Field(default_factory=lambda: ["*"])
    allowed_methods: List[str] = Field(default_factory=lambda: ["*"])
    allowed_headers: List[str] = Field(default_factory=lambda: ["*"])
    
    # Rate limiting
    rate_limit_requests: int = Field(default=100)  # requests per minute
    rate_limit_window: int = Field(default=60)     # window in seconds
    
    # Audio processing limits
    max_file_size_mb: int = Field(default=100)
    max_audio_duration_seconds: int = Field(default=1800)  # 30 minutes
    supported_formats: List[str] = Field(
        default_factory=lambda: ["mp3", "wav", "flac", "m4a", "ogg"]
    )
    
    # Voxtral backend settings
    voxtral_host: str = Field(default="localhost")
    voxtral_port: int = Field(default=8000)
    voxtral_timeout: float = Field(default=300.0)
    
    @classmethod
    def from_env(cls) -> "AppConfig":
        """Create configuration from environment variables."""
        return cls(
            app_name=os.getenv("APP_NAME", "Voxtral Transcription API"),
            app_version=os.getenv("APP_VERSION", "0.1.0"),
            debug=os.getenv("DEBUG", "false").lower() == "true",
            host=os.getenv("HOST", "0.0.0.0"),
            port=int(os.getenv("PORT", "8080")),
            rate_limit_requests=int(os.getenv("RATE_LIMIT_REQUESTS", "100")),
            rate_limit_window=int(os.getenv("RATE_LIMIT_WINDOW", "60")),
            max_file_size_mb=int(os.getenv("MAX_FILE_SIZE_MB", "100")),
            max_audio_duration_seconds=int(os.getenv("MAX_AUDIO_DURATION_SECONDS", "1800")),
            voxtral_host=os.getenv("VOXTRAL_HOST", "localhost"),
            voxtral_port=int(os.getenv("VOXTRAL_PORT", "8000")),
            voxtral_timeout=float(os.getenv("VOXTRAL_TIMEOUT", "300.0")),
        )


# Global configuration instance
config = AppConfig.from_env()
