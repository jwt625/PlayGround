"""Configuration management for Voxtral."""

import os
from typing import Optional
from pydantic import BaseModel, Field, validator


class VoxtralConfig(BaseModel):
    """Configuration for Voxtral client."""
    
    # Server configuration
    server_host: str = Field(default="localhost")
    server_port: int = Field(default=8000, gt=0, le=65535)
    api_key: str = Field(default="EMPTY")
    
    # Model configuration
    model_id: str = Field(default="mistralai/Voxtral-Mini-3B-2507")
    tokenizer_mode: str = Field(default="mistral")
    config_format: str = Field(default="mistral")
    load_format: str = Field(default="mistral")
    
    # Request timeouts
    request_timeout: float = Field(default=300.0, gt=0)
    connection_timeout: float = Field(default=30.0, gt=0)
    
    # Default generation parameters
    default_temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    default_top_p: float = Field(default=0.95, ge=0.0, le=1.0)
    default_max_tokens: int = Field(default=500, gt=0)
    
    # Audio processing
    max_audio_duration: int = Field(default=1800, gt=0)  # 30 minutes in seconds
    supported_formats: list[str] = Field(
        default_factory=lambda: ["mp3", "wav", "flac", "m4a", "ogg"]
    )
    
    @validator("server_host")
    def validate_server_host(cls, v: str) -> str:
        """Validate server host format."""
        if not v or v.isspace():
            raise ValueError("Server host cannot be empty")
        return v.strip()
    
    @property
    def base_url(self) -> str:
        """Get the base URL for the API server."""
        return f"http://{self.server_host}:{self.server_port}/v1"
    
    @property
    def server_url(self) -> str:
        """Get the server URL without API path."""
        return f"http://{self.server_host}:{self.server_port}"
    
    @classmethod
    def from_env(cls) -> "VoxtralConfig":
        """Create configuration from environment variables."""
        return cls(
            server_host=os.getenv("VOXTRAL_SERVER_HOST", "localhost"),
            server_port=int(os.getenv("VOXTRAL_SERVER_PORT", "8000")),
            api_key=os.getenv("VOXTRAL_API_KEY", "EMPTY"),
            model_id=os.getenv("VOXTRAL_MODEL_ID", "mistralai/Voxtral-Mini-3B-2507"),
            request_timeout=float(os.getenv("VOXTRAL_REQUEST_TIMEOUT", "300.0")),
            connection_timeout=float(os.getenv("VOXTRAL_CONNECTION_TIMEOUT", "30.0")),
        )
    
    def to_server_args(self) -> list[str]:
        """Convert config to vLLM server command line arguments."""
        return [
            self.model_id,
            f"--tokenizer_mode={self.tokenizer_mode}",
            f"--config_format={self.config_format}",
            f"--load_format={self.load_format}",
            f"--port={self.server_port}",
            f"--host={self.server_host}",
        ]
