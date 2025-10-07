"""Configuration module for GLM-4.5-Air server."""

import secrets
from pathlib import Path

from pydantic import BaseModel, Field, validator


class InferenceConfig(BaseModel):
    """Configuration for GLM-4.5-Air inference server with H100 optimizations."""

    # Model configuration
    model_path: str = Field(
        default="models/GLM-4.5-Air-FP8",
        description="Path or HuggingFace model identifier for GLM-4.5-Air",
    )

    # Server configuration
    host: str = Field(default="0.0.0.0", description="Host address to bind the server")
    port: int = Field(
        default=8000, ge=1024, le=65535, description="Port number for the server"
    )

    # Authentication
    api_key: str | None = Field(
        default=None,
        description="API key for authentication. If None, will be auto-generated"
    )
    require_auth: bool = Field(
        default=True,
        description="Whether to require API key authentication"
    )

    # GPU and parallelism configuration (optimized for 2x H100)
    tensor_parallel_size: int = Field(
        default=2,
        ge=1,
        le=8,
        description="Number of GPUs to use for tensor parallelism",
    )
    gpu_memory_utilization: float = Field(
        default=0.95, ge=0.1, le=1.0,
        description="GPU memory utilization ratio (higher for H100)"
    )

    # Model length and batching (optimized for H100)
    max_model_len: int | None = Field(
        default=32768,
        description="Maximum sequence length (32K optimal for H100)"
    )
    max_num_seqs: int = Field(
        default=512,
        ge=1,
        description="Maximum number of sequences to process in parallel",
    )
    max_num_batched_tokens: int = Field(
        default=8192,
        description="Maximum number of batched tokens for chunked prefill"
    )

    # Performance optimizations for H100
    enable_prefix_caching: bool = Field(
        default=True,
        description="Enable prefix caching for better efficiency"
    )
    enable_chunked_prefill: bool = Field(
        default=True,
        description="Enable chunked prefill for better memory usage"
    )
    block_size: int = Field(
        default=32,
        description="Block size for attention (32 optimal for H100)"
    )
    swap_space: int = Field(
        default=8,
        description="Swap space in GB for CPU offloading"
    )

    # Model-specific settings
    trust_remote_code: bool = Field(
        default=True,
        description="Trust remote code for GLM models"
    )
    dtype: str = Field(
        default="auto",
        description="Data type (auto lets vLLM choose best)"
    )
    quantization: str | None = Field(
        default="compressed-tensors",
        description="Quantization method for FP8 models"
    )

    # GLM-specific features
    enable_thinking_mode: bool = Field(
        default=True, description="Enable GLM-4.5 thinking mode for complex reasoning"
    )
    tool_calling_enabled: bool = Field(
        default=True, description="Enable tool calling capabilities"
    )

    # Logging and monitoring
    disable_log_stats: bool = Field(
        default=False,
        description="Disable logging of statistics"
    )
    log_level: str = Field(
        default="INFO",
        description="Logging level"
    )

    @validator("api_key", pre=True, always=True)
    def validate_api_key(cls, v: str | None) -> str:
        """Generate API key if not provided."""
        if v is None:
            # Generate a secure random API key
            return f"glm-{secrets.token_urlsafe(32)}"
        return v

    @validator("model_path")
    def validate_model_path(cls, v: str) -> str:
        """Validate model path exists if it's a local path."""
        if not v.startswith(("zai-org/", "huggingface.co/")):
            # Local path validation
            path = Path(v)
            if not path.exists():
                raise ValueError(f"Local model path does not exist: {v}")
        return v

    @validator("tensor_parallel_size")
    def validate_tensor_parallel_size(cls, v: int) -> int:
        """Validate tensor parallel size against available GPUs."""
        try:
            import torch

            available_gpus = torch.cuda.device_count()
            if v > available_gpus:
                raise ValueError(
                    f"Tensor parallel size ({v}) exceeds available GPUs ({available_gpus})"
                )
        except ImportError:
            # If torch is not available, skip validation
            pass
        return v


def get_config() -> InferenceConfig:
    """Get the global configuration instance."""
    return InferenceConfig()


def get_optimized_h100_config(model_path: str = "models/GLM-4.5-Air-FP8") -> InferenceConfig:
    """Get optimized configuration for 2x H100 setup."""
    return InferenceConfig(
        model_path=model_path,
        tensor_parallel_size=2,
        gpu_memory_utilization=0.95,
        max_model_len=32768,
        max_num_seqs=512,
        max_num_batched_tokens=8192,
        enable_prefix_caching=True,
        enable_chunked_prefill=True,
        block_size=32,
        swap_space=8,
        trust_remote_code=True,
        dtype="auto",
        quantization="compressed-tensors",
        require_auth=True,
    )
