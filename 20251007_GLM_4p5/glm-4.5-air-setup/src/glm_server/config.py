"""Configuration module for GLM-4.5-Air server."""

from pathlib import Path

from pydantic import BaseModel, Field, validator


class InferenceConfig(BaseModel):
    """Configuration for GLM-4.5-Air inference server."""

    model_path: str = Field(
        default="zai-org/GLM-4.5-Air",
        description="Path or HuggingFace model identifier for GLM-4.5-Air",
    )
    tensor_parallel_size: int = Field(
        default=2,
        ge=1,
        le=8,
        description="Number of GPUs to use for tensor parallelism",
    )
    host: str = Field(default="0.0.0.0", description="Host address to bind the server")
    port: int = Field(
        default=8000, ge=1024, le=65535, description="Port number for the server"
    )
    max_model_len: int | None = Field(
        default=None, description="Maximum sequence length for the model"
    )
    gpu_memory_utilization: float = Field(
        default=0.9, ge=0.1, le=1.0, description="GPU memory utilization ratio"
    )
    enable_thinking_mode: bool = Field(
        default=True, description="Enable GLM-4.5 thinking mode for complex reasoning"
    )
    tool_calling_enabled: bool = Field(
        default=True, description="Enable tool calling capabilities"
    )
    max_num_seqs: int = Field(
        default=256,
        ge=1,
        description="Maximum number of sequences to process in parallel",
    )

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
