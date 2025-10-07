"""Tests for configuration module."""

import pytest

from glm_server.config import InferenceConfig, get_config


def test_inference_config_defaults() -> None:
    """Test default configuration values."""
    config = InferenceConfig()

    assert config.model_path == "zai-org/GLM-4.5-Air"
    assert config.tensor_parallel_size == 2
    assert config.host == "0.0.0.0"
    assert config.port == 8000
    assert config.max_model_len is None
    assert config.gpu_memory_utilization == 0.9
    assert config.enable_thinking_mode is True
    assert config.tool_calling_enabled is True
    assert config.max_num_seqs == 256


def test_inference_config_validation() -> None:
    """Test configuration validation."""
    # Test valid configuration
    config = InferenceConfig(
        model_path="zai-org/GLM-4.5-Air",
        tensor_parallel_size=1,
        port=9000,
        gpu_memory_utilization=0.8
    )
    assert config.tensor_parallel_size == 1
    assert config.port == 9000
    assert config.gpu_memory_utilization == 0.8


def test_inference_config_invalid_values() -> None:
    """Test configuration validation with invalid values."""
    # Test invalid port
    with pytest.raises(ValueError):
        InferenceConfig(port=100)  # Too low

    with pytest.raises(ValueError):
        InferenceConfig(port=70000)  # Too high

    # Test invalid GPU memory utilization
    with pytest.raises(ValueError):
        InferenceConfig(gpu_memory_utilization=0.0)  # Too low

    with pytest.raises(ValueError):
        InferenceConfig(gpu_memory_utilization=1.5)  # Too high

    # Test invalid tensor parallel size
    with pytest.raises(ValueError):
        InferenceConfig(tensor_parallel_size=0)  # Too low


def test_get_config() -> None:
    """Test get_config function."""
    config = get_config()
    assert isinstance(config, InferenceConfig)
    assert config.model_path == "zai-org/GLM-4.5-Air"


def test_config_serialization() -> None:
    """Test configuration serialization."""
    config = InferenceConfig()
    config_dict = config.model_dump()

    assert isinstance(config_dict, dict)
    assert "model_path" in config_dict
    assert "tensor_parallel_size" in config_dict
    assert config_dict["model_path"] == "zai-org/GLM-4.5-Air"
