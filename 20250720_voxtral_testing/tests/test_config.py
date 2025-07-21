"""Tests for VoxtralConfig."""

import os
import pytest
from unittest.mock import patch

from voxtral.config import VoxtralConfig
from voxtral.exceptions import VoxtralConfigError


class TestVoxtralConfig:
    """Test cases for VoxtralConfig."""
    
    def test_default_config(self) -> None:
        """Test default configuration values."""
        config = VoxtralConfig()
        
        assert config.server_host == "localhost"
        assert config.server_port == 8000
        assert config.api_key == "EMPTY"
        assert config.model_id == "mistralai/Voxtral-Mini-3B-2507"
        assert config.tokenizer_mode == "mistral"
        assert config.config_format == "mistral"
        assert config.load_format == "mistral"
        assert config.request_timeout == 300.0
        assert config.connection_timeout == 30.0
        assert config.default_temperature == 0.2
        assert config.default_top_p == 0.95
        assert config.default_max_tokens == 500
        assert config.max_audio_duration == 1800
        assert "mp3" in config.supported_formats
        assert "wav" in config.supported_formats
    
    def test_custom_config(self) -> None:
        """Test custom configuration values."""
        config = VoxtralConfig(
            server_host="example.com",
            server_port=9000,
            api_key="test-key",
            request_timeout=600.0,
        )
        
        assert config.server_host == "example.com"
        assert config.server_port == 9000
        assert config.api_key == "test-key"
        assert config.request_timeout == 600.0
    
    def test_base_url_property(self) -> None:
        """Test base_url property."""
        config = VoxtralConfig(server_host="example.com", server_port=9000)
        assert config.base_url == "http://example.com:9000/v1"
    
    def test_server_url_property(self) -> None:
        """Test server_url property."""
        config = VoxtralConfig(server_host="example.com", server_port=9000)
        assert config.server_url == "http://example.com:9000"
    
    def test_from_env(self) -> None:
        """Test configuration from environment variables."""
        env_vars = {
            "VOXTRAL_SERVER_HOST": "env-host",
            "VOXTRAL_SERVER_PORT": "9999",
            "VOXTRAL_API_KEY": "env-key",
            "VOXTRAL_MODEL_ID": "custom/model",
            "VOXTRAL_REQUEST_TIMEOUT": "120.0",
            "VOXTRAL_CONNECTION_TIMEOUT": "15.0",
        }
        
        with patch.dict(os.environ, env_vars):
            config = VoxtralConfig.from_env()
            
            assert config.server_host == "env-host"
            assert config.server_port == 9999
            assert config.api_key == "env-key"
            assert config.model_id == "custom/model"
            assert config.request_timeout == 120.0
            assert config.connection_timeout == 15.0
    
    def test_from_env_defaults(self) -> None:
        """Test configuration from environment with defaults."""
        # Clear any existing env vars
        env_vars = {
            "VOXTRAL_SERVER_HOST": "",
            "VOXTRAL_SERVER_PORT": "",
            "VOXTRAL_API_KEY": "",
            "VOXTRAL_MODEL_ID": "",
            "VOXTRAL_REQUEST_TIMEOUT": "",
            "VOXTRAL_CONNECTION_TIMEOUT": "",
        }
        
        with patch.dict(os.environ, env_vars, clear=True):
            config = VoxtralConfig.from_env()
            
            assert config.server_host == "localhost"
            assert config.server_port == 8000
            assert config.api_key == "EMPTY"
            assert config.model_id == "mistralai/Voxtral-Mini-3B-2507"
            assert config.request_timeout == 300.0
            assert config.connection_timeout == 30.0
    
    def test_to_server_args(self) -> None:
        """Test conversion to server command line arguments."""
        config = VoxtralConfig(
            server_host="example.com",
            server_port=9000,
            model_id="custom/model",
        )
        
        args = config.to_server_args()
        
        assert "custom/model" in args
        assert "--tokenizer_mode=mistral" in args
        assert "--config_format=mistral" in args
        assert "--load_format=mistral" in args
        assert "--port=9000" in args
        assert "--host=example.com" in args
    
    def test_validation_errors(self) -> None:
        """Test configuration validation errors."""
        # Test invalid port
        with pytest.raises(ValueError):
            VoxtralConfig(server_port=0)
        
        with pytest.raises(ValueError):
            VoxtralConfig(server_port=70000)
        
        # Test invalid timeout
        with pytest.raises(ValueError):
            VoxtralConfig(request_timeout=-1.0)
        
        # Test invalid temperature
        with pytest.raises(ValueError):
            VoxtralConfig(default_temperature=-0.1)
        
        with pytest.raises(ValueError):
            VoxtralConfig(default_temperature=2.1)
        
        # Test invalid top_p
        with pytest.raises(ValueError):
            VoxtralConfig(default_top_p=-0.1)
        
        with pytest.raises(ValueError):
            VoxtralConfig(default_top_p=1.1)
        
        # Test empty server host
        with pytest.raises(ValueError):
            VoxtralConfig(server_host="")
        
        with pytest.raises(ValueError):
            VoxtralConfig(server_host="   ")
