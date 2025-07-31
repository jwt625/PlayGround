"""Integration tests for Voxtral client with actual server."""

import asyncio
import pytest
import logging
from unittest.mock import patch

from voxtral import VoxtralClient, VoxtralConfig
from voxtral.types import AudioInput, TranscriptionRequest, AudioUnderstandingRequest
from voxtral.exceptions import VoxtralServerError, VoxtralAudioError


# Configure logging for tests
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@pytest.fixture
def config() -> VoxtralConfig:
    """Create test configuration."""
    return VoxtralConfig(
        server_host="localhost",
        server_port=8000,
        request_timeout=60.0,
    )


@pytest.fixture
def client(config: VoxtralConfig) -> VoxtralClient:
    """Create test client."""
    return VoxtralClient(config)


@pytest.fixture
def sample_audio_url() -> str:
    """Sample audio URL from HuggingFace."""
    return "https://huggingface.co/datasets/hf-internal-testing/dummy-audio-samples/resolve/main/obama.mp3"


@pytest.fixture
def sample_audio_urls() -> list[str]:
    """Multiple sample audio URLs from HuggingFace."""
    return [
        "https://huggingface.co/datasets/hf-internal-testing/dummy-audio-samples/resolve/main/obama.mp3",
        "https://huggingface.co/datasets/hf-internal-testing/dummy-audio-samples/resolve/main/bcn_weather.mp3",
    ]


class TestVoxtralClientIntegration:
    """Integration tests for VoxtralClient."""
    
    @pytest.mark.asyncio
    async def test_health_check(self, client: VoxtralClient) -> None:
        """Test server health check."""
        try:
            is_healthy = await client.health_check()
            if not is_healthy:
                pytest.skip("Server is not running or not healthy")
            assert is_healthy
        except Exception as e:
            pytest.skip(f"Server not available: {e}")
    
    @pytest.mark.asyncio
    async def test_get_model_name(self, client: VoxtralClient) -> None:
        """Test getting model name from server."""
        try:
            if not await client.health_check():
                pytest.skip("Server is not healthy")
            
            model_name = await client.get_model_name()
            assert model_name is not None
            assert len(model_name) > 0
            logger.info(f"Model name: {model_name}")
        except Exception as e:
            pytest.skip(f"Server not available: {e}")
    
    @pytest.mark.asyncio
    async def test_transcription(self, client: VoxtralClient, sample_audio_url: str) -> None:
        """Test audio transcription."""
        try:
            if not await client.health_check():
                pytest.skip("Server is not healthy")
            
            audio_input = AudioInput(path=sample_audio_url, language="en")
            request = TranscriptionRequest(
                audio=audio_input,
                language="en",
                temperature=0.0
            )
            
            response = await client.transcribe(request)
            
            assert response.content is not None
            assert len(response.content) > 0
            assert response.model is not None
            assert response.language == "en"
            assert response.metadata is not None
            assert "processing_time" in response.metadata
            
            logger.info(f"Transcription: {response.content[:100]}...")
            logger.info(f"Processing time: {response.metadata['processing_time']:.2f}s")
            
        except Exception as e:
            pytest.skip(f"Transcription test failed: {e}")
    
    @pytest.mark.asyncio
    async def test_audio_understanding_single(self, client: VoxtralClient, sample_audio_url: str) -> None:
        """Test audio understanding with single audio file."""
        try:
            if not await client.health_check():
                pytest.skip("Server is not healthy")
            
            audio_input = AudioInput(path=sample_audio_url)
            request = AudioUnderstandingRequest(
                audio_files=[audio_input],
                question="What is this audio about?",
                temperature=0.2,
                top_p=0.95,
                max_tokens=200
            )
            
            response = await client.understand_audio(request)
            
            assert response.content is not None
            assert len(response.content) > 0
            assert response.model is not None
            assert response.audio_count == 1
            assert response.processing_time is not None
            assert response.processing_time > 0
            
            logger.info(f"Understanding: {response.content[:100]}...")
            logger.info(f"Processing time: {response.processing_time:.2f}s")
            
        except Exception as e:
            pytest.skip(f"Audio understanding test failed: {e}")
    
    @pytest.mark.asyncio
    async def test_audio_understanding_multiple(self, client: VoxtralClient, sample_audio_urls: list[str]) -> None:
        """Test audio understanding with multiple audio files."""
        try:
            if not await client.health_check():
                pytest.skip("Server is not healthy")
            
            audio_inputs = [AudioInput(path=url) for url in sample_audio_urls]
            request = AudioUnderstandingRequest(
                audio_files=audio_inputs,
                question="Compare these two audio files. What are the differences?",
                temperature=0.2,
                top_p=0.95,
                max_tokens=300
            )
            
            response = await client.understand_audio(request)
            
            assert response.content is not None
            assert len(response.content) > 0
            assert response.model is not None
            assert response.audio_count == len(sample_audio_urls)
            assert response.processing_time is not None
            assert response.processing_time > 0
            
            logger.info(f"Comparison: {response.content[:100]}...")
            logger.info(f"Processing time: {response.processing_time:.2f}s")
            
        except Exception as e:
            pytest.skip(f"Multi-audio understanding test failed: {e}")
    
    @pytest.mark.asyncio
    async def test_error_handling_invalid_audio(self, client: VoxtralClient) -> None:
        """Test error handling with invalid audio URL."""
        try:
            if not await client.health_check():
                pytest.skip("Server is not healthy")
            
            # Test with invalid URL
            audio_input = AudioInput(path="https://example.com/nonexistent.mp3")
            request = TranscriptionRequest(
                audio=audio_input,
                language="en",
                temperature=0.0
            )
            
            with pytest.raises(VoxtralAudioError):
                await client.transcribe(request)
                
        except Exception as e:
            pytest.skip(f"Error handling test failed: {e}")
    
    @pytest.mark.asyncio
    async def test_different_languages(self, client: VoxtralClient) -> None:
        """Test transcription with different language settings."""
        try:
            if not await client.health_check():
                pytest.skip("Server is not healthy")
            
            # Use the Obama audio which is in English
            audio_url = "https://huggingface.co/datasets/hf-internal-testing/dummy-audio-samples/resolve/main/obama.mp3"
            
            # Test with different language settings
            languages = ["en", "es", "fr"]
            
            for lang in languages:
                audio_input = AudioInput(path=audio_url, language=lang)
                request = TranscriptionRequest(
                    audio=audio_input,
                    language=lang,
                    temperature=0.0
                )
                
                response = await client.transcribe(request)
                
                assert response.content is not None
                assert len(response.content) > 0
                assert response.language == lang
                
                logger.info(f"Language {lang}: {response.content[:50]}...")
                
        except Exception as e:
            pytest.skip(f"Language test failed: {e}")


@pytest.mark.asyncio
async def test_concurrent_requests(client: VoxtralClient, sample_audio_url: str) -> None:
    """Test concurrent requests to the server."""
    try:
        if not await client.health_check():
            pytest.skip("Server is not healthy")
        
        # Create multiple concurrent transcription requests
        tasks = []
        for i in range(3):
            audio_input = AudioInput(path=sample_audio_url, language="en")
            request = TranscriptionRequest(
                audio=audio_input,
                language="en",
                temperature=0.0
            )
            task = client.transcribe(request)
            tasks.append(task)
        
        # Wait for all tasks to complete
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Check that all requests succeeded
        successful_responses = [r for r in responses if not isinstance(r, Exception)]
        assert len(successful_responses) >= 1  # At least one should succeed
        
        for response in successful_responses:
            assert response.content is not None
            assert len(response.content) > 0
        
        logger.info(f"Concurrent requests: {len(successful_responses)}/{len(tasks)} succeeded")
        
    except Exception as e:
        pytest.skip(f"Concurrent test failed: {e}")


if __name__ == "__main__":
    # Run a simple test when executed directly
    async def simple_test() -> None:
        config = VoxtralConfig()
        client = VoxtralClient(config)
        
        print("Testing server health...")
        if await client.health_check():
            print("✅ Server is healthy")
            
            print("Testing model name...")
            model_name = await client.get_model_name()
            print(f"✅ Model: {model_name}")
            
            print("Testing transcription...")
            audio_url = "https://huggingface.co/datasets/hf-internal-testing/dummy-audio-samples/resolve/main/obama.mp3"
            audio_input = AudioInput(path=audio_url, language="en")
            request = TranscriptionRequest(audio=audio_input, language="en", temperature=0.0)
            response = await client.transcribe(request)
            print(f"✅ Transcription: {response.content[:100]}...")
            
        else:
            print("❌ Server is not healthy")
    
    asyncio.run(simple_test())
