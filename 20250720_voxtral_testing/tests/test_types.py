"""Tests for Voxtral types."""

import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from voxtral.types import (
    AudioInput,
    TranscriptionRequest,
    AudioUnderstandingRequest,
    TranscriptionResponse,
    AudioUnderstandingResponse,
)


class TestAudioInput:
    """Test cases for AudioInput."""
    
    def test_valid_url(self) -> None:
        """Test AudioInput with valid URL."""
        audio = AudioInput(path="https://example.com/audio.mp3")
        assert str(audio.path) == "https://example.com/audio.mp3"
        assert audio.format == "mp3"
    
    def test_format_inference(self) -> None:
        """Test format inference from file extension."""
        test_cases = [
            ("audio.mp3", "mp3"),
            ("audio.wav", "wav"),
            ("audio.flac", "flac"),
            ("audio.m4a", "m4a"),
            ("audio.ogg", "ogg"),
            ("AUDIO.MP3", "mp3"),  # Case insensitive
        ]
        
        for filename, expected_format in test_cases:
            with patch("pathlib.Path.exists", return_value=True):
                audio = AudioInput(path=filename)
                assert audio.format == expected_format
    
    def test_explicit_format(self) -> None:
        """Test explicit format specification."""
        with patch("pathlib.Path.exists", return_value=True):
            audio = AudioInput(path="audio.unknown", format="mp3")
            assert audio.format == "mp3"
    
    def test_nonexistent_file(self) -> None:
        """Test validation of nonexistent file."""
        with pytest.raises(ValueError, match="Audio file not found"):
            AudioInput(path="/nonexistent/file.mp3")
    
    def test_url_bypasses_existence_check(self) -> None:
        """Test that URLs bypass file existence check."""
        # Should not raise an error
        audio = AudioInput(path="https://example.com/nonexistent.mp3")
        assert str(audio.path) == "https://example.com/nonexistent.mp3"
    
    def test_language_setting(self) -> None:
        """Test language setting."""
        with patch("pathlib.Path.exists", return_value=True):
            audio = AudioInput(path="audio.mp3", language="en")
            assert audio.language == "en"


class TestTranscriptionRequest:
    """Test cases for TranscriptionRequest."""
    
    def test_valid_request(self) -> None:
        """Test valid transcription request."""
        with patch("pathlib.Path.exists", return_value=True):
            audio = AudioInput(path="audio.mp3")
            request = TranscriptionRequest(
                audio=audio,
                language="en",
                temperature=0.0
            )
            
            assert request.audio == audio
            assert request.language == "en"
            assert request.temperature == 0.0
    
    def test_default_temperature(self) -> None:
        """Test default temperature value."""
        with patch("pathlib.Path.exists", return_value=True):
            audio = AudioInput(path="audio.mp3")
            request = TranscriptionRequest(audio=audio)
            assert request.temperature == 0.0
    
    def test_temperature_validation(self) -> None:
        """Test temperature validation."""
        with patch("pathlib.Path.exists", return_value=True):
            audio = AudioInput(path="audio.mp3")
            
            # Valid temperatures
            TranscriptionRequest(audio=audio, temperature=0.0)
            TranscriptionRequest(audio=audio, temperature=1.0)
            TranscriptionRequest(audio=audio, temperature=2.0)
            
            # Invalid temperatures
            with pytest.raises(ValueError):
                TranscriptionRequest(audio=audio, temperature=-0.1)
            
            with pytest.raises(ValueError):
                TranscriptionRequest(audio=audio, temperature=2.1)


class TestAudioUnderstandingRequest:
    """Test cases for AudioUnderstandingRequest."""
    
    def test_valid_request(self) -> None:
        """Test valid audio understanding request."""
        with patch("pathlib.Path.exists", return_value=True):
            audio1 = AudioInput(path="audio1.mp3")
            audio2 = AudioInput(path="audio2.mp3")
            
            request = AudioUnderstandingRequest(
                audio_files=[audio1, audio2],
                question="What is discussed in these audio files?",
                temperature=0.2,
                top_p=0.95,
                max_tokens=500
            )
            
            assert len(request.audio_files) == 2
            assert request.question == "What is discussed in these audio files?"
            assert request.temperature == 0.2
            assert request.top_p == 0.95
            assert request.max_tokens == 500
    
    def test_default_values(self) -> None:
        """Test default parameter values."""
        with patch("pathlib.Path.exists", return_value=True):
            audio = AudioInput(path="audio.mp3")
            request = AudioUnderstandingRequest(
                audio_files=[audio],
                question="Test question"
            )
            
            assert request.temperature == 0.2
            assert request.top_p == 0.95
            assert request.max_tokens == 500
    
    def test_empty_audio_files(self) -> None:
        """Test validation of empty audio files list."""
        with pytest.raises(ValueError, match="At least one audio file must be provided"):
            AudioUnderstandingRequest(
                audio_files=[],
                question="Test question"
            )
    
    def test_parameter_validation(self) -> None:
        """Test parameter validation."""
        with patch("pathlib.Path.exists", return_value=True):
            audio = AudioInput(path="audio.mp3")
            
            # Valid parameters
            AudioUnderstandingRequest(
                audio_files=[audio],
                question="Test",
                temperature=0.0,
                top_p=0.0,
                max_tokens=1
            )
            
            AudioUnderstandingRequest(
                audio_files=[audio],
                question="Test",
                temperature=2.0,
                top_p=1.0,
                max_tokens=1000
            )
            
            # Invalid parameters
            with pytest.raises(ValueError):
                AudioUnderstandingRequest(
                    audio_files=[audio],
                    question="Test",
                    temperature=-0.1
                )
            
            with pytest.raises(ValueError):
                AudioUnderstandingRequest(
                    audio_files=[audio],
                    question="Test",
                    top_p=-0.1
                )
            
            with pytest.raises(ValueError):
                AudioUnderstandingRequest(
                    audio_files=[audio],
                    question="Test",
                    max_tokens=0
                )


class TestResponses:
    """Test cases for response types."""
    
    def test_transcription_response(self) -> None:
        """Test TranscriptionResponse."""
        response = TranscriptionResponse(
            content="Hello world",
            model="voxtral-mini",
            language="en",
            confidence=0.95,
            metadata={"processing_time": 1.5}
        )
        
        assert response.content == "Hello world"
        assert response.model == "voxtral-mini"
        assert response.language == "en"
        assert response.confidence == 0.95
        assert response.metadata["processing_time"] == 1.5
    
    def test_audio_understanding_response(self) -> None:
        """Test AudioUnderstandingResponse."""
        response = AudioUnderstandingResponse(
            content="The audio discusses...",
            model="voxtral-mini",
            audio_count=2,
            processing_time=3.2,
            usage={"total_tokens": 150}
        )
        
        assert response.content == "The audio discusses..."
        assert response.model == "voxtral-mini"
        assert response.audio_count == 2
        assert response.processing_time == 3.2
        assert response.usage["total_tokens"] == 150
