#!/usr/bin/env python3
"""Example script for audio transcription using Voxtral."""

import asyncio
import logging
import sys
from pathlib import Path

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from voxtral import VoxtralClient, VoxtralConfig
from voxtral.types import AudioInput, TranscriptionRequest
from voxtral.exceptions import VoxtralError


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


async def main() -> None:
    """Main function to demonstrate transcription."""
    try:
        # Create configuration
        config = VoxtralConfig.from_env()
        logger.info(f"Connecting to server at {config.base_url}")
        
        # Create client
        client = VoxtralClient(config)
        
        # Check server health
        if not await client.health_check():
            logger.error("Server health check failed. Is the vLLM server running?")
            return
        
        logger.info("Server is healthy")
        
        # Example with remote audio file from HuggingFace
        audio_url = "https://huggingface.co/datasets/hf-internal-testing/dummy-audio-samples/resolve/main/obama.mp3"
        
        audio_input = AudioInput(path=audio_url, language="en")
        request = TranscriptionRequest(
            audio=audio_input,
            language="en",
            temperature=0.0
        )
        
        logger.info(f"Transcribing audio from: {audio_url}")
        response = await client.transcribe(request)
        
        print("\n" + "="*80)
        print("TRANSCRIPTION RESULT")
        print("="*80)
        print(f"Model: {response.model}")
        print(f"Language: {response.language}")
        print(f"Processing time: {response.metadata.get('processing_time', 'N/A'):.2f}s")
        print("\nTranscription:")
        print("-" * 40)
        print(response.content)
        print("="*80)
        
    except VoxtralError as e:
        logger.error(f"Voxtral error: {e.message}")
        if e.details:
            logger.error(f"Details: {e.details}")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")


if __name__ == "__main__":
    asyncio.run(main())
