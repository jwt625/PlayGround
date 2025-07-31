#!/usr/bin/env python3
"""Example script for audio understanding using Voxtral."""

import asyncio
import logging
import sys
from pathlib import Path

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from voxtral import VoxtralClient, VoxtralConfig
from voxtral.types import AudioInput, AudioUnderstandingRequest
from voxtral.exceptions import VoxtralError


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


async def main() -> None:
    """Main function to demonstrate audio understanding."""
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
        
        # Example with multiple audio files from HuggingFace
        audio_files = [
            AudioInput(
                path="https://huggingface.co/datasets/hf-internal-testing/dummy-audio-samples/resolve/main/obama.mp3"
            ),
            AudioInput(
                path="https://huggingface.co/datasets/hf-internal-testing/dummy-audio-samples/resolve/main/bcn_weather.mp3"
            ),
        ]
        
        question = "Which speaker is more inspiring? Why? How are they different from each other?"
        
        request = AudioUnderstandingRequest(
            audio_files=audio_files,
            question=question,
            temperature=0.2,
            top_p=0.95,
            max_tokens=500
        )
        
        logger.info(f"Analyzing {len(audio_files)} audio files")
        logger.info(f"Question: {question}")
        
        response = await client.understand_audio(request)
        
        print("\n" + "="*80)
        print("AUDIO UNDERSTANDING RESULT")
        print("="*80)
        print(f"Model: {response.model}")
        print(f"Audio files processed: {response.audio_count}")
        print(f"Processing time: {response.processing_time:.2f}s")
        print(f"\nQuestion: {question}")
        print("\nAnswer:")
        print("-" * 40)
        print(response.content)
        print("="*80)
        
        # Follow-up question example
        follow_up_audio = [audio_files[0]]  # Just the first audio
        follow_up_question = "Please summarize the content of this audio."
        
        follow_up_request = AudioUnderstandingRequest(
            audio_files=follow_up_audio,
            question=follow_up_question,
            temperature=0.2,
            top_p=0.95,
            max_tokens=300
        )
        
        logger.info(f"Follow-up question: {follow_up_question}")
        follow_up_response = await client.understand_audio(follow_up_request)
        
        print("\n" + "="*80)
        print("FOLLOW-UP QUESTION RESULT")
        print("="*80)
        print(f"Question: {follow_up_question}")
        print("\nAnswer:")
        print("-" * 40)
        print(follow_up_response.content)
        print("="*80)
        
    except VoxtralError as e:
        logger.error(f"Voxtral error: {e.message}")
        if e.details:
            logger.error(f"Details: {e.details}")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")


if __name__ == "__main__":
    asyncio.run(main())
