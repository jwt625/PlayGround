#!/usr/bin/env python3
"""
Manual Server Test for Voxtral Mini 3B

This script assumes you have already started the vLLM server manually.
It only tests the client functionality without managing the server.

To start the server manually:
    vllm serve mistralai/Voxtral-Mini-3B-2507 \
        --tokenizer_mode=mistral \
        --config_format=mistral \
        --load_format=mistral \
        --port=8000 \
        --host=localhost

Then run this script to test the client functionality.
"""

import asyncio
import logging
import sys
import time
from pathlib import Path
from typing import Dict, Any

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent / "src"))

from voxtral import VoxtralClient, VoxtralConfig
from voxtral.types import AudioInput, TranscriptionRequest, AudioUnderstandingRequest
from voxtral.exceptions import VoxtralError


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


async def test_server_connection() -> bool:
    """Test if we can connect to the server."""
    logger.info("🔌 Testing server connection...")
    
    config = VoxtralConfig()
    client = VoxtralClient(config)
    
    try:
        if await client.health_check():
            logger.info("✅ Server is healthy and responding")
            
            model_name = await client.get_model_name()
            logger.info(f"✅ Model loaded: {model_name}")
            return True
        else:
            logger.error("❌ Server health check failed")
            return False
            
    except Exception as e:
        logger.error(f"❌ Connection failed: {e}")
        logger.error("💡 Make sure the vLLM server is running on localhost:8000")
        return False


async def test_transcription() -> bool:
    """Test basic transcription functionality."""
    logger.info("🎤 Testing transcription...")
    
    config = VoxtralConfig()
    client = VoxtralClient(config)
    
    try:
        # Use a simple test audio URL
        audio_url = "https://huggingface.co/datasets/hf-internal-testing/dummy-audio-samples/resolve/main/obama.mp3"
        
        audio_input = AudioInput(path=audio_url, language="en")
        request = TranscriptionRequest(
            audio=audio_input,
            language="en",
            temperature=0.0
        )
        
        start_time = time.time()
        response = await client.transcribe(request)
        processing_time = time.time() - start_time
        
        if response.content and len(response.content.strip()) > 0:
            logger.info("✅ Transcription successful!")
            logger.info(f"   Content: {response.content[:100]}...")
            logger.info(f"   Language: {response.language}")
            logger.info(f"   Processing time: {processing_time:.2f}s")
            return True
        else:
            logger.error("❌ Transcription returned empty content")
            return False
            
    except Exception as e:
        logger.error(f"❌ Transcription failed: {e}")
        return False


async def test_audio_understanding() -> bool:
    """Test basic audio understanding functionality."""
    logger.info("🧠 Testing audio understanding...")
    
    config = VoxtralConfig()
    client = VoxtralClient(config)
    
    try:
        # Use the same test audio
        audio_url = "https://huggingface.co/datasets/hf-internal-testing/dummy-audio-samples/resolve/main/obama.mp3"
        
        audio_input = AudioInput(path=audio_url)
        request = AudioUnderstandingRequest(
            audio_files=[audio_input],
            question="What is this audio about? Provide a brief summary.",
            temperature=0.2,
            top_p=0.95,
            max_tokens=150
        )
        
        start_time = time.time()
        response = await client.understand_audio(request)
        processing_time = time.time() - start_time
        
        if response.content and len(response.content.strip()) > 0:
            logger.info("✅ Audio understanding successful!")
            logger.info(f"   Question: What is this audio about?")
            logger.info(f"   Answer: {response.content[:100]}...")
            logger.info(f"   Processing time: {processing_time:.2f}s")
            return True
        else:
            logger.error("❌ Audio understanding returned empty content")
            return False
            
    except Exception as e:
        logger.error(f"❌ Audio understanding failed: {e}")
        return False


async def main() -> None:
    """Main function to run manual tests."""
    logger.info("🚀 Starting Manual Voxtral Test")
    logger.info("=" * 50)
    logger.info("This test assumes vLLM server is already running!")
    logger.info("=" * 50)
    
    tests = [
        ("Server Connection", test_server_connection),
        ("Basic Transcription", test_transcription),
        ("Basic Audio Understanding", test_audio_understanding),
    ]
    
    results = {}
    overall_success = True
    
    for test_name, test_func in tests:
        logger.info(f"\n📋 Running: {test_name}")
        logger.info("-" * 30)
        
        try:
            success = await test_func()
            results[test_name] = success
            
            if success:
                logger.info(f"✅ {test_name}: PASSED")
            else:
                logger.error(f"❌ {test_name}: FAILED")
                overall_success = False
                
        except Exception as e:
            logger.error(f"❌ {test_name}: ERROR - {e}")
            results[test_name] = False
            overall_success = False
    
    # Print final results
    logger.info("\n" + "=" * 50)
    logger.info("📊 TEST RESULTS SUMMARY")
    logger.info("=" * 50)
    
    for test_name, success in results.items():
        status = "✅ PASSED" if success else "❌ FAILED"
        logger.info(f"{test_name:.<25} {status}")
    
    logger.info("-" * 50)
    
    if overall_success:
        logger.info("🎉 ALL TESTS PASSED!")
        logger.info("✅ Voxtral Mini 3B is working correctly")
        logger.info("✅ Ready for next development phase")
    else:
        logger.error("❌ SOME TESTS FAILED!")
        logger.error("❌ Check the errors above")
        
        if not results.get("Server Connection", False):
            logger.error("💡 Start the server with:")
            logger.error("   vllm serve mistralai/Voxtral-Mini-3B-2507 \\")
            logger.error("       --tokenizer_mode=mistral \\")
            logger.error("       --config_format=mistral \\")
            logger.error("       --load_format=mistral \\")
            logger.error("       --port=8000 \\")
            logger.error("       --host=localhost")
    
    logger.info("=" * 50)
    
    sys.exit(0 if overall_success else 1)


if __name__ == "__main__":
    asyncio.run(main())
