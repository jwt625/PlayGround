#!/usr/bin/env python3
"""
Minimal Quick Test for Voxtral Mini 3B Setup

This script performs a quick verification that the Voxtral Mini 3B model
and basic functionalities are working before proceeding with full implementation.

Test Steps:
1. Check if vLLM server can start
2. Verify model loads correctly
3. Test basic transcription functionality
4. Test basic audio understanding functionality
5. Report results and readiness for next phase
"""

import asyncio
import logging
import sys
import time
from pathlib import Path
from typing import Dict, Any, Optional

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent / "src"))

from voxtral import VoxtralClient, VoxtralConfig
from voxtral.types import AudioInput, TranscriptionRequest, AudioUnderstandingRequest
from voxtral.exceptions import VoxtralError
from examples.server_manager import ServerManager


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class QuickTest:
    """Minimal quick test for Voxtral setup verification."""

    def __init__(self) -> None:
        """Initialize the quick test."""
        self.config = VoxtralConfig(
            server_host="localhost",
            server_port=8000,
            request_timeout=120.0  # Longer timeout for initial model loading
        )
        self.client = VoxtralClient(self.config)
        self.server_manager = ServerManager(self.config)
        self.results: Dict[str, Any] = {}
        self.server_started_by_us = False

        # Test audio URLs from HuggingFace
        self.test_audio_url = "https://huggingface.co/datasets/hf-internal-testing/dummy-audio-samples/resolve/main/obama.mp3"
    
    async def run_all_tests(self) -> bool:
        """Run all quick tests and return overall success status."""
        logger.info("🚀 Starting Voxtral Mini 3B Quick Test")
        logger.info("=" * 60)
        
        tests = [
            ("Server Check", self.test_server_check),
            ("Model Loading", self.test_model_loading),
            ("Basic Transcription", self.test_basic_transcription),
            ("Basic Audio Understanding", self.test_basic_audio_understanding),
        ]
        
        overall_success = True
        
        for test_name, test_func in tests:
            logger.info(f"\n📋 Running: {test_name}")
            logger.info("-" * 40)
            
            try:
                success = await test_func()
                self.results[test_name] = {
                    "success": success,
                    "error": None
                }
                
                if success:
                    logger.info(f"✅ {test_name}: PASSED")
                else:
                    logger.error(f"❌ {test_name}: FAILED")
                    overall_success = False
                    
            except Exception as e:
                logger.error(f"❌ {test_name}: ERROR - {e}")
                self.results[test_name] = {
                    "success": False,
                    "error": str(e)
                }
                overall_success = False
        
        # Print final results
        self.print_final_results(overall_success)
        
        # Cleanup
        await self.cleanup()
        
        return overall_success
    
    async def test_server_check(self) -> bool:
        """Test 1: Check if vLLM server is running, start if needed."""
        try:
            logger.info("Checking if vLLM server is already running...")

            # First check if server is already running
            if await self.client.health_check():
                logger.info("✅ Server is already running and healthy")
                return True

            logger.info("No server detected. Starting vLLM server...")
            logger.info("⚠️  This will download the model (~9.5GB) on first run")
            logger.info("⚠️  Server startup may take 2-5 minutes...")

            if not self.server_manager.start_server():
                logger.error("Failed to start vLLM server process")
                return False

            self.server_started_by_us = True
            logger.info("Waiting for server to become healthy...")

            if not await self.server_manager.wait_for_health(timeout=300):  # 5 minutes for model download + loading
                logger.error("Server failed to become healthy within timeout")
                return False

            logger.info("✅ Server started successfully and is healthy")
            return True

        except Exception as e:
            logger.error(f"Server check failed: {e}")
            return False
    
    async def test_model_loading(self) -> bool:
        """Test 2: Verify model loads correctly and responds to basic queries."""
        try:
            logger.info("Testing model loading and basic health check...")
            
            # Check health
            if not await self.client.health_check():
                logger.error("Health check failed")
                return False
            
            # Get model name
            model_name = await self.client.get_model_name()
            if not model_name:
                logger.error("Failed to get model name")
                return False
            
            logger.info(f"✅ Model loaded successfully: {model_name}")
            self.results["model_name"] = model_name
            return True
            
        except Exception as e:
            logger.error(f"Model loading test failed: {e}")
            return False
    
    async def test_basic_transcription(self) -> bool:
        """Test 3: Test basic transcription functionality."""
        try:
            logger.info(f"Testing transcription with sample audio: {self.test_audio_url}")
            
            # Create transcription request
            audio_input = AudioInput(path=self.test_audio_url, language="en")
            request = TranscriptionRequest(
                audio=audio_input,
                language="en",
                temperature=0.0
            )
            
            # Perform transcription
            start_time = time.time()
            response = await self.client.transcribe(request)
            processing_time = time.time() - start_time
            
            # Validate response
            if not response.content or len(response.content.strip()) == 0:
                logger.error("Transcription returned empty content")
                return False
            
            logger.info(f"✅ Transcription successful!")
            logger.info(f"   Content: {response.content[:100]}...")
            logger.info(f"   Language: {response.language}")
            logger.info(f"   Processing time: {processing_time:.2f}s")
            
            self.results["transcription"] = {
                "content": response.content[:200],  # Store first 200 chars
                "language": response.language,
                "processing_time": processing_time
            }
            
            return True
            
        except Exception as e:
            logger.error(f"Transcription test failed: {e}")
            return False
    
    async def test_basic_audio_understanding(self) -> bool:
        """Test 4: Test basic audio understanding functionality."""
        try:
            logger.info("Testing audio understanding with sample question...")
            
            # Create audio understanding request
            audio_input = AudioInput(path=self.test_audio_url)
            request = AudioUnderstandingRequest(
                audio_files=[audio_input],
                question="What is this audio about? Provide a brief summary.",
                temperature=0.2,
                top_p=0.95,
                max_tokens=150
            )
            
            # Perform audio understanding
            start_time = time.time()
            response = await self.client.understand_audio(request)
            processing_time = time.time() - start_time
            
            # Validate response
            if not response.content or len(response.content.strip()) == 0:
                logger.error("Audio understanding returned empty content")
                return False
            
            logger.info(f"✅ Audio understanding successful!")
            logger.info(f"   Question: What is this audio about?")
            logger.info(f"   Answer: {response.content[:100]}...")
            logger.info(f"   Processing time: {processing_time:.2f}s")
            
            self.results["audio_understanding"] = {
                "question": "What is this audio about? Provide a brief summary.",
                "answer": response.content[:200],  # Store first 200 chars
                "processing_time": processing_time
            }
            
            return True
            
        except Exception as e:
            logger.error(f"Audio understanding test failed: {e}")
            return False
    
    def print_final_results(self, overall_success: bool) -> None:
        """Print final test results summary."""
        logger.info("\n" + "=" * 60)
        logger.info("📊 QUICK TEST RESULTS SUMMARY")
        logger.info("=" * 60)
        
        for test_name, result in self.results.items():
            if test_name in ["model_name", "transcription", "audio_understanding"]:
                continue  # Skip metadata

            status = "✅ PASSED" if result["success"] else "❌ FAILED"
            logger.info(f"{test_name:.<30} {status}")

            if not result["success"] and result["error"]:
                logger.info(f"  Error: {result['error']}")
        
        logger.info("-" * 60)
        
        if overall_success:
            logger.info("🎉 ALL TESTS PASSED!")
            logger.info("✅ Voxtral Mini 3B setup is working correctly")
            logger.info("✅ Ready to proceed with FastAPI server implementation")
            
            # Print some key metrics
            if "model_name" in self.results:
                logger.info(f"📋 Model: {self.results['model_name']}")
            
            if "transcription" in self.results:
                t_time = self.results["transcription"]["processing_time"]
                logger.info(f"📋 Transcription time: {t_time:.2f}s")
            
            if "audio_understanding" in self.results:
                u_time = self.results["audio_understanding"]["processing_time"]
                logger.info(f"📋 Understanding time: {u_time:.2f}s")
                
        else:
            logger.error("❌ SOME TESTS FAILED!")
            logger.error("❌ Please review the errors above before proceeding")
            logger.error("❌ Check server logs and configuration")
        
        logger.info("=" * 60)
    
    async def cleanup(self) -> None:
        """Clean up resources after testing."""
        try:
            logger.info("🧹 Cleaning up...")
            # Only stop server if we started it
            if self.server_started_by_us and self.server_manager.is_running():
                logger.info("Stopping server (started by this test)...")
                self.server_manager.stop_server()
                logger.info("✅ Server stopped")
            elif self.server_manager.is_running():
                logger.info("✅ Server left running (was already running before test)")
        except Exception as e:
            logger.warning(f"Cleanup warning: {e}")


async def main() -> None:
    """Main function to run the quick test."""
    test = QuickTest()
    
    try:
        success = await test.run_all_tests()
        sys.exit(0 if success else 1)
        
    except KeyboardInterrupt:
        logger.info("\n🛑 Test interrupted by user")
        await test.cleanup()
        sys.exit(1)
        
    except Exception as e:
        logger.error(f"💥 Unexpected error: {e}")
        await test.cleanup()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
