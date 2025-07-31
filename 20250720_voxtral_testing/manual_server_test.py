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
import json
import requests
from pathlib import Path
from typing import Dict, Any
from datetime import datetime

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


def download_test_audio(url: str, local_path: Path) -> bool:
    """Download test audio file if it doesn't exist locally."""
    if local_path.exists():
        logger.info(f"✅ Audio file already exists: {local_path}")
        return True

    try:
        logger.info(f"📥 Downloading audio from: {url}")
        response = requests.get(url, stream=True)
        response.raise_for_status()

        # Create directory if it doesn't exist
        local_path.parent.mkdir(parents=True, exist_ok=True)

        # Download with progress
        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0

        with open(local_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        progress = (downloaded / total_size) * 100
                        print(f"\r📥 Download progress: {progress:.1f}%", end="", flush=True)

        print()  # New line after progress
        logger.info(f"✅ Downloaded: {local_path} ({downloaded:,} bytes)")
        return True

    except Exception as e:
        logger.error(f"❌ Failed to download audio: {e}")
        return False


def save_results(results: Dict[str, Any], output_file: Path) -> None:
    """Save test results to a JSON file."""
    try:
        # Add timestamp
        results["timestamp"] = datetime.now().isoformat()
        results["test_type"] = "voxtral_mini_3b"

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        logger.info(f"💾 Results saved to: {output_file}")

        # Also save a human-readable version
        txt_file = output_file.with_suffix('.txt')
        with open(txt_file, 'w', encoding='utf-8') as f:
            f.write("VOXTRAL MINI 3B TEST RESULTS\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Timestamp: {results['timestamp']}\n")
            f.write(f"Model: {results.get('model_name', 'Unknown')}\n\n")

            if 'transcription' in results:
                t = results['transcription']
                f.write("TRANSCRIPTION RESULTS:\n")
                f.write("-" * 25 + "\n")
                f.write(f"Processing Time: {t['processing_time']:.2f} seconds\n")
                f.write(f"Language: {t['language']}\n")
                f.write(f"Content:\n{t['content']}\n\n")

            if 'audio_understanding' in results:
                u = results['audio_understanding']
                f.write("AUDIO UNDERSTANDING RESULTS:\n")
                f.write("-" * 30 + "\n")
                f.write(f"Processing Time: {u['processing_time']:.2f} seconds\n")
                f.write(f"Question: {u['question']}\n")
                f.write(f"Answer:\n{u['answer']}\n\n")

        logger.info(f"📄 Human-readable results saved to: {txt_file}")

    except Exception as e:
        logger.error(f"❌ Failed to save results: {e}")


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


async def test_transcription() -> Dict[str, Any]:
    """Test basic transcription functionality."""
    logger.info("🎤 Testing transcription...")

    # Setup paths
    test_data_dir = Path("test_data")
    audio_file = test_data_dir / "obama_speech.mp3"

    # Download test audio
    audio_url = "https://huggingface.co/datasets/hf-internal-testing/dummy-audio-samples/resolve/main/obama.mp3"
    if not download_test_audio(audio_url, audio_file):
        return {"success": False, "error": "Failed to download test audio"}

    config = VoxtralConfig()
    client = VoxtralClient(config)

    try:
        logger.info(f"Using local audio file: {audio_file}")
        audio_input = AudioInput(path=str(audio_file), language="en")
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

            return {
                "success": True,
                "content": response.content,
                "language": response.language,
                "processing_time": processing_time
            }
        else:
            logger.error("❌ Transcription returned empty content")
            return {"success": False, "error": "Empty content"}

    except Exception as e:
        logger.error(f"❌ Transcription failed: {e}")
        return {"success": False, "error": str(e)}


async def test_audio_understanding() -> Dict[str, Any]:
    """Test basic audio understanding functionality."""
    logger.info("🧠 Testing audio understanding...")

    # Use the same audio file that was downloaded
    test_data_dir = Path("test_data")
    audio_file = test_data_dir / "obama_speech.mp3"

    if not audio_file.exists():
        return {"success": False, "error": "Audio file not found"}

    config = VoxtralConfig()
    client = VoxtralClient(config)

    try:
        audio_input = AudioInput(path=str(audio_file))
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

            return {
                "success": True,
                "question": "What is this audio about? Provide a brief summary.",
                "answer": response.content,
                "processing_time": processing_time
            }
        else:
            logger.error("❌ Audio understanding returned empty content")
            return {"success": False, "error": "Empty content"}

    except Exception as e:
        logger.error(f"❌ Audio understanding failed: {e}")
        return {"success": False, "error": str(e)}


async def main() -> None:
    """Main function to run manual tests."""
    logger.info("🚀 Starting Manual Voxtral Test")
    logger.info("=" * 50)
    logger.info("This test assumes vLLM server is already running!")
    logger.info("=" * 50)
    
    # Setup results file
    test_data_dir = Path("test_data")
    results_file = test_data_dir / "voxtral_mini_results.json"

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
            result = await test_func()

            if test_name == "Server Connection":
                # Server connection returns boolean
                success = result
                results[test_name] = {"success": success}
            else:
                # Other tests return dict with success flag
                success = result.get("success", False)
                results[test_name] = result

            if success:
                logger.info(f"✅ {test_name}: PASSED")
            else:
                logger.error(f"❌ {test_name}: FAILED")
                overall_success = False

        except Exception as e:
            logger.error(f"❌ {test_name}: ERROR - {e}")
            results[test_name] = {"success": False, "error": str(e)}
            overall_success = False
    
    # Print final results
    logger.info("\n" + "=" * 50)
    logger.info("📊 TEST RESULTS SUMMARY")
    logger.info("=" * 50)
    
    for test_name, result in results.items():
        success = result.get("success", False) if isinstance(result, dict) else result
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

        server_connection_success = results.get("Server Connection", {}).get("success", False)
        if not server_connection_success:
            logger.error("💡 Start the server with:")
            logger.error("   vllm serve mistralai/Voxtral-Mini-3B-2507 \\")
            logger.error("       --tokenizer_mode=mistral \\")
            logger.error("       --config_format=mistral \\")
            logger.error("       --load_format=mistral \\")
            logger.error("       --port=8000 \\")
            logger.error("       --host=localhost")

    # Save results
    save_results(results, results_file)
    
    logger.info("=" * 50)
    
    sys.exit(0 if overall_success else 1)


if __name__ == "__main__":
    asyncio.run(main())
