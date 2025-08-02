#!/usr/bin/env python3
"""
Test Script for Voxtral Small 24B

This script tests the Voxtral Small 24B model running on port 8001
and compares its performance with the Mini 3B version.
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
        results["test_type"] = "voxtral_small_24b"

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        logger.info(f"💾 Results saved to: {output_file}")

        # Also save a human-readable version
        txt_file = output_file.with_suffix('.txt')
        with open(txt_file, 'w', encoding='utf-8') as f:
            f.write("VOXTRAL SMALL 24B TEST RESULTS\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Timestamp: {results['timestamp']}\n")
            f.write(f"Model: {results.get('model_name', 'Unknown')}\n\n")

            if 'transcription' in results and results['transcription'].get('success'):
                t = results['transcription']
                f.write("TRANSCRIPTION RESULTS:\n")
                f.write("-" * 25 + "\n")
                f.write(f"Processing Time: {t['processing_time']:.2f} seconds\n")
                f.write(f"Language: {t['language']}\n")
                f.write(f"Content:\n{t['content']}\n\n")

            if 'audio_understanding' in results and results['audio_understanding'].get('success'):
                u = results['audio_understanding']
                f.write("AUDIO UNDERSTANDING RESULTS:\n")
                f.write("-" * 30 + "\n")
                f.write(f"Processing Time: {u['processing_time']:.2f} seconds\n")
                f.write(f"Question: {u['question']}\n")
                f.write(f"Answer:\n{u['answer']}\n\n")

        logger.info(f"📄 Human-readable results saved to: {txt_file}")

    except Exception as e:
        logger.error(f"❌ Failed to save results: {e}")


async def test_voxtral_small() -> Dict[str, Any]:
    """Test Voxtral Small 24B functionality."""
    logger.info("🚀 Testing Voxtral Small 24B Model")
    logger.info("=" * 60)

    # Setup paths
    test_data_dir = Path("test_data")
    audio_file = test_data_dir / "obama_speech.mp3"
    results_file = test_data_dir / "voxtral_small_results.json"

    # Download test audio
    audio_url = "https://huggingface.co/datasets/hf-internal-testing/dummy-audio-samples/resolve/main/obama.mp3"
    if not download_test_audio(audio_url, audio_file):
        return {"error": "Failed to download test audio"}

    # Configure for port 8001 (Voxtral Small)
    config = VoxtralConfig(
        server_host="localhost",
        server_port=8001,
        request_timeout=120.0
    )
    client = VoxtralClient(config)

    results = {}

    try:
        # Test 1: Server Connection
        logger.info("🔌 Testing server connection...")
        if not await client.health_check():
            logger.error("❌ Health check failed")
            return {"error": "Health check failed"}

        model_name = await client.get_model_name()
        logger.info(f"✅ Connected to: {model_name}")
        results["model_name"] = model_name
        results["audio_file"] = str(audio_file)

        # Test 2: Transcription
        logger.info("\n🎤 Testing transcription...")
        logger.info(f"Using local audio file: {audio_file}")
        
        audio_input = AudioInput(path=str(audio_file), language="en")
        request = TranscriptionRequest(
            audio=audio_input,
            language="en",
            temperature=0.0
        )
        
        start_time = time.time()
        response = await client.transcribe(request)
        transcription_time = time.time() - start_time
        
        if response.content and len(response.content.strip()) > 0:
            logger.info("✅ Transcription successful!")
            logger.info(f"   Content: {response.content[:100]}...")
            logger.info(f"   Language: {response.language}")
            logger.info(f"   Processing time: {transcription_time:.2f}s")
            
            results["transcription"] = {
                "content": response.content,  # Save full content
                "content_preview": response.content[:200],  # Keep preview for logs
                "language": response.language,
                "processing_time": transcription_time,
                "success": True
            }
        else:
            logger.error("❌ Transcription returned empty content")
            results["transcription"] = {"success": False, "error": "Empty content"}
        
        # Test 3: Audio Understanding
        logger.info("\n🧠 Testing audio understanding...")

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
        understanding_time = time.time() - start_time
        
        if response.content and len(response.content.strip()) > 0:
            logger.info("✅ Audio understanding successful!")
            logger.info(f"   Question: What is this audio about?")
            logger.info(f"   Answer: {response.content[:100]}...")
            logger.info(f"   Processing time: {understanding_time:.2f}s")
            
            results["audio_understanding"] = {
                "question": "What is this audio about? Provide a brief summary.",
                "answer": response.content,  # Save full answer
                "answer_preview": response.content[:200],  # Keep preview for logs
                "processing_time": understanding_time,
                "success": True
            }
        else:
            logger.error("❌ Audio understanding returned empty content")
            results["audio_understanding"] = {"success": False, "error": "Empty content"}
        
        # Save results
        save_results(results, results_file)
        return results

    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        error_results = {"error": str(e)}
        save_results(error_results, results_file)
        return error_results


async def main() -> None:
    """Main function to run Voxtral Small tests."""
    results = await test_voxtral_small()
    
    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("📊 VOXTRAL SMALL 24B TEST RESULTS")
    logger.info("=" * 60)
    
    if "error" in results:
        logger.error(f"❌ Test failed: {results['error']}")
        sys.exit(1)
    
    # Print results
    if "model_name" in results:
        logger.info(f"📋 Model: {results['model_name']}")
    
    if "transcription" in results and results["transcription"]["success"]:
        t_time = results["transcription"]["processing_time"]
        logger.info(f"✅ Transcription: {t_time:.2f}s")
    else:
        logger.error("❌ Transcription: Failed")
    
    if "audio_understanding" in results and results["audio_understanding"]["success"]:
        u_time = results["audio_understanding"]["processing_time"]
        logger.info(f"✅ Audio Understanding: {u_time:.2f}s")
    else:
        logger.error("❌ Audio Understanding: Failed")
    
    logger.info("=" * 60)
    
    # Check if all tests passed
    transcription_ok = results.get("transcription", {}).get("success", False)
    understanding_ok = results.get("audio_understanding", {}).get("success", False)
    
    if transcription_ok and understanding_ok:
        logger.info("🎉 ALL TESTS PASSED!")
        logger.info("✅ Voxtral Small 24B is working correctly")
        sys.exit(0)
    else:
        logger.error("❌ SOME TESTS FAILED!")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
