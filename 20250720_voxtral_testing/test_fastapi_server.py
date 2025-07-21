#!/usr/bin/env python3
"""
Test script for the Voxtral FastAPI server.

This script tests the FastAPI endpoints to ensure they work correctly.
"""

import asyncio
import base64
import json
import sys
import time
from pathlib import Path
from typing import Dict, Any

import httpx

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent / "src"))


class FastAPIServerTest:
    """Test class for the FastAPI server."""
    
    def __init__(self, base_url: str = "http://localhost:8080") -> None:
        """Initialize the test client."""
        self.base_url = base_url
        self.client = httpx.AsyncClient(timeout=60.0)
    
    async def test_health_check(self) -> bool:
        """Test the health check endpoint."""
        try:
            print("Testing health check endpoint...")
            response = await self.client.get(f"{self.base_url}/health/")
            
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Health check passed: {data['status']} - {data['model']}")
                return True
            else:
                print(f"❌ Health check failed: {response.status_code} - {response.text}")
                return False
        
        except Exception as e:
            print(f"❌ Health check error: {e}")
            return False
    
    async def test_root_endpoint(self) -> bool:
        """Test the root endpoint."""
        try:
            print("Testing root endpoint...")
            response = await self.client.get(f"{self.base_url}/")
            
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Root endpoint passed: {data['name']} v{data['version']}")
                return True
            else:
                print(f"❌ Root endpoint failed: {response.status_code} - {response.text}")
                return False
        
        except Exception as e:
            print(f"❌ Root endpoint error: {e}")
            return False
    
    def _encode_audio_file(self, file_path: Path) -> str:
        """Encode an audio file to base64."""
        with open(file_path, "rb") as f:
            audio_data = f.read()
        return base64.b64encode(audio_data).decode("utf-8")
    
    async def test_transcription(self) -> bool:
        """Test the transcription endpoint."""
        try:
            print("Testing transcription endpoint...")
            
            # Check if test audio file exists
            audio_file = Path("test_data/obama_speech.mp3")
            if not audio_file.exists():
                print(f"❌ Test audio file not found: {audio_file}")
                return False
            
            # Encode audio file
            audio_data = self._encode_audio_file(audio_file)
            
            # Create request
            request_data = {
                "audio_file": audio_data,
                "format": "mp3",
                "language": "en",
                "temperature": 0.0
            }
            
            # Send request
            start_time = time.time()
            response = await self.client.post(
                f"{self.base_url}/transcribe/",
                json=request_data
            )
            request_time = time.time() - start_time
            
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Transcription passed in {request_time:.2f}s")
                print(f"   Processing time: {data['processing_time_ms']}ms")
                print(f"   Transcription: {data['transcription'][:100]}...")
                return True
            else:
                print(f"❌ Transcription failed: {response.status_code} - {response.text}")
                return False
        
        except Exception as e:
            print(f"❌ Transcription error: {e}")
            return False
    
    async def test_audio_understanding(self) -> bool:
        """Test the audio understanding endpoint."""
        try:
            print("Testing audio understanding endpoint...")
            
            # Check if test audio file exists
            audio_file = Path("test_data/obama_speech.mp3")
            if not audio_file.exists():
                print(f"❌ Test audio file not found: {audio_file}")
                return False
            
            # Encode audio file
            audio_data = self._encode_audio_file(audio_file)
            
            # Create request
            request_data = {
                "audio_files": [
                    {
                        "data": audio_data,
                        "format": "mp3",
                        "id": "test_audio"
                    }
                ],
                "question": "What is this audio about?",
                "temperature": 0.2,
                "max_tokens": 200,
                "top_p": 0.95
            }
            
            # Send request
            start_time = time.time()
            response = await self.client.post(
                f"{self.base_url}/understand/",
                json=request_data
            )
            request_time = time.time() - start_time
            
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Audio understanding passed in {request_time:.2f}s")
                print(f"   Processing time: {data['processing_time_ms']}ms")
                print(f"   Answer: {data['answer'][:100]}...")
                return True
            else:
                print(f"❌ Audio understanding failed: {response.status_code} - {response.text}")
                return False
        
        except Exception as e:
            print(f"❌ Audio understanding error: {e}")
            return False
    
    async def test_openapi_docs(self) -> bool:
        """Test that OpenAPI documentation is available."""
        try:
            print("Testing OpenAPI documentation...")
            response = await self.client.get(f"{self.base_url}/openapi.json")
            
            if response.status_code == 200:
                data = response.json()
                print(f"✅ OpenAPI docs available: {data['info']['title']} v{data['info']['version']}")
                return True
            else:
                print(f"❌ OpenAPI docs failed: {response.status_code}")
                return False
        
        except Exception as e:
            print(f"❌ OpenAPI docs error: {e}")
            return False
    
    async def run_all_tests(self) -> None:
        """Run all tests."""
        print(f"Starting FastAPI server tests against {self.base_url}")
        print("=" * 60)
        
        tests = [
            ("Root Endpoint", self.test_root_endpoint),
            ("Health Check", self.test_health_check),
            ("OpenAPI Docs", self.test_openapi_docs),
            ("Transcription", self.test_transcription),
            ("Audio Understanding", self.test_audio_understanding),
        ]
        
        results = []
        for test_name, test_func in tests:
            print(f"\n{test_name}:")
            result = await test_func()
            results.append((test_name, result))
        
        print("\n" + "=" * 60)
        print("Test Results:")
        
        passed = 0
        for test_name, result in results:
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"  {status} {test_name}")
            if result:
                passed += 1
        
        print(f"\nTotal: {passed}/{len(results)} tests passed")
        
        await self.client.aclose()


async def main() -> None:
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test the Voxtral FastAPI server")
    parser.add_argument(
        "--url",
        default="http://localhost:8080",
        help="Base URL of the FastAPI server (default: http://localhost:8080)"
    )
    
    args = parser.parse_args()
    
    tester = FastAPIServerTest(args.url)
    await tester.run_all_tests()


if __name__ == "__main__":
    asyncio.run(main())
