#!/usr/bin/env python3
"""
Hunyuan3D API Server Test Script

This script tests the Hunyuan3D API server endpoints:
- Health check
- Synchronous 3D generation
- Asynchronous 3D generation with status tracking

Usage:
    uv run main.py
    # or
    uv run test-api

Requirements managed by uv in pyproject.toml

The script will:
1. Load API URL from .env file
2. Test all available endpoints
3. Use images from the 'images' folder
4. Save generated models to the 'output' folder
"""

import os
import sys
import time
import base64
import json
import requests
from pathlib import Path
from typing import Optional, Dict, Any
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Hunyuan3DAPITester:
    def __init__(self):
        self.base_url = os.getenv('API_BASE_URL')
        if not self.base_url:
            raise ValueError("API_BASE_URL not found in .env file")

        self.images_dir = Path("images")
        self.output_dir = Path("output")

        # Create output directory if it doesn't exist
        self.output_dir.mkdir(exist_ok=True)

        print(f" Hunyuan3D API Tester initialized")
        print(f" API Base URL: {self.base_url}")
        print(f" Images directory: {self.images_dir}")
        print(f" Output directory: {self.output_dir}")

    def encode_image_to_base64(self, image_path: Path) -> str:
        """Convert image file to base64 string."""
        try:
            with open(image_path, "rb") as image_file:
                encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
                return encoded_string
        except Exception as e:
            raise Exception(f"Failed to encode image {image_path}: {str(e)}")

    def save_model_from_base64(self, base64_data: str, filename: str) -> Path:
        """Save base64 encoded model data to file."""
        try:
            model_data = base64.b64decode(base64_data)
            output_path = self.output_dir / filename
            with open(output_path, "wb") as f:
                f.write(model_data)
            return output_path
        except Exception as e:
            raise Exception(f"Failed to save model to {filename}: {str(e)}")

    def test_health_check(self) -> bool:
        """Test the health check endpoint."""
        print("\n Testing health check endpoint...")
        try:
            response = requests.get(f"{self.base_url}/health", timeout=10)
            response.raise_for_status()

            health_data = response.json()
            print(f" Health check passed!")
            print(f"   Status: {health_data.get('status', 'Unknown')}")
            print(f"   Worker ID: {health_data.get('worker_id', 'Unknown')}")
            return True

        except requests.exceptions.RequestException as e:
            print(f" Health check failed: {str(e)}")
            return False

    def test_synchronous_generation(self, image_path: Path,
                                  texture: bool = False,
                                  remove_background: bool = True) -> Optional[Path]:
        """Test synchronous 3D model generation."""
        print(f"\n Testing synchronous generation with {image_path.name}...")

        try:
            # Encode image
            image_base64 = self.encode_image_to_base64(image_path)

            # Prepare request
            request_data = {
                "image": image_base64,
                "remove_background": remove_background,
                "texture": texture,
                "seed": 1234,
                "octree_resolution": 256,
                "num_inference_steps": 5,
                "guidance_scale": 5.0,
                "num_chunks": 8000,
                "face_count": 40000
            }

            print(f"    Sending generation request...")
            print(f"    Texture generation: {'enabled' if texture else 'disabled'}")
            print(f"     Background removal: {'enabled' if remove_background else 'disabled'}")

            # Send request
            response = requests.post(
                f"{self.base_url}/generate",
                json=request_data,
                timeout=300  # 5 minutes timeout
            )
            response.raise_for_status()

            # Handle response
            if response.headers.get('content-type', '').startswith('application/json'):
                # JSON response with base64 model
                result = response.json()
                if 'model_base64' in result:
                    model_base64 = result['model_base64']
                else:
                    print(f" No model data in response: {result}")
                    return None
            else:
                # Direct file response
                model_base64 = base64.b64encode(response.content).decode('utf-8')

            # Save model
            suffix = "_textured" if texture else "_no_texture"
            bg_suffix = "_nobg" if remove_background else "_withbg"
            filename = f"{image_path.stem}_sync{suffix}{bg_suffix}.glb"
            output_path = self.save_model_from_base64(model_base64, filename)

            print(f" Synchronous generation completed!")
            print(f"    Model saved to: {output_path}")
            return output_path

        except Exception as e:
            print(f" Synchronous generation failed: {str(e)}")
            return None

    def test_asynchronous_generation(self, image_path: Path,
                                   texture: bool = True,
                                   remove_background: bool = True) -> Optional[Path]:
        """Test asynchronous 3D model generation with status tracking."""
        print(f"\n Testing asynchronous generation with {image_path.name}...")

        try:
            # Encode image
            image_base64 = self.encode_image_to_base64(image_path)

            # Prepare request
            request_data = {
                "image": image_base64,
                "remove_background": remove_background,
                "texture": texture,
                "seed": 5678,  # Different seed for async test
                "octree_resolution": 256,
                "num_inference_steps": 5,
                "guidance_scale": 5.0,
                "num_chunks": 8000,
                "face_count": 40000
            }

            print(f"    Sending async generation request...")
            print(f"    Texture generation: {'enabled' if texture else 'disabled'}")
            print(f"     Background removal: {'enabled' if remove_background else 'disabled'}")

            # Send async request
            response = requests.post(
                f"{self.base_url}/send",
                json=request_data,
                timeout=30
            )
            response.raise_for_status()

            result = response.json()
            task_uid = result.get('uid')
            if not task_uid:
                print(f" No task UID in response: {result}")
                return None

            print(f"    Task UID: {task_uid}")
            print(f"     Polling for status...")

            # Poll for status
            max_attempts = 60  # 5 minutes with 5-second intervals
            for attempt in range(max_attempts):
                time.sleep(5)  # Wait 5 seconds between polls

                status_response = requests.get(
                    f"{self.base_url}/status/{task_uid}",
                    timeout=10
                )
                status_response.raise_for_status()

                status_data = status_response.json()
                status = status_data.get('status', 'unknown')

                # Show appropriate status message
                if status == 'pending':
                    status_msg = " Pending"
                elif status == 'processing':
                    status_msg = " Processing"
                elif status == 'texturing':
                    status_msg = " Texturing"
                elif status == 'completed':
                    status_msg = " Completed"
                elif status == 'error':
                    status_msg = " Error"
                else:
                    status_msg = f"  Unknown: {status}"

                print(f"    Attempt {attempt + 1}/{max_attempts}: {status_msg}")

                if status == 'completed':
                    model_base64 = status_data.get('model_base64')
                    if not model_base64:
                        print(f" No model data in completed response")
                        return None

                    # Save model
                    suffix = "_textured" if texture else "_no_texture"
                    bg_suffix = "_nobg" if remove_background else "_withbg"
                    filename = f"{image_path.stem}_async{suffix}{bg_suffix}.glb"
                    output_path = self.save_model_from_base64(model_base64, filename)

                    print(f" Asynchronous generation completed!")
                    print(f"    Model saved to: {output_path}")
                    return output_path

                elif status == 'error':
                    error_message = status_data.get('message', 'Unknown error')
                    print(f" Generation failed with error: {error_message}")
                    return None

                elif status in ['pending', 'processing', 'texturing']:
                    continue  # Keep polling

                else:
                    print(f"  Unexpected status: {status}")
                    continue

            print(f" Timeout: Task did not complete within {max_attempts * 5} seconds")
            return None

        except Exception as e:
            print(f" Asynchronous generation failed: {str(e)}")
            return None

    def run_all_tests(self):
        """Run all available tests."""
        print(" Starting Hunyuan3D API Tests")
        print("=" * 50)

        # Test health check first
        if not self.test_health_check():
            print(" Health check failed. Aborting tests.")
            return

        # Find test images
        image_files = list(self.images_dir.glob("*.png")) + \
                     list(self.images_dir.glob("*.jpg")) + \
                     list(self.images_dir.glob("*.jpeg"))

        if not image_files:
            print(f" No image files found in {self.images_dir}")
            return

        print(f"\n Found {len(image_files)} test image(s):")
        for img in image_files:
            print(f"   - {img.name}")

        # Test with first image
        test_image = image_files[0]

        # Test synchronous generation (without texture for speed)
        sync_result = self.test_synchronous_generation(
            test_image,
            texture=False,
            remove_background=True
        )

        # Test asynchronous generation (with texture)
        async_result = self.test_asynchronous_generation(
            test_image,
            texture=True,
            remove_background=True
        )

        # Summary
        print("\n Test Summary")
        print("=" * 50)
        print(f" Health check:  Passed")
        print(f" Sync generation: {' Passed' if sync_result else ' Failed'}")
        print(f" Async generation: {' Passed' if async_result else ' Failed'}")

        if sync_result or async_result:
            print(f"\n Generated models saved in: {self.output_dir}")
            output_files = list(self.output_dir.glob("*.glb"))
            for file in output_files:
                size_mb = file.stat().st_size / (1024 * 1024)
                print(f"   - {file.name} ({size_mb:.2f} MB)")


def main():
    """Main function to run the tests."""
    try:
        tester = Hunyuan3DAPITester()
        tester.run_all_tests()
    except Exception as e:
        print(f" Test setup failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
