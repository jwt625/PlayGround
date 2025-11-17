#!/usr/bin/env python3
"""
Test client for the Qwen3-VL vLLM server.

This script demonstrates how to make authenticated requests to the server.
"""

import os
import sys
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get configuration
API_KEY = os.getenv("API_KEY")
HOST = os.getenv("HOST", "0.0.0.0")
PORT = os.getenv("PORT", "8000")
BASE_URL = f"http://{HOST}:{PORT}/v1"

if not API_KEY:
    print("ERROR: API_KEY not found in .env file!")
    sys.exit(1)

# Initialize OpenAI client with vLLM server
client = OpenAI(
    api_key=API_KEY,
    base_url=BASE_URL,
)

print("=" * 80)
print("Qwen3-VL vLLM Server Test Client")
print("=" * 80)
print(f"Server URL: {BASE_URL}")
print(f"API Key: {'*' * (len(API_KEY) - 4)}{API_KEY[-4:]}")
print("=" * 80)
print()

# Test 1: Text-only completion
print("Test 1: Text-only completion")
print("-" * 80)

try:
    response = client.chat.completions.create(
        model="Qwen/Qwen3-VL-32B-Instruct",
        messages=[
            {
                "role": "user",
                "content": "What is the capital of France? Answer in one sentence."
            }
        ],
        max_tokens=100,
        temperature=0.7,
    )
    
    print(f"Response: {response.choices[0].message.content}")
    print(f"Tokens used: {response.usage.total_tokens}")
    print()
    
except Exception as e:
    print(f"Error: {e}")
    print()

# Test 2: Image understanding (example with URL)
print("Test 2: Image understanding (with URL)")
print("-" * 80)
print("Note: This requires an image URL. Uncomment and modify the code below to test.")
print()

"""
# Uncomment to test with an image URL
try:
    response = client.chat.completions.create(
        model="Qwen/Qwen3-VL-32B-Instruct",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "https://example.com/your-image.jpg"
                        }
                    },
                    {
                        "type": "text",
                        "text": "What do you see in this image?"
                    }
                ]
            }
        ],
        max_tokens=200,
        temperature=0.7,
    )
    
    print(f"Response: {response.choices[0].message.content}")
    print(f"Tokens used: {response.usage.total_tokens}")
    print()
    
except Exception as e:
    print(f"Error: {e}")
    print()
"""

# Test 3: List available models
print("Test 3: List available models")
print("-" * 80)

try:
    models = client.models.list()
    print("Available models:")
    for model in models.data:
        print(f"  - {model.id}")
    print()
    
except Exception as e:
    print(f"Error: {e}")
    print()

print("=" * 80)
print("Testing complete!")
print("=" * 80)

