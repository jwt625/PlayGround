#!/usr/bin/env python3
"""
Test script for Kimi-K2-Thinking OpenAI-compatible API
"""

import openai
import sys
import os

# Read API key from file
API_KEY_FILE = "/home/ubuntu/fs2/kimi_K2_thinking/.api_key"
try:
    with open(API_KEY_FILE, 'r') as f:
        api_key = f.read().strip()
except FileNotFoundError:
    print(f"Error: API key file not found at {API_KEY_FILE}")
    print("Please run ./launch_kimi_k2.sh first to generate the API key")
    sys.exit(1)

# Configure OpenAI client to use local vLLM server
client = openai.OpenAI(
    base_url="http://localhost:8000/v1",
    api_key=api_key
)

def test_simple_chat():
    """Test basic chat completion"""
    print("=" * 60)
    print("Test 1: Simple Chat Completion")
    print("=" * 60)
    
    messages = [
        {"role": "system", "content": "You are Kimi, an AI assistant created by Moonshot AI."},
        {"role": "user", "content": "Which one is bigger, 9.11 or 9.9? Think carefully."}
    ]
    
    try:
        response = client.chat.completions.create(
            model="kimi-k2-thinking",
            messages=messages,
            temperature=1.0,
            max_tokens=4096
        )
        
        print(f"\n✓ Response received")
        print(f"Answer: {response.choices[0].message.content}")
        
        if hasattr(response.choices[0].message, 'reasoning_content'):
            print(f"\n--- Reasoning Content ---")
            print(response.choices[0].message.reasoning_content)
        
        return True
    except Exception as e:
        print(f"\n✗ Error: {e}")
        return False

def test_streaming():
    """Test streaming response"""
    print("\n" + "=" * 60)
    print("Test 2: Streaming Response")
    print("=" * 60)
    
    messages = [
        {"role": "system", "content": "You are Kimi, an AI assistant created by Moonshot AI."},
        {"role": "user", "content": "Count from 1 to 5."}
    ]
    
    try:
        stream = client.chat.completions.create(
            model="kimi-k2-thinking",
            messages=messages,
            temperature=1.0,
            max_tokens=100,
            stream=True
        )
        
        print("\n✓ Streaming response:")
        for chunk in stream:
            if chunk.choices[0].delta.content:
                print(chunk.choices[0].delta.content, end='', flush=True)
        print("\n")
        return True
    except Exception as e:
        print(f"\n✗ Error: {e}")
        return False

def test_model_list():
    """Test model listing endpoint"""
    print("\n" + "=" * 60)
    print("Test 3: List Available Models")
    print("=" * 60)
    
    try:
        models = client.models.list()
        print(f"\n✓ Available models:")
        for model in models.data:
            print(f"  - {model.id}")
        return True
    except Exception as e:
        print(f"\n✗ Error: {e}")
        return False

def main():
    print("\n" + "=" * 60)
    print("Kimi-K2-Thinking API Test Suite")
    print("=" * 60)
    print(f"Server: http://localhost:8000")
    print("=" * 60 + "\n")
    
    results = []
    
    # Run tests
    results.append(("Model List", test_model_list()))
    results.append(("Simple Chat", test_simple_chat()))
    results.append(("Streaming", test_streaming()))
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {test_name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    print("=" * 60 + "\n")
    
    return 0 if passed == total else 1

if __name__ == "__main__":
    sys.exit(main())

