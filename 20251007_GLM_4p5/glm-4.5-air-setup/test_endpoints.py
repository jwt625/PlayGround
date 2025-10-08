#!/usr/bin/env python3
"""Test script to verify both GLM and Lambda API endpoints are working."""

import os
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

GLM_API_BASE = os.getenv("GLM_API_BASE")
GLM_API_KEY = os.getenv("GLM_API_KEY")
LAMBDA_API_BASE = os.getenv("LAMBDA_API_BASE")
LAMBDA_API_KEY = os.getenv("LAMBDA_API_KEY")
LAMBDA_MODEL = os.getenv("LAMBDA_MODEL", "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8")


def test_glm_endpoint():
    """Test the GLM 4.5 vLLM endpoint."""
    print("\n" + "="*60)
    print("Testing GLM 4.5 Endpoint")
    print("="*60)
    
    url = f"{GLM_API_BASE}/completions"
    headers = {
        "Authorization": f"Bearer {GLM_API_KEY}",
        "Content-Type": "application/json"
    }
    
    # Format prompt with system message
    prompt = "[gMASK]<sop><|system|>\nYou are a helpful AI assistant.<|user|>\nWhat is the capital of France?<|assistant|>\n"
    
    data = {
        "prompt": prompt,
        "max_tokens": 100,
        "temperature": 0.7
    }
    
    try:
        print(f"Sending request to: {url}")
        response = requests.post(url, headers=headers, json=data, timeout=30)
        response.raise_for_status()
        
        result = response.json()
        print("\nResponse received successfully!")
        print(f"Status Code: {response.status_code}")
        print(f"\nGenerated text:")
        if "choices" in result and len(result["choices"]) > 0:
            print(result["choices"][0].get("text", ""))
        else:
            print(result)
        
        return True
    except requests.exceptions.RequestException as e:
        print(f"\nError testing GLM endpoint: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"Response status: {e.response.status_code}")
            print(f"Response body: {e.response.text}")
        return False


def test_lambda_endpoint():
    """Test the Lambda (Llama 4 Maverick) endpoint."""
    print("\n" + "="*60)
    print("Testing Lambda (Llama 4 Maverick) Endpoint")
    print("="*60)
    
    url = f"{LAMBDA_API_BASE}/chat/completions"
    headers = {
        "Authorization": f"Bearer {LAMBDA_API_KEY}",
        "Content-Type": "application/json"
    }
    
    data = {
        "model": LAMBDA_MODEL,
        "messages": [
            {"role": "system", "content": "You are a helpful AI assistant."},
            {"role": "user", "content": "What is the capital of France?"}
        ],
        "max_tokens": 100,
        "temperature": 0.7
    }
    
    try:
        print(f"Sending request to: {url}")
        response = requests.post(url, headers=headers, json=data, timeout=30)
        response.raise_for_status()
        
        result = response.json()
        print("\nResponse received successfully!")
        print(f"Status Code: {response.status_code}")
        print(f"\nGenerated text:")
        if "choices" in result and len(result["choices"]) > 0:
            print(result["choices"][0].get("message", {}).get("content", ""))
        else:
            print(result)
        
        return True
    except requests.exceptions.RequestException as e:
        print(f"\nError testing Lambda endpoint: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"Response status: {e.response.status_code}")
            print(f"Response body: {e.response.text}")
        return False


def main():
    """Run tests for both endpoints."""
    print("Starting endpoint tests...")
    print(f"\nGLM API Base: {GLM_API_BASE}")
    print(f"Lambda API Base: {LAMBDA_API_BASE}")
    
    glm_success = test_glm_endpoint()
    lambda_success = test_lambda_endpoint()
    
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    print(f"GLM 4.5 Endpoint: {'✓ PASSED' if glm_success else '✗ FAILED'}")
    print(f"Lambda Endpoint: {'✓ PASSED' if lambda_success else '✗ FAILED'}")
    print()
    
    if glm_success and lambda_success:
        print("All endpoints are working correctly!")
        return 0
    else:
        print("Some endpoints failed. Please check the errors above.")
        return 1


if __name__ == "__main__":
    exit(main())

