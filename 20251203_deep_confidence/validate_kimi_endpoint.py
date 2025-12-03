#!/usr/bin/env python3
"""
Validation script for kimi-k2 endpoint to verify planning doc claims.
Tests: multi-completion, logprobs, reasoning fields, and performance.
"""

import os
import time
import json
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Kimi-k2 endpoint configuration
KIMI_API_BASE = os.getenv("KIMI_API_BASE")
KIMI_API_KEY = os.getenv("KIMI_API_KEY", "")
KIMI_MODEL_ID = os.getenv("KIMI_MODEL_ID", "kimi-k2")

if not KIMI_API_BASE:
    raise ValueError("KIMI_API_BASE environment variable is required")

def test_basic_completion():
    """Test 1: Basic single completion"""
    print("\n" + "="*80)
    print("TEST 1: Basic Single Completion")
    print("="*80)

    client = OpenAI(api_key=KIMI_API_KEY or "dummy", base_url=KIMI_API_BASE)

    try:
        start_time = time.time()
        response = client.chat.completions.create(
            model=KIMI_MODEL_ID,
            messages=[{"role": "user", "content": "What is 15 + 27?"}],
            max_tokens=5000,
            temperature=0.7
        )
        elapsed = time.time() - start_time

        content = response.choices[0].message.content or ""
        print(f"✓ Basic completion successful ({elapsed:.2f}s)")
        print(f"Response: {content[:200]}")
        return True
    except Exception as e:
        print(f"✗ Basic completion failed: {e}")
        return False

def test_multiple_completions():
    """Test 2: Multiple completions (n=5)"""
    print("\n" + "="*80)
    print("TEST 2: Multiple Completions (n=5)")
    print("="*80)

    client = OpenAI(api_key=KIMI_API_KEY or "dummy", base_url=KIMI_API_BASE)

    try:
        start_time = time.time()
        response = client.chat.completions.create(
            model=KIMI_MODEL_ID,
            messages=[{"role": "user", "content": "What is 23 * 17?"}],
            max_tokens=5000,
            temperature=0.8,
            n=5
        )
        elapsed = time.time() - start_time
        
        num_choices = len(response.choices)
        print(f"✓ Multiple completions successful ({elapsed:.2f}s)")
        print(f"  Requested: 5, Received: {num_choices}")
        print(f"  Avg time per completion: {elapsed/num_choices:.2f}s")
        
        for i, choice in enumerate(response.choices):
            content = choice.message.content or ""
            reasoning = getattr(choice.message, 'reasoning', None) or ""
            print(f"\n  Choice {i+1}:")
            print(f"    Content: {content[:100] if content else '[EMPTY]'}...")
            print(f"    Reasoning: {reasoning[:100] if reasoning else '[EMPTY]'}...")

        return num_choices == 5
    except Exception as e:
        print(f"✗ Multiple completions failed: {e}")
        return False

def test_logprobs_support():
    """Test 3: Logprobs and top_logprobs support"""
    print("\n" + "="*80)
    print("TEST 3: Logprobs Support")
    print("="*80)

    client = OpenAI(api_key=KIMI_API_KEY or "dummy", base_url=KIMI_API_BASE)

    try:
        response = client.chat.completions.create(
            model=KIMI_MODEL_ID,
            messages=[{"role": "user", "content": "What is 5 + 3?"}],
            max_tokens=5000,
            temperature=0.7,
            logprobs=True,
            top_logprobs=3
        )
        
        choice = response.choices[0]
        has_logprobs = hasattr(choice, 'logprobs') and choice.logprobs is not None
        
        if has_logprobs:
            print(f"✓ Logprobs supported")
            print(f"  Logprobs object: {type(choice.logprobs)}")
            if hasattr(choice.logprobs, 'content') and choice.logprobs.content:
                print(f"  Number of tokens with logprobs: {len(choice.logprobs.content)}")
                if len(choice.logprobs.content) > 0:
                    first_token = choice.logprobs.content[0]
                    print(f"  First token: {first_token.token}")
                    print(f"  First token logprob: {first_token.logprob}")
                    if hasattr(first_token, 'top_logprobs') and first_token.top_logprobs:
                        print(f"  Top logprobs count: {len(first_token.top_logprobs)}")
            return True
        else:
            print(f"✗ Logprobs not available")
            return False
    except Exception as e:
        print(f"✗ Logprobs test failed: {e}")
        return False

def test_reasoning_fields():
    """Test 4: Reasoning field separation"""
    print("\n" + "="*80)
    print("TEST 4: Reasoning Field Separation")
    print("="*80)

    client = OpenAI(api_key=KIMI_API_KEY or "dummy", base_url=KIMI_API_BASE)

    try:
        response = client.chat.completions.create(
            model=KIMI_MODEL_ID,
            messages=[{"role": "user", "content": "Solve: If a train travels 120 km in 2 hours, what is its average speed?"}],
            max_tokens=5000,
            temperature=0.7
        )
        
        choice = response.choices[0]
        message = choice.message
        
        # Check for reasoning fields
        has_reasoning = hasattr(message, 'reasoning') or 'reasoning' in message.model_dump()
        has_reasoning_content = hasattr(message, 'reasoning_content') or 'reasoning_content' in message.model_dump()
        
        print(f"  Has 'reasoning' field: {has_reasoning}")
        print(f"  Has 'reasoning_content' field: {has_reasoning_content}")
        print(f"  Message attributes: {list(message.model_dump().keys())}")
        
        # Print full response for inspection
        print(f"\n  Full message dump:")
        print(json.dumps(message.model_dump(), indent=2))
        
        return True
    except Exception as e:
        print(f"✗ Reasoning fields test failed: {e}")
        return False

def test_high_volume_completions():
    """Test 5: High volume completions (n=10)"""
    print("\n" + "="*80)
    print("TEST 5: High Volume Completions (n=10)")
    print("="*80)
    
    client = OpenAI(api_key=KIMI_API_KEY or "dummy", base_url=KIMI_API_BASE)
    
    try:
        start_time = time.time()
        response = client.chat.completions.create(
            model=KIMI_MODEL_ID,
            messages=[{"role": "user", "content": "What is 144 / 12?"}],
            max_tokens=5000,
            temperature=0.8,
            n=10
        )
        elapsed = time.time() - start_time
        
        num_choices = len(response.choices)
        print(f"✓ High volume completions successful ({elapsed:.2f}s)")
        print(f"  Requested: 10, Received: {num_choices}")
        print(f"  Avg time per completion: {elapsed/num_choices:.2f}s")
        
        return num_choices == 10
    except Exception as e:
        print(f"✗ High volume completions failed: {e}")
        return False

if __name__ == "__main__":
    print("\n" + "="*80)
    print("KIMI-K2 ENDPOINT VALIDATION")
    print("="*80)
    print(f"Endpoint: {KIMI_API_BASE}")
    print(f"Model: {KIMI_MODEL_ID}")
    
    results = {
        "Basic Completion": test_basic_completion(),
        "Multiple Completions (n=5)": test_multiple_completions(),
        "Logprobs Support": test_logprobs_support(),
        "Reasoning Fields": test_reasoning_fields(),
        "High Volume (n=10)": test_high_volume_completions()
    }
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {test_name}")
    
    total = len(results)
    passed = sum(results.values())
    print(f"\nTotal: {passed}/{total} tests passed")

