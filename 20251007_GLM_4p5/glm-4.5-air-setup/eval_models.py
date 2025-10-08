#!/usr/bin/env python3
"""
Evaluation script to compare GLM 4.5 and Llama 4 Maverick performance on Q&A tasks.
Results are saved to a timestamped JSON file.
"""

import os
import json
import time
from datetime import datetime
from typing import Dict, List, Any
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

GLM_API_BASE = os.getenv("GLM_API_BASE")
GLM_API_KEY = os.getenv("GLM_API_KEY")
LAMBDA_API_BASE = os.getenv("LAMBDA_API_BASE")
LAMBDA_API_KEY = os.getenv("LAMBDA_API_KEY")
LAMBDA_MODEL = os.getenv("LAMBDA_MODEL", "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8")


# Test cases for evaluation
TEST_CASES = [
    {
        "category": "General Knowledge",
        "question": "What is the capital of France?",
        "system_prompt": "You are a helpful AI assistant."
    },
    {
        "category": "General Knowledge",
        "question": "Who wrote the novel '1984'?",
        "system_prompt": "You are a helpful AI assistant."
    },
    {
        "category": "Math",
        "question": "What is 15% of 240?",
        "system_prompt": "You are a helpful AI assistant. Provide clear explanations for mathematical problems."
    },
    {
        "category": "Science",
        "question": "What is photosynthesis and why is it important?",
        "system_prompt": "You are a helpful AI assistant with expertise in science."
    },
    {
        "category": "Programming",
        "question": "Explain what a binary search algorithm is and when to use it.",
        "system_prompt": "You are a helpful AI assistant with expertise in computer science and programming."
    },
    {
        "category": "History",
        "question": "What were the main causes of World War I?",
        "system_prompt": "You are a helpful AI assistant with expertise in history."
    },
    {
        "category": "Reasoning",
        "question": "If all roses are flowers and some flowers fade quickly, can we conclude that some roses fade quickly?",
        "system_prompt": "You are a helpful AI assistant. Think step by step through logical problems."
    },
    {
        "category": "Creative Writing",
        "question": "Write a short haiku about autumn.",
        "system_prompt": "You are a creative AI assistant."
    },
    {
        "category": "Language",
        "question": "What is the difference between 'affect' and 'effect'?",
        "system_prompt": "You are a helpful AI assistant with expertise in English language and grammar."
    },
    {
        "category": "Problem Solving",
        "question": "You have a 3-gallon jug and a 5-gallon jug. How can you measure exactly 4 gallons?",
        "system_prompt": "You are a helpful AI assistant. Think through problems step by step."
    }
]


def query_glm(question: str, system_prompt: str, max_tokens: int = 200, temperature: float = 0.7) -> Dict[str, Any]:
    """Query the GLM 4.5 model."""
    url = f"{GLM_API_BASE}/completions"
    headers = {
        "Authorization": f"Bearer {GLM_API_KEY}",
        "Content-Type": "application/json"
    }
    
    # Format prompt with system message
    prompt = f"[gMASK]<sop><|system|>\n{system_prompt}<|user|>\n{question}<|assistant|>\n"
    
    data = {
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature
    }
    
    start_time = time.time()
    try:
        response = requests.post(url, headers=headers, json=data, timeout=60)
        response.raise_for_status()
        elapsed_time = time.time() - start_time
        
        result = response.json()
        answer = ""
        if "choices" in result and len(result["choices"]) > 0:
            answer = result["choices"][0].get("text", "").strip()
        
        return {
            "success": True,
            "answer": answer,
            "response_time": elapsed_time,
            "error": None,
            "raw_response": result
        }
    except Exception as e:
        elapsed_time = time.time() - start_time
        return {
            "success": False,
            "answer": None,
            "response_time": elapsed_time,
            "error": str(e),
            "raw_response": None
        }


def query_lambda(question: str, system_prompt: str, max_tokens: int = 200, temperature: float = 0.7) -> Dict[str, Any]:
    """Query the Lambda (Llama 4 Maverick) model."""
    url = f"{LAMBDA_API_BASE}/chat/completions"
    headers = {
        "Authorization": f"Bearer {LAMBDA_API_KEY}",
        "Content-Type": "application/json"
    }
    
    data = {
        "model": LAMBDA_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question}
        ],
        "max_tokens": max_tokens,
        "temperature": temperature
    }
    
    start_time = time.time()
    try:
        response = requests.post(url, headers=headers, json=data, timeout=60)
        response.raise_for_status()
        elapsed_time = time.time() - start_time
        
        result = response.json()
        answer = ""
        if "choices" in result and len(result["choices"]) > 0:
            answer = result["choices"][0].get("message", {}).get("content", "").strip()
        
        return {
            "success": True,
            "answer": answer,
            "response_time": elapsed_time,
            "error": None,
            "raw_response": result
        }
    except Exception as e:
        elapsed_time = time.time() - start_time
        return {
            "success": False,
            "answer": None,
            "response_time": elapsed_time,
            "error": str(e),
            "raw_response": None
        }


def run_evaluation() -> Dict[str, Any]:
    """Run evaluation on all test cases for both models."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    results = {
        "timestamp": timestamp,
        "datetime": datetime.now().isoformat(),
        "models": {
            "glm": {
                "name": "GLM 4.5",
                "api_base": GLM_API_BASE
            },
            "lambda": {
                "name": "Llama 4 Maverick",
                "model": LAMBDA_MODEL,
                "api_base": LAMBDA_API_BASE
            }
        },
        "test_cases": []
    }
    
    print(f"\n{'='*80}")
    print(f"Starting Model Evaluation - {results['datetime']}")
    print(f"{'='*80}\n")
    
    for idx, test_case in enumerate(TEST_CASES, 1):
        print(f"\n[{idx}/{len(TEST_CASES)}] Category: {test_case['category']}")
        print(f"Question: {test_case['question']}")
        print("-" * 80)
        
        # Query GLM
        print("Querying GLM 4.5...")
        glm_result = query_glm(
            test_case["question"],
            test_case["system_prompt"]
        )
        
        if glm_result["success"]:
            print(f"✓ GLM Response ({glm_result['response_time']:.2f}s):")
            print(f"  {glm_result['answer'][:150]}{'...' if len(glm_result['answer']) > 150 else ''}")
        else:
            print(f"✗ GLM Error: {glm_result['error']}")
        
        # Small delay between requests
        time.sleep(1)
        
        # Query Lambda
        print("\nQuerying Llama 4 Maverick...")
        lambda_result = query_lambda(
            test_case["question"],
            test_case["system_prompt"]
        )
        
        if lambda_result["success"]:
            print(f"✓ Lambda Response ({lambda_result['response_time']:.2f}s):")
            print(f"  {lambda_result['answer'][:150]}{'...' if len(lambda_result['answer']) > 150 else ''}")
        else:
            print(f"✗ Lambda Error: {lambda_result['error']}")
        
        # Store results
        test_result = {
            "test_number": idx,
            "category": test_case["category"],
            "question": test_case["question"],
            "system_prompt": test_case["system_prompt"],
            "glm_response": glm_result,
            "lambda_response": lambda_result
        }
        
        results["test_cases"].append(test_result)
        
        # Small delay between test cases
        time.sleep(1)
    
    return results, timestamp


def save_results(results: Dict[str, Any], timestamp: str):
    """Save results to a JSON file."""
    filename = f"eval_results_{timestamp}.json"
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*80}")
    print(f"Results saved to: {filename}")
    print(f"{'='*80}\n")
    
    return filename


def print_summary(results: Dict[str, Any]):
    """Print a summary of the evaluation results."""
    print("\n" + "="*80)
    print("EVALUATION SUMMARY")
    print("="*80)
    
    glm_successes = sum(1 for tc in results["test_cases"] if tc["glm_response"]["success"])
    lambda_successes = sum(1 for tc in results["test_cases"] if tc["lambda_response"]["success"])
    total_tests = len(results["test_cases"])
    
    glm_times = [tc["glm_response"]["response_time"] for tc in results["test_cases"] if tc["glm_response"]["success"]]
    lambda_times = [tc["lambda_response"]["response_time"] for tc in results["test_cases"] if tc["lambda_response"]["success"]]
    
    print(f"\nTotal Test Cases: {total_tests}")
    print(f"\nGLM 4.5:")
    print(f"  Success Rate: {glm_successes}/{total_tests} ({glm_successes/total_tests*100:.1f}%)")
    if glm_times:
        print(f"  Avg Response Time: {sum(glm_times)/len(glm_times):.2f}s")
        print(f"  Min/Max Time: {min(glm_times):.2f}s / {max(glm_times):.2f}s")
    
    print(f"\nLlama 4 Maverick:")
    print(f"  Success Rate: {lambda_successes}/{total_tests} ({lambda_successes/total_tests*100:.1f}%)")
    if lambda_times:
        print(f"  Avg Response Time: {sum(lambda_times)/len(lambda_times):.2f}s")
        print(f"  Min/Max Time: {min(lambda_times):.2f}s / {max(lambda_times):.2f}s")
    
    print("\n" + "="*80)


def main():
    """Main execution function."""
    # Verify environment variables
    if not all([GLM_API_BASE, GLM_API_KEY, LAMBDA_API_BASE, LAMBDA_API_KEY]):
        print("Error: Missing required environment variables.")
        print("Please ensure .env file contains:")
        print("  - GLM_API_BASE")
        print("  - GLM_API_KEY")
        print("  - LAMBDA_API_BASE")
        print("  - LAMBDA_API_KEY")
        return 1

    # Run evaluation
    results, timestamp = run_evaluation()

    # Save results
    filename = save_results(results, timestamp)

    # Print summary
    print_summary(results)

    print(f"\nEvaluation complete! Results saved to: {filename}\n")

    # Check if Lambda endpoint had issues
    lambda_failures = sum(1 for tc in results["test_cases"] if not tc["lambda_response"]["success"])
    if lambda_failures == len(results["test_cases"]):
        print("WARNING: All Lambda endpoint requests failed.")
        print("This may be due to network connectivity issues.")
        print("The Lambda API might not be accessible from this instance.\n")

    return 0


if __name__ == "__main__":
    exit(main())

