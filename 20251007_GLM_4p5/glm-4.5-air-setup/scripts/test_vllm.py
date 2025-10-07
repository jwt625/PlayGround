#!/usr/bin/env python3
"""Test script for vLLM GLM-4.5-Air inference."""

import asyncio
import argparse
import logging
import sys
import time
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from glm_server.config import InferenceConfig
from glm_server.vllm_server import create_glm_server


def setup_logging(level: str = "INFO") -> None:
    """Set up logging configuration."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)]
    )


async def test_basic_generation(server, prompt: str = "Hello, how are you?") -> None:
    """Test basic text generation."""
    print("\n" + "="*60)
    print("Testing Basic Generation")
    print("="*60)
    
    print(f"Prompt: {prompt}")
    print("-" * 40)
    
    start_time = time.time()
    result = await server.generate(
        prompt=prompt,
        max_tokens=100,
        temperature=0.7,
        top_p=0.9
    )
    generation_time = time.time() - start_time
    
    print(f"Response: {result['choices'][0]['text']}")
    print(f"Generation time: {generation_time:.2f} seconds")
    print(f"Tokens: {result['usage']['total_tokens']} total ({result['usage']['prompt_tokens']} prompt + {result['usage']['completion_tokens']} completion)")
    print(f"Tokens/sec: {result['usage']['completion_tokens'] / generation_time:.2f}")


async def test_streaming_generation(server, prompt: str = "Explain quantum computing in simple terms:") -> None:
    """Test streaming text generation."""
    print("\n" + "="*60)
    print("Testing Streaming Generation")
    print("="*60)
    
    print(f"Prompt: {prompt}")
    print("-" * 40)
    print("Streaming response:")
    
    start_time = time.time()
    token_count = 0
    
    async for chunk in await server.generate(
        prompt=prompt,
        max_tokens=200,
        temperature=0.7,
        stream=True
    ):
        if chunk['choices'][0]['text']:
            print(chunk['choices'][0]['text'], end='', flush=True)
            token_count += 1
        elif chunk['choices'][0]['finish_reason']:
            print(f"\n\nFinish reason: {chunk['choices'][0]['finish_reason']}")
            if 'usage' in chunk:
                print(f"Total tokens: {chunk['usage']['total_tokens']}")
    
    generation_time = time.time() - start_time
    print(f"Streaming time: {generation_time:.2f} seconds")
    print(f"Approximate tokens/sec: {token_count / generation_time:.2f}")


async def test_thinking_mode(server) -> None:
    """Test GLM-4.5-Air thinking mode capabilities."""
    print("\n" + "="*60)
    print("Testing Thinking Mode")
    print("="*60)
    
    # Complex reasoning prompt that should trigger thinking mode
    prompt = """Solve this step by step:
A train leaves Station A at 2:00 PM traveling at 60 mph toward Station B.
Another train leaves Station B at 2:30 PM traveling at 80 mph toward Station A.
The distance between the stations is 350 miles.
At what time will the trains meet?"""
    
    print(f"Complex reasoning prompt:")
    print(prompt)
    print("-" * 40)
    
    start_time = time.time()
    result = await server.generate(
        prompt=prompt,
        max_tokens=500,
        temperature=0.3,  # Lower temperature for reasoning
        top_p=0.9
    )
    generation_time = time.time() - start_time
    
    print(f"Response: {result['choices'][0]['text']}")
    print(f"Generation time: {generation_time:.2f} seconds")
    print(f"Tokens: {result['usage']['total_tokens']} total")


async def test_performance_benchmark(server, num_requests: int = 5) -> None:
    """Run performance benchmark."""
    print("\n" + "="*60)
    print(f"Performance Benchmark ({num_requests} requests)")
    print("="*60)
    
    prompts = [
        "Write a short story about a robot:",
        "Explain the theory of relativity:",
        "What are the benefits of renewable energy?",
        "Describe the process of photosynthesis:",
        "How does machine learning work?"
    ]
    
    total_time = 0
    total_tokens = 0
    
    for i in range(num_requests):
        prompt = prompts[i % len(prompts)]
        print(f"\nRequest {i+1}: {prompt[:50]}...")
        
        start_time = time.time()
        result = await server.generate(
            prompt=prompt,
            max_tokens=150,
            temperature=0.7
        )
        request_time = time.time() - start_time
        
        total_time += request_time
        total_tokens += result['usage']['completion_tokens']
        
        print(f"  Time: {request_time:.2f}s, Tokens: {result['usage']['completion_tokens']}, Rate: {result['usage']['completion_tokens']/request_time:.2f} tok/s")
    
    avg_time = total_time / num_requests
    avg_tokens_per_sec = total_tokens / total_time
    
    print(f"\n--- Benchmark Results ---")
    print(f"Average time per request: {avg_time:.2f} seconds")
    print(f"Average tokens per second: {avg_tokens_per_sec:.2f}")
    print(f"Total tokens generated: {total_tokens}")
    print(f"Total time: {total_time:.2f} seconds")


async def main() -> None:
    """Main async function."""
    parser = argparse.ArgumentParser(description="Test vLLM GLM-4.5-Air inference")
    parser.add_argument(
        "--model-path",
        default="models/GLM-4.5-Air-FP8",
        help="Path to the GLM model"
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=2,
        help="Tensor parallel size (number of GPUs)"
    )
    parser.add_argument(
        "--test",
        choices=["basic", "streaming", "thinking", "benchmark", "all"],
        default="all",
        help="Which test to run"
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )
    parser.add_argument(
        "--benchmark-requests",
        type=int,
        default=5,
        help="Number of requests for benchmark test"
    )
    
    args = parser.parse_args()
    
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    # Check if model exists
    model_path = Path(args.model_path)
    if not model_path.exists():
        logger.error(f"Model path does not exist: {model_path}")
        sys.exit(1)
    
    # Create configuration
    config = InferenceConfig(
        model_path=str(model_path.absolute()),
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=8192,  # Reasonable limit for testing
        gpu_memory_utilization=0.9
    )
    
    logger.info(f"Creating vLLM server with config: {config.model_dump()}")
    
    try:
        # Initialize server
        server = await create_glm_server(config)
        
        # Health check
        health = await server.health_check()
        print("\n" + "="*60)
        print("Health Check")
        print("="*60)
        print(f"Status: {health['status']}")
        if health['status'] == 'healthy':
            print(f"Model: {health['model']}")
            print(f"Tensor Parallel Size: {health['tensor_parallel_size']}")
            print(f"Startup Time: {health['startup_time']:.2f} seconds")
            print(f"Test Generation: {health['test_generation']['response']}")
        else:
            print(f"Error: {health['message']}")
            return
        
        # Run tests
        if args.test in ["basic", "all"]:
            await test_basic_generation(server)
        
        if args.test in ["streaming", "all"]:
            await test_streaming_generation(server)
        
        if args.test in ["thinking", "all"]:
            await test_thinking_mode(server)
        
        if args.test in ["benchmark", "all"]:
            await test_performance_benchmark(server, args.benchmark_requests)
        
        print("\n" + "="*60)
        print("All tests completed successfully!")
        print("="*60)
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        sys.exit(1)
    finally:
        if 'server' in locals():
            await server.shutdown()


def cli_main() -> None:
    """CLI entry point."""
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nTests interrupted by user")
        sys.exit(1)


if __name__ == "__main__":
    cli_main()
