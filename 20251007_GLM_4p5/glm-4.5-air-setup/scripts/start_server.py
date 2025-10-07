#!/usr/bin/env python3
"""Start GLM-4.5-Air server with optimal H100 configuration."""

import argparse
import logging
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from glm_server.api_server import run_server
from glm_server.config import get_optimized_h100_config


def setup_logging(level: str = "INFO") -> None:
    """Set up logging configuration."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("glm_server.log")
        ]
    )


def print_startup_info(config) -> None:
    """Print startup information."""
    print("=" * 80)
    print("🚀 GLM-4.5-Air Server Starting")
    print("=" * 80)
    print(f"📍 Server URL: http://{config.host}:{config.port}")
    print(f"🔑 API Key: {config.api_key}")
    print(f"🎯 Model: {config.model_path}")
    print(f"🔧 GPUs: {config.tensor_parallel_size}x H100")
    print(f"💾 GPU Memory: {config.gpu_memory_utilization * 100:.0f}%")
    print(f"📏 Max Length: {config.max_model_len:,} tokens")
    print(f"🔄 Max Sequences: {config.max_num_seqs}")
    print(f"🧠 Thinking Mode: {'✅' if config.enable_thinking_mode else '❌'}")
    print(f"🛠️  Tool Calling: {'✅' if config.tool_calling_enabled else '❌'}")
    print(f"🔐 Authentication: {'✅' if config.require_auth else '❌'}")
    print("=" * 80)
    print()
    print("📋 API Endpoints:")
    print(f"  • Health Check: GET  http://{config.host}:{config.port}/health")
    print(f"  • Completions:  POST http://{config.host}:{config.port}/v1/completions")
    print(f"  • Streaming:    POST http://{config.host}:{config.port}/v1/completions/stream")
    print(f"  • Generate:     POST http://{config.host}:{config.port}/generate")
    print(f"  • Config:       GET  http://{config.host}:{config.port}/config")
    print()
    print("🔑 Authentication Header:")
    print(f"  Authorization: Bearer {config.api_key}")
    print()
    print("📖 Example curl command:")
    print(f"""  curl -X POST "http://{config.host}:{config.port}/v1/completions" \\
    -H "Authorization: Bearer {config.api_key}" \\
    -H "Content-Type: application/json" \\
    -d '{{"prompt": "Hello, how are you?", "max_tokens": 100}}'""")
    print("=" * 80)


def main() -> None:
    """Main function."""
    parser = argparse.ArgumentParser(description="Start GLM-4.5-Air server")
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Host address to bind the server (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port number for the server (default: 8000)"
    )
    parser.add_argument(
        "--model-path",
        default="models/GLM-4.5-Air-FP8",
        help="Path to the GLM model (default: models/GLM-4.5-Air-FP8)"
    )
    parser.add_argument(
        "--api-key",
        help="Custom API key (if not provided, will be auto-generated)"
    )
    parser.add_argument(
        "--no-auth",
        action="store_true",
        help="Disable authentication (not recommended for production)"
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=32768,
        help="Maximum model length in tokens (default: 32768)"
    )
    parser.add_argument(
        "--max-num-seqs",
        type=int,
        default=512,
        help="Maximum number of sequences (default: 512)"
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.95,
        help="GPU memory utilization ratio (default: 0.95)"
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=2,
        help="Number of GPUs for tensor parallelism (default: 2)"
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)"
    )
    parser.add_argument(
        "--save-config",
        help="Save configuration to file"
    )

    args = parser.parse_args()

    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)

    # Check if model exists
    model_path = Path(args.model_path)
    if not model_path.exists():
        logger.error(f"Model path does not exist: {model_path}")
        print(f"❌ Error: Model path does not exist: {model_path}")
        print(f"💡 Make sure you have downloaded the model to: {model_path.absolute()}")
        sys.exit(1)

    # Get optimized configuration
    config = get_optimized_h100_config(str(model_path.absolute()))

    # Override with command line arguments
    config.host = args.host
    config.port = args.port
    config.max_model_len = args.max_model_len
    config.max_num_seqs = args.max_num_seqs
    config.gpu_memory_utilization = args.gpu_memory_utilization
    config.tensor_parallel_size = args.tensor_parallel_size
    config.require_auth = not args.no_auth

    if args.api_key:
        config.api_key = args.api_key

    # Save configuration if requested
    if args.save_config:
        config_dict = config.model_dump()
        import json
        with open(args.save_config, 'w') as f:
            json.dump(config_dict, f, indent=2)
        print(f"💾 Configuration saved to: {args.save_config}")

    # Print startup information
    print_startup_info(config)

    # Validate GPU availability
    try:
        import torch
        available_gpus = torch.cuda.device_count()
        if available_gpus < config.tensor_parallel_size:
            logger.error(f"Not enough GPUs available. Required: {config.tensor_parallel_size}, Available: {available_gpus}")
            sys.exit(1)

        print(f"🎮 GPU Check: {available_gpus} GPUs available, using {config.tensor_parallel_size}")

        # Print GPU information
        for i in range(available_gpus):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
            print(f"  GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")

    except ImportError:
        logger.warning("PyTorch not available for GPU validation")

    print("\n🚀 Starting server...")

    try:
        # Start the server
        run_server(
            host=config.host,
            port=config.port,
            model_path=config.model_path,
            log_level=args.log_level
        )
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
    except Exception as e:
        logger.error(f"Server error: {e}")
        print(f"❌ Server error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
