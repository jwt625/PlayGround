"""Main entry point for GLM-4.5-Air server."""

import asyncio
import logging
import sys
from typing import Any

from .config import get_config


def setup_logging(level: str = "INFO") -> None:
    """Set up logging configuration."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def check_system_requirements() -> dict[str, Any]:
    """Check system requirements for GLM-4.5-Air."""
    requirements: dict[str, Any] = {
        "torch_available": False,
        "cuda_available": False,
        "gpu_count": 0,
        "gpu_memory": [],
        "system_memory": 0,
    }

    try:
        import torch

        requirements["torch_available"] = True
        requirements["cuda_available"] = torch.cuda.is_available()
        requirements["gpu_count"] = torch.cuda.device_count()

        if requirements["cuda_available"]:
            gpu_count = int(requirements["gpu_count"])
            for i in range(gpu_count):
                gpu_props = torch.cuda.get_device_properties(i)
                gpu_memory_list = requirements["gpu_memory"]
                assert isinstance(gpu_memory_list, list)
                gpu_memory_list.append(
                    {
                        "device": i,
                        "name": gpu_props.name,
                        "total_memory": gpu_props.total_memory,
                        "memory_gb": round(gpu_props.total_memory / (1024**3), 2),
                    }
                )
    except ImportError:
        logging.error("PyTorch not available")

    try:
        import psutil

        requirements["system_memory"] = round(
            psutil.virtual_memory().total / (1024**3), 2
        )
    except ImportError:
        logging.warning("psutil not available, cannot check system memory")

    return requirements


async def main() -> None:
    """Main async function."""
    setup_logging()
    logger = logging.getLogger(__name__)

    logger.info("Starting GLM-4.5-Air server setup...")

    # Load configuration
    config = get_config()
    logger.info(f"Configuration loaded: {config.model_dump()}")

    # Check system requirements
    requirements = check_system_requirements()
    logger.info(f"System requirements check: {requirements}")

    # Validate requirements
    if not requirements["torch_available"]:
        logger.error("PyTorch is not available")
        sys.exit(1)

    if not requirements["cuda_available"]:
        logger.error("CUDA is not available")
        sys.exit(1)

    if requirements["gpu_count"] < config.tensor_parallel_size:
        logger.error(
            f"Insufficient GPUs: need {config.tensor_parallel_size}, "
            f"have {requirements['gpu_count']}"
        )
        sys.exit(1)

    total_gpu_memory = sum(gpu["memory_gb"] for gpu in requirements["gpu_memory"])
    logger.info(f"Total GPU memory: {total_gpu_memory:.2f} GB")

    if total_gpu_memory < 100:  # GLM-4.5-Air needs ~110GB
        logger.warning(
            f"GPU memory may be insufficient for GLM-4.5-Air: "
            f"{total_gpu_memory:.2f} GB available, ~110 GB recommended"
        )

    logger.info("System requirements check passed!")
    logger.info("GLM-4.5-Air server setup complete")


def cli_main() -> None:
    """CLI entry point."""
    asyncio.run(main())


if __name__ == "__main__":
    cli_main()
