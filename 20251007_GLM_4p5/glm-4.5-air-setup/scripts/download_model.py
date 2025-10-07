#!/usr/bin/env python3
"""CLI script to download GLM-4.5-Air model."""

import argparse
import asyncio
import logging
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from glm_server.model_downloader import (
    ModelDownloader,
    ModelDownloadError,
    download_glm_model,
)


def setup_logging(level: str = "INFO") -> None:
    """Set up logging configuration."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)]
    )


def print_model_info(downloader: ModelDownloader) -> None:
    """Print detailed model information."""
    try:
        print("\n" + "="*60)
        print("GLM-4.5-Air Model Information")
        print("="*60)

        # Get basic model info
        model_info = downloader.get_model_info()
        print(f"Model ID: {model_info['model_id']}")
        print(f"Pipeline: {model_info.get('pipeline_tag', 'N/A')}")
        print(f"Library: {model_info.get('library_name', 'N/A')}")
        print(f"Downloads: {model_info.get('downloads', 'N/A'):,}")
        print(f"Likes: {model_info.get('likes', 'N/A'):,}")
        print(f"Last Modified: {model_info.get('last_modified', 'N/A')}")

        # Get size information
        print("\n" + "-"*40)
        print("Download Size Estimation")
        print("-"*40)

        size_info = downloader.estimate_download_size()
        print(f"Total Size: {size_info['total_size_gb']:.2f} GB")
        print(f"Total Files: {size_info['total_files']:,}")

        # Show large files
        if size_info['large_files']:
            print("\nLarge Files (>100MB):")
            for file_info in size_info['large_files'][:10]:  # Show top 10
                lfs_marker = " (LFS)" if file_info['lfs'] else ""
                print(f"  {file_info['path']}: {file_info['size_gb']:.2f} GB{lfs_marker}")

            if len(size_info['large_files']) > 10:
                print(f"  ... and {len(size_info['large_files']) - 10} more large files")

        # Show file type breakdown
        if size_info['file_types']:
            print("\nFile Type Breakdown:")
            for ext, size_gb in sorted(size_info['file_types'].items(),
                                     key=lambda x: x[1], reverse=True):
                if size_gb > 0.01:  # Only show files >10MB
                    ext_display = ext if ext else "(no extension)"
                    print(f"  {ext_display}: {size_gb:.2f} GB")

        print("\n" + "="*60)

    except ModelDownloadError as e:
        print(f"Error getting model info: {e}")
        sys.exit(1)


async def main() -> None:
    """Main async function."""
    parser = argparse.ArgumentParser(description="Download GLM-4.5-Air model")
    parser.add_argument(
        "--model-id",
        default="zai-org/GLM-4.5-Air",
        help="HuggingFace model identifier"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory for downloaded model"
    )
    parser.add_argument(
        "--token",
        help="HuggingFace authentication token"
    )
    parser.add_argument(
        "--info-only",
        action="store_true",
        help="Only show model information, don't download"
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )
    parser.add_argument(
        "--verify-only",
        type=Path,
        help="Only verify existing model at given path"
    )

    args = parser.parse_args()

    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)

    # Create downloader
    downloader = ModelDownloader(
        model_id=args.model_id,
        token=args.token
    )

    # Handle verify-only mode
    if args.verify_only:
        logger.info(f"Verifying model at: {args.verify_only}")
        try:
            verification = downloader.verify_model_files(args.verify_only)
            print("\n" + "="*50)
            print("Model Verification Results")
            print("="*50)
            print(f"Model Path: {verification['model_path']}")
            print(f"Total Size: {verification['total_size_gb']:.2f} GB")
            print(f"Weight Files: {verification['weight_files_count']}")
            print(f"Essential Files Found: {len(verification['essential_files_found'])}")

            if verification['essential_files_missing']:
                print(f"Missing Files: {verification['essential_files_missing']}")

            if verification['verification_passed']:
                print("\n✅ Model verification PASSED")
                sys.exit(0)
            else:
                print("\n❌ Model verification FAILED")
                sys.exit(1)

        except ModelDownloadError as e:
            logger.error(f"Verification failed: {e}")
            sys.exit(1)

    # Show model information
    print_model_info(downloader)

    # Handle info-only mode
    if args.info_only:
        logger.info("Info-only mode, exiting without download")
        return

    # Confirm download
    size_info = downloader.estimate_download_size()
    print(f"\nThis will download {size_info['total_size_gb']:.2f} GB of data.")

    if not args.output_dir:
        default_dir = Path.cwd() / "models" / args.model_id.replace("/", "_")
        print(f"Default download location: {default_dir}")

    response = input("\nProceed with download? [y/N]: ").strip().lower()
    if response not in ['y', 'yes']:
        print("Download cancelled.")
        return

    # Download model
    try:
        logger.info("Starting model download...")
        model_path = await download_glm_model(
            model_id=args.model_id,
            local_dir=args.output_dir,
            token=args.token
        )

        print(f"\n✅ Model successfully downloaded to: {model_path}")

        # Show final verification
        verification = downloader.verify_model_files(model_path)
        print(f"📊 Final size: {verification['total_size_gb']:.2f} GB")
        print(f"📁 Weight files: {verification['weight_files_count']}")

    except ModelDownloadError as e:
        logger.error(f"Download failed: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        logger.info("Download interrupted by user")
        sys.exit(1)


def cli_main() -> None:
    """CLI entry point."""
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        sys.exit(1)


if __name__ == "__main__":
    cli_main()
