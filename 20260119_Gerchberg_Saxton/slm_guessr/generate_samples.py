#!/usr/bin/env python3
"""
CLI script to generate SLM-Guessr training samples.

Usage:
    python slm_guessr/generate_samples.py                    # Generate all
    python slm_guessr/generate_samples.py --list             # List available samples
    python slm_guessr/generate_samples.py --samples id1 id2  # Generate specific samples
"""

import argparse
import sys
from pathlib import Path

# Ensure parent directory is in path for imports
script_dir = Path(__file__).parent
parent_dir = script_dir.parent
sys.path.insert(0, str(parent_dir))

from slm_guessr.generator import (
    generate_all_samples,
    generate_selected_samples,
    get_all_samples,
)


def main():
    """Generate training samples."""
    parser = argparse.ArgumentParser(description="Generate SLM-Guessr training samples")
    parser.add_argument(
        "--samples", "-s",
        nargs="+",
        help="Sample IDs to generate (default: all)"
    )
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="List available sample IDs"
    )
    args = parser.parse_args()

    all_samples = get_all_samples()

    if args.list:
        print("Available samples:")
        for s in all_samples:
            print(f"  {s.id:30} L{s.level} - {s.name}")
        return

    # Output to slm-guessr/static/assets
    output_dir = parent_dir / "slm-guessr" / "static" / "assets"

    print("=" * 50)
    print("SLM-Guessr Sample Generator")
    print("=" * 50)
    print(f"Output directory: {output_dir}")
    print()

    if args.samples:
        # Generate selected samples only
        manifest = generate_selected_samples(output_dir, args.samples)
    else:
        # Generate all samples
        manifest = generate_all_samples(output_dir)

    print()
    print("=" * 50)
    print(f"Generated {len(manifest['samples'])} samples")
    print("=" * 50)


if __name__ == "__main__":
    main()

