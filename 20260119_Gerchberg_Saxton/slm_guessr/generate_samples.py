#!/usr/bin/env python3
"""
CLI script to generate SLM-Guessr training samples.

Usage:
    python -m slm_guessr.generate_samples
    
Or from the parent directory:
    python slm_guessr/generate_samples.py
"""

import sys
from pathlib import Path

# Ensure parent directory is in path for imports
script_dir = Path(__file__).parent
parent_dir = script_dir.parent
sys.path.insert(0, str(parent_dir))

from slm_guessr.generator import generate_all_samples


def main():
    """Generate all training samples."""
    # Output to slm-guessr/static/assets
    output_dir = parent_dir / "slm-guessr" / "static" / "assets"
    
    print("=" * 50)
    print("SLM-Guessr Sample Generator")
    print("=" * 50)
    print(f"Output directory: {output_dir}")
    print()
    
    # Generate samples
    manifest = generate_all_samples(output_dir)
    
    print()
    print("=" * 50)
    print(f"Generated {len(manifest['samples'])} samples")
    print("=" * 50)


if __name__ == "__main__":
    main()

