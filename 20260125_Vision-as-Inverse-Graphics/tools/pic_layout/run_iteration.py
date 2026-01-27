#!/usr/bin/env python3
"""Run iterative layout generation for chip1.png"""

import sys
import os
import json
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add gdsfactory venv to path if specified
venv_python = os.getenv('GDSFACTORY_VENV_PYTHON')
if venv_python:
    venv_site_packages = Path(venv_python).parent.parent / 'lib' / f'python{sys.version_info.major}.{sys.version_info.minor}' / 'site-packages'
    if venv_site_packages.exists():
        sys.path.insert(0, str(venv_site_packages))

from manual_test_pic_layout import test_generator_code, save_iteration_summary
from call_verifier import call_verifier, save_verifier_result

# Configuration
TARGET_IMAGE = "source/chip1.png"
OUTPUT_BASE = "output/chip1_iterations"
TARGET_DESCRIPTION = """
A photonic integrated circuit layout containing multiple components:
- Waveguide structures with bends and straight sections
- Multiple interconnected components
- Port connections and routing paths
- The layout appears to show a complex photonic circuit with various functional blocks
"""

def run_iteration(iteration, code_str):
    """Run one iteration: execute code, render, get verifier feedback."""
    
    print(f"\n{'='*80}")
    print(f"CHIP1 ITERATION {iteration}")
    print(f"{'='*80}\n")
    
    # Execute and render
    result = test_generator_code(code_str, iteration=iteration, output_base=OUTPUT_BASE)
    
    if not result['success']:
        print(f"\n✗ Code execution failed. Fix errors and try again.")
        return None
    
    # Save iteration summary (without feedback yet)
    save_iteration_summary(iteration, code_str, result, output_base=OUTPUT_BASE)
    
    # Call verifier
    print(f"\n[4] Calling Qwen3-VL verifier...")
    verifier_result = call_verifier(
        target_image_path=TARGET_IMAGE,
        current_image_path=result['png_path'],
        target_description=TARGET_DESCRIPTION,
        current_code=code_str
    )
    
    # Save verifier result
    verifier_path = Path(OUTPUT_BASE) / f"iteration_{iteration:02d}" / "verifier_feedback.json"
    save_verifier_result(verifier_result, verifier_path)
    
    print(f"\n{'='*80}")
    print(f"VERIFIER FEEDBACK:")
    print(f"{'='*80}")
    print(json.dumps(verifier_result['feedback'], indent=2))
    print(f"\nTokens used: {verifier_result['usage']['total_tokens']}")
    
    # Update summary with feedback
    save_iteration_summary(iteration, code_str, result, 
                          feedback=verifier_result['feedback'], 
                          output_base=OUTPUT_BASE)
    
    return verifier_result['feedback']


if __name__ == "__main__":
    print("Chip1 Iterative Layout Generation")
    print("="*80)
    print(f"Target: {TARGET_IMAGE}")
    print(f"Output: {OUTPUT_BASE}")
    print("="*80)
    
    # Check target exists
    if not Path(TARGET_IMAGE).exists():
        print(f"Error: Target image not found: {TARGET_IMAGE}")
        sys.exit(1)
    
    print("\nReady to start iterations.")
    print("Use: run_iteration(iteration_number, code_string)")

