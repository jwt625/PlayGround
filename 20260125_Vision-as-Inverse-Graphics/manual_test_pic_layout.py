#!/usr/bin/env python3
"""Manual test script for PIC layout generation workflow.

This script helps you manually drive the generator-verifier loop for testing.
You provide the code, it executes and renders, then you can send to verifier.
"""

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

from utils.gds_render import execute_and_render, get_component_info


def test_generator_code(code_str, iteration=1, output_base="output/manual_test"):
    """Test generator code by executing and rendering.
    
    Args:
        code_str: gdsfactory Python code to execute
        iteration: Iteration number for file naming
        output_base: Base directory for outputs
    
    Returns:
        dict with execution results
    """
    output_dir = Path(output_base) / f"iteration_{iteration:02d}"
    
    print(f"\n{'='*60}")
    print(f"ITERATION {iteration}")
    print(f"{'='*60}")
    
    print("\n[1] Executing code...")
    result = execute_and_render(code_str, output_dir, f"layout_iter{iteration:02d}")
    
    if not result['success']:
        print(f"✗ Execution failed!")
        print(f"  Error: {result['error']}")
        return result
    
    print(f"✓ Execution successful!")
    print(f"  GDS: {result['gds_path']}")
    print(f"  PNG: {result['png_path']}")
    
    print("\n[2] Component info:")
    info = get_component_info(result['component'])
    print(f"  Name: {info['name']}")
    print(f"  Size: {info['size']['width']:.2f} x {info['size']['height']:.2f} {info['size']['units']}")
    print(f"  Ports: {info['ports']}")
    
    # Save code for reference
    code_path = output_dir / f"code_iter{iteration:02d}.py"
    code_path.write_text(code_str)
    print(f"\n[3] Code saved: {code_path}")
    
    print(f"\n{'='*60}")
    print(f"Next steps:")
    print(f"  1. View the render: {result['png_path']}")
    print(f"  2. Compare with target image")
    print(f"  3. Send both images to Qwen3-VL verifier")
    print(f"  4. Use verifier feedback to refine code")
    print(f"{'='*60}\n")
    
    return result


def save_iteration_summary(iteration, code, result, feedback=None, output_base="output/manual_test"):
    """Save a summary of the iteration for tracking.
    
    Args:
        iteration: Iteration number
        code: Generator code
        result: Execution result dict
        feedback: Optional verifier feedback
        output_base: Base directory for outputs
    """
    output_dir = Path(output_base) / f"iteration_{iteration:02d}"
    summary_path = output_dir / "summary.json"
    
    summary = {
        "iteration": iteration,
        "code": code,
        "execution": {
            "success": result['success'],
            "gds_path": result.get('gds_path'),
            "png_path": result.get('png_path'),
            "error": result.get('error')
        }
    }
    
    if result['success']:
        info = get_component_info(result['component'])
        summary["component_info"] = info
    
    if feedback:
        summary["verifier_feedback"] = feedback
    
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"Summary saved: {summary_path}")


# Example usage
if __name__ == "__main__":
    print("PIC Layout Manual Test Script")
    print("=" * 60)
    
    # Example: Simple MZI test
    example_code = """
import gdsfactory as gf

# Create a simple MZI
c = gf.Component("test_mzi")
mzi = c << gf.components.mzi(delta_length=10)
c.add_ports(mzi.ports)
"""
    
    print("\nExample test with simple MZI:")
    print("-" * 60)
    print(example_code)
    print("-" * 60)
    
    result = test_generator_code(example_code, iteration=0, output_base="output/example_test")
    
    if result['success']:
        save_iteration_summary(0, example_code, result, output_base="output/example_test")
        print("\n✓ Example test completed successfully!")
        print(f"\nTo use this script for your own tests:")
        print(f"  1. Import: from manual_test_pic_layout import test_generator_code")
        print(f"  2. Run: test_generator_code(your_code, iteration=1)")
        print(f"  3. Review the output PNG")
        print(f"  4. Get verifier feedback from Qwen3-VL")
        print(f"  5. Iterate!")

