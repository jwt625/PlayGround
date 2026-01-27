#!/usr/bin/env python3
"""Test gdsfactory rendering to PNG."""

import sys
sys.path.insert(0, '/Users/wentaojiang/Documents/GitHub/PlayGround/20251129_gdsfactory/.venv/lib/python3.12/site-packages')

import gdsfactory as gf
import matplotlib.pyplot as plt
from pathlib import Path

def test_render():
    """Test rendering a simple MZI to PNG."""
    print("Creating MZI component...")
    c = gf.components.mzi(delta_length=10)
    
    # Create output directory
    output_dir = Path(__file__).parent / "output" / "test_render"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save GDS
    gds_path = output_dir / "test_mzi.gds"
    c.write_gds(gds_path)
    print(f"✓ GDS saved: {gds_path}")
    
    # Render to PNG
    png_path = output_dir / "test_mzi.png"
    fig = c.plot()
    plt.savefig(png_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"✓ PNG saved: {png_path}")
    
    # Print component info
    print(f"\nComponent info:")
    print(f"  Name: {c.name}")
    print(f"  Size: {c.xsize:.2f} x {c.ysize:.2f} µm")
    print(f"  Ports: {[p.name for p in c.ports]}")
    
    return png_path

if __name__ == "__main__":
    png_path = test_render()
    print(f"\n✓ Test completed successfully!")
    print(f"  View output: {png_path}")

