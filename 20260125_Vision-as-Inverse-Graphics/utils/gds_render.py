"""Utilities for rendering GDS layouts to PNG images."""

import sys
import os
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

import gdsfactory as gf
import matplotlib.pyplot as plt


def render_gds_to_png(component, output_path, dpi=300):
    """Render gdsfactory component to PNG.
    
    Args:
        component: gdsfactory Component object
        output_path: Path to save PNG file
        dpi: Resolution in dots per inch (default: 300)
    
    Returns:
        Path to saved PNG file
    
    Note:
        gdsfactory 9.23.0 plot() does not accept show_ports parameter.
        Ports are shown by default in the plot.
    """
    fig = component.plot()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close(fig)
    return output_path


def execute_and_render(code_str, output_dir, component_name="layout"):
    """Execute gdsfactory code and render to PNG.
    
    Args:
        code_str: Python code string to execute
        output_dir: Directory to save outputs
        component_name: Base name for output files
    
    Returns:
        dict with keys:
            - 'gds_path': Path to GDS file
            - 'png_path': Path to PNG render
            - 'component': gdsfactory Component object
            - 'success': True if successful, False otherwise
            - 'error': Error message if failed
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Execute code in isolated namespace
        namespace = {'gf': gf, '__name__': '__main__'}
        exec(code_str, namespace)
        
        # Find the component (expect variable named 'c')
        component = namespace.get('c')
        if component is None:
            component = namespace.get('component')
        
        if component is None:
            return {
                'success': False,
                'error': "No component found. Code must create a component named 'c' or 'component'."
            }
        
        # Save GDS
        gds_path = output_dir / f'{component_name}.gds'
        component.write_gds(gds_path)
        
        # Render PNG
        png_path = output_dir / f'{component_name}.png'
        render_gds_to_png(component, png_path, dpi=300)
        
        return {
            'success': True,
            'gds_path': str(gds_path),
            'png_path': str(png_path),
            'component': component,
            'error': None
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': f"{type(e).__name__}: {str(e)}"
        }


def get_component_info(component):
    """Extract useful information from a gdsfactory component.

    Args:
        component: gdsfactory Component object

    Returns:
        dict with component information
    """
    return {
        'name': component.name,
        'size': {
            'width': component.xsize,
            'height': component.ysize,
            'units': 'µm'
        },
        'ports': [p.name for p in component.ports],
        'num_ports': len(component.ports)
    }


if __name__ == "__main__":
    # Test the rendering utility
    print("Testing GDS rendering utility...")
    
    test_code = """
import gdsfactory as gf

c = gf.Component("test_mzi")
mzi = c << gf.components.mzi(delta_length=10)
c.add_ports(mzi.ports)
"""
    
    result = execute_and_render(test_code, "output/test_utils", "test_mzi")
    
    if result['success']:
        print(f"✓ Success!")
        print(f"  GDS: {result['gds_path']}")
        print(f"  PNG: {result['png_path']}")
        info = get_component_info(result['component'])
        print(f"  Component: {info['name']}")
        print(f"  Size: {info['size']['width']:.2f} x {info['size']['height']:.2f} {info['size']['units']}")
        print(f"  Ports: {info['ports']}")
    else:
        print(f"✗ Failed: {result['error']}")

