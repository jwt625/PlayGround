"""
Path configuration for the thin film project.
All paths are relative to the project root.
"""

from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent

# Data directory
DATA_DIR = PROJECT_ROOT / 'data'

# Output directories
OUTPUT_DIR = PROJECT_ROOT / 'outputs'
OUTPUT_REFLECTANCE = OUTPUT_DIR / 'reflectance'
OUTPUT_COLOR = OUTPUT_DIR / 'color'
OUTPUT_ELLIPSOMETRY = OUTPUT_DIR / 'ellipsometry'
OUTPUT_THICKNESS = OUTPUT_DIR / 'thickness'
OUTPUT_ANALYSIS = OUTPUT_DIR / 'analysis'

# Data files
IMAGE_PATH = DATA_DIR / 'image.png'
ELLIPSE_PARAMS_PATH = DATA_DIR / 'ellipse_params.json'
THICKNESS_LUT_PATH = DATA_DIR / 'thickness_lut.npz'
THICKNESS_MAP_PATH = DATA_DIR / 'thickness_map.npy'

