"""
Configuration file for Bay Bridge Traffic Cam - Car and Pedestrian Detection
"""

# Webcam Configuration
WEBCAM_BASE_URL = "http://192.168.12.6:4747"
WEBCAM_DEFAULT_ENDPOINT = "/video"
WEBCAM_TIMEOUT = 10  # seconds

# YOLO Model Configuration
MODEL_NAME = "yolov8n.pt"  # Options: yolov8n.pt (fastest), yolov8s.pt, yolov8m.pt, yolov8l.pt, yolov8x.pt (most accurate)

# Detection Classes and Parameters
# COCO dataset class IDs and their detection settings
DETECTION_CLASSES = {
    0: {
        'name': 'person',
        'color': (255, 0, 0),      # Blue (BGR format)
        'min_confidence': 0.3,     # Lower threshold for pedestrians (they can be smaller/harder to detect)
        'enabled': True
    },
    2: {
        'name': 'car', 
        'color': (0, 255, 0),      # Green
        'min_confidence': 0.4,     # Lowered for smaller cars in zoomed out view
        'enabled': True
    },
    3: {
        'name': 'motorcycle',
        'color': (0, 255, 255),    # Yellow
        'min_confidence': 0.4,     # Motorcycles can be small
        'enabled': True
    },
    5: {
        'name': 'bus',
        'color': (255, 0, 255),    # Magenta
        'min_confidence': 0.5,     # Buses are usually large and easier to detect
        'enabled': True
    },
    7: {
        'name': 'truck',
        'color': (0, 128, 255),    # Orange
        'min_confidence': 0.5,     # Trucks are usually large
        'enabled': True
    },
    1: {
        'name': 'bicycle',
        'color': (255, 255, 0),    # Cyan
        'min_confidence': 0.3,     # Bicycles can be small
        'enabled': False           # Disabled by default, set to True to enable
    }
}

# Visual Settings
BOUNDING_BOX_THICKNESS = 2
LABEL_FONT = 1  # cv2.FONT_HERSHEY_SIMPLEX
LABEL_FONT_SCALE = 0.5
LABEL_FONT_THICKNESS = 2
LABEL_TEXT_COLOR = (255, 255, 255)  # White text on colored background

# Output Settings
OUTPUT_BASE_DIR = "outputs"
SAVE_ORIGINAL_FRAME = True
SAVE_ANNOTATED_FRAME = True
TIMESTAMP_FORMAT = "%Y%m%d_%H%M%S"

# Demo Mode Settings
DEMO_IMAGE_SIZE = (640, 480)  # Width, Height
DEMO_BACKGROUND_COLOR = (100, 50, 0)  # BGR format

# Resolution Settings
USE_HIGHEST_RESOLUTION = True  # Always try to get the highest resolution available
FORCE_RESOLUTION = None  # Set to (width, height) to force specific resolution, or None for auto
MIN_RESOLUTION = (640, 480)  # Minimum acceptable resolution
PREFERRED_ASPECT_RATIO = "landscape"  # "landscape", "portrait", or "auto"

# Advanced Detection Settings
# Set to True to enable additional processing for small objects
ENHANCE_SMALL_OBJECTS = True  # Enabled by default for better small object detection
SMALL_OBJECT_ENHANCEMENT_FACTOR = 1.2  # Conservative upscaling to avoid too much processing

# Webcam Fallback Endpoints (tried in order if main endpoint fails)
FALLBACK_ENDPOINTS = [
    "/video",
    "/video.mjpg", 
    "/mjpeg",
    "/stream",
    "/cam.mjpg",
    "/video.cgi",
    "/videostream.cgi",
    "/mjpegfeed"
]

# Performance Settings
MAX_DETECTIONS_PER_FRAME = 100  # Limit to prevent performance issues
VERBOSE_YOLO = False  # Set to True for detailed YOLO output

# Confidence Adjustment Presets
# You can quickly switch between these by changing CONFIDENCE_PRESET
CONFIDENCE_PRESETS = {
    "high_precision": {
        "person": 0.6,
        "car": 0.7,
        "motorcycle": 0.6,
        "bus": 0.7,
        "truck": 0.7
    },
    "balanced": {
        "person": 0.4,
        "car": 0.5,
        "motorcycle": 0.4,
        "bus": 0.5,
        "truck": 0.5
    },
    "high_recall": {  # Good for zoomed out views with small objects
        "person": 0.25,
        "car": 0.3,
        "motorcycle": 0.25,
        "bus": 0.4,
        "truck": 0.4
    }
}

# Active preset - change this to switch confidence levels quickly
CONFIDENCE_PRESET = "high_recall"  # Options: "high_precision", "balanced", "high_recall"

def get_active_confidence_thresholds():
    """Get the confidence thresholds based on the active preset."""
    preset = CONFIDENCE_PRESETS.get(CONFIDENCE_PRESET, CONFIDENCE_PRESETS["balanced"])
    
    # Update DETECTION_CLASSES with preset values
    updated_classes = DETECTION_CLASSES.copy()
    for class_id, class_info in updated_classes.items():
        class_name = class_info['name']
        if class_name in preset:
            class_info['min_confidence'] = preset[class_name]
    
    return updated_classes

def print_config_summary():
    """Print a summary of current configuration."""
    print("=== CONFIGURATION SUMMARY ===")
    print(f"Model: {MODEL_NAME}")
    print(f"Confidence Preset: {CONFIDENCE_PRESET}")
    print(f"Webcam URL: {WEBCAM_BASE_URL}{WEBCAM_DEFAULT_ENDPOINT}")

    print(f"\nResolution Settings:")
    print(f"  Use highest resolution: {USE_HIGHEST_RESOLUTION}")
    print(f"  Force resolution: {FORCE_RESOLUTION if FORCE_RESOLUTION else 'Auto'}")
    print(f"  Preferred aspect ratio: {PREFERRED_ASPECT_RATIO}")
    print(f"  Enhanced small objects: {ENHANCE_SMALL_OBJECTS}")
    if ENHANCE_SMALL_OBJECTS:
        print(f"  Enhancement factor: {SMALL_OBJECT_ENHANCEMENT_FACTOR}x")

    active_classes = get_active_confidence_thresholds()
    print("\nDetection Classes:")
    for class_id, info in active_classes.items():
        if info['enabled']:
            print(f"  {info['name']}: confidence >= {info['min_confidence']}")
    print()
