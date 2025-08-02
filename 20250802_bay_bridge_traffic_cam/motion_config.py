"""
Configuration file for Motion-Based Traffic Detection
Optimized for Bay Bridge side view with small/distant vehicles
"""

# Webcam Configuration
WEBCAM_URL = "http://192.168.12.6:4747/video"
WEBCAM_TIMEOUT = 5  # seconds

# Background Subtraction Parameters
BACKGROUND_SUBTRACTOR_CONFIG = {
    "detectShadows": True,      # Helps reduce shadow false positives
    "varThreshold": 50,         # Lower = more sensitive to motion (16-50 good range)
    "history": 500,             # Frames to build background model (300-1000)
    "learningRate": -1,         # Auto learning rate (-1 = automatic)
    "shadowValue": 0,           # Shadow pixel value
    "shadowThreshold": 0.5      # Shadow detection threshold
}

# Motion Detection Thresholds
MOTION_DETECTION = {
    # Object size filters (in pixels)
    "min_contour_area": 20,     # LOWERED from 100 - detects smaller cars
    "max_contour_area": 8000,   # INCREASED from 5000 - allows larger vehicles
    
    # Morphological operations (noise reduction)
    "morph_kernel_size": (3, 3),  # Kernel for opening/closing operations
    "morph_iterations": 1,        # Number of morphological iterations
    
    # Contour filtering
    "min_aspect_ratio": 0.1,     # Minimum width/height ratio (filters thin lines)
    "max_aspect_ratio": 5.0,     # Maximum width/height ratio
    "min_extent": 0.2,           # Minimum contour area / bounding box area
}

# Advanced Detection Settings
ADVANCED_DETECTION = {
    # Multi-scale detection for very small objects
    "enable_multiscale": True,          # Process at multiple scales
    "scale_factors": [1.0, 1.5, 2.0],  # Image scaling for small object detection
    
    # Temporal filtering (reduce noise across frames)
    "enable_temporal_filter": True,     # Track objects across frames
    "min_detection_frames": 2,          # Object must appear in N consecutive frames
    
    # Motion direction filtering
    "enable_direction_filter": False,   # Filter by movement direction
    "valid_directions": ["left", "right", "both"],  # Expected traffic directions
}

# Visual Display Settings
DISPLAY_CONFIG = {
    # Colors (BGR format)
    "roi_color": (255, 255, 0),        # Yellow for ROI rectangle
    "detection_color": (0, 255, 0),    # Green for detection boxes
    "contour_color": (0, 255, 255),    # Yellow for contours
    "text_color": (255, 255, 255),     # White for text
    
    # Line thickness
    "roi_thickness": 2,
    "detection_thickness": 2,
    "contour_thickness": 1,
    
    # Font settings
    "font": 1,  # cv2.FONT_HERSHEY_SIMPLEX
    "font_scale": 0.7,
    "font_thickness": 2,
    "small_font_scale": 0.5,
    "small_font_thickness": 1,
    
    # Display windows
    "show_motion_mask": True,           # Show background subtraction mask
    "show_debug_info": True,            # Show detection statistics
    "window_resize": True,              # Allow window resizing
}

# Performance Settings
PERFORMANCE = {
    "target_fps": 0,                    # Target frame rate (0 = unlimited, 30 = 30fps cap)
    "frame_skip": 1,                    # Process every Nth frame (1 = all frames)
    "max_processing_time": 0.1,         # Max time per frame (seconds)
    "enable_gpu_acceleration": False,   # Use GPU if available (experimental)
    "show_fps": True,                   # Display FPS counter
}

# ROI (Region of Interest) Settings
ROI_CONFIG = {
    "auto_detect_bridge": False,        # Attempt automatic bridge detection
    "default_roi": None,                # Default ROI (x, y, w, h) or None for manual
    "roi_margin": 10,                   # Pixels to expand ROI for edge cases
    "save_roi": True,                   # Save ROI selection for next run
    "roi_file": "saved_roi.txt",        # File to save ROI coordinates
}

# Traffic Counting Settings
COUNTING_CONFIG = {
    "enable_counting": True,            # Enable traffic counting
    "counting_lines": [],               # List of counting line coordinates [(x1,y1,x2,y2), ...]
    "count_direction": "both",          # "left", "right", or "both"
    "min_crossing_distance": 20,       # Minimum pixels to count as crossing
    "count_timeout": 2.0,               # Seconds before same object can be counted again
}

# Logging and Output
OUTPUT_CONFIG = {
    "save_detections": True,            # Save detection results
    "output_dir": "motion_outputs",     # Directory for saved files
    "save_annotated_frames": True,      # Save frames with annotations
    "save_motion_mask": False,          # Save motion mask images
    "log_detections": True,             # Log detection events
    "log_file": "motion_detection.log", # Log file name
    "timestamp_format": "%Y%m%d_%H%M%S", # Timestamp format for files
}

# Demo Mode Settings (when webcam unavailable)
DEMO_CONFIG = {
    "demo_frame_size": (640, 480),      # Demo frame dimensions
    "demo_background_color": (50, 50, 50),  # Background color (BGR)
    "demo_object_count": 3,             # Number of moving demo objects
    "demo_speed_range": (1, 3),         # Speed multiplier range for demo objects
    "demo_object_colors": [             # Colors for demo objects
        (0, 255, 0),    # Green
        (255, 0, 0),    # Blue  
        (0, 0, 255),    # Red
    ],
}

# Presets for different scenarios
DETECTION_PRESETS = {
    "high_sensitivity": {
        "min_contour_area": 25,         # Very small objects
        "varThreshold": 30,             # More sensitive background subtraction
        "enable_multiscale": True,
    },
    "balanced": {
        "min_contour_area": 50,         # Current default
        "varThreshold": 50,
        "enable_multiscale": True,
    },
    "low_noise": {
        "min_contour_area": 100,        # Larger objects only
        "varThreshold": 70,             # Less sensitive (reduces noise)
        "enable_multiscale": False,
    },
    "distant_traffic": {
        "min_contour_area": 20,         # Very small for distant cars
        "varThreshold": 25,             # Very sensitive
        "enable_multiscale": True,
        "scale_factors": [1.0, 1.5, 2.0, 2.5],  # More scales
    }
}

# Active preset - change this to switch detection modes
ACTIVE_PRESET = "high_sensitivity"  # Options: "high_sensitivity", "balanced", "low_noise", "distant_traffic"

def apply_preset(preset_name):
    """Apply a detection preset to the current configuration."""
    global MOTION_DETECTION, BACKGROUND_SUBTRACTOR_CONFIG, ADVANCED_DETECTION
    
    if preset_name not in DETECTION_PRESETS:
        print(f"Warning: Preset '{preset_name}' not found. Using 'balanced'.")
        preset_name = "balanced"
    
    preset = DETECTION_PRESETS[preset_name]
    
    # Apply preset values
    if "min_contour_area" in preset:
        MOTION_DETECTION["min_contour_area"] = preset["min_contour_area"]
    if "varThreshold" in preset:
        BACKGROUND_SUBTRACTOR_CONFIG["varThreshold"] = preset["varThreshold"]
    if "enable_multiscale" in preset:
        ADVANCED_DETECTION["enable_multiscale"] = preset["enable_multiscale"]
    if "scale_factors" in preset:
        ADVANCED_DETECTION["scale_factors"] = preset["scale_factors"]
    
    print(f"Applied preset: {preset_name}")
    print(f"  Min contour area: {MOTION_DETECTION['min_contour_area']}")
    print(f"  Variance threshold: {BACKGROUND_SUBTRACTOR_CONFIG['varThreshold']}")
    print(f"  Multiscale enabled: {ADVANCED_DETECTION['enable_multiscale']}")

def get_current_config():
    """Get current configuration summary."""
    config_summary = {
        "preset": ACTIVE_PRESET,
        "min_object_size": MOTION_DETECTION["min_contour_area"],
        "max_object_size": MOTION_DETECTION["max_contour_area"],
        "sensitivity": BACKGROUND_SUBTRACTOR_CONFIG["varThreshold"],
        "multiscale": ADVANCED_DETECTION["enable_multiscale"],
    }
    return config_summary

def print_config_summary():
    """Print current configuration."""
    print("=== MOTION DETECTION CONFIGURATION ===")
    config = get_current_config()
    print(f"Active Preset: {config['preset']}")
    print(f"Object Size Range: {config['min_object_size']} - {config['max_object_size']} pixels")
    print(f"Motion Sensitivity: {config['sensitivity']} (lower = more sensitive)")
    print(f"Multi-scale Detection: {config['multiscale']}")
    print(f"Webcam URL: {WEBCAM_URL}")
    print("=" * 40)

# Apply the active preset on import
apply_preset(ACTIVE_PRESET)
