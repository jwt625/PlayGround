# Bay Bridge Traffic Cam - Traffic Detection with Object Tracking

A real-time traffic detection and tracking system for Bay Bridge monitoring with advanced object tracking capabilities:

1. **Motion-Based Detection with Object Tracking** (Recommended) - Real-time motion tracking with persistent object IDs
2. **YOLO-Based Detection** (Legacy) - Object detection using YOLOv8 models

## Setup

This project uses `uv` for dependency management:

```bash
# Dependencies are already installed via uv
uv run python car_detector.py --help
```

## Usage

### Motion-Based Detection with Object Tracking (Recommended)

**Best for**: Bay Bridge side views, real-time monitoring, small/distant vehicles, traffic counting

**Features:**
- Real-time object tracking with persistent IDs
- Traffic counting with directional analysis
- Speed estimation in pixels per second
- Trajectory visualization
- Interactive counting line setup

```bash
# Run motion-based traffic detection with tracking
python motion_detector.py
```

**Controls:**
- `q` - Quit
- `r` - Reset/Set ROI (Region of Interest)
- `s` - Save current frame
- `c` - Cycle through detection presets
- `1-4` - Switch to specific preset
- `t` - Toggle object tracking on/off
- `l` - Set counting line for traffic counting
- `SPACE` - Pause/Resume

**Test the tracking system:**
```bash
# Run tracking tests
python test_tracking.py
```

**Features:**
- **30+ FPS** real-time processing
- **Detects vehicles as small as 20 pixels**
- **Handles occlusion** from bridge infrastructure
- **Interactive ROI selection** for bridge deck area
- **Debug visualization** with color-coded detection analysis
- **Multiple detection presets** for different scenarios

**Controls:**
- `q` - Quit
- `r` - Reset/Set ROI (Region of Interest)
- `s` - Save current frame
- `c` - Cycle through detection presets
- `1-4` - Switch to specific preset

### YOLO-Based Detection (Legacy)

**Best for**: Clear, unobstructed vehicle views

```bash
# Test YOLO detection with sample image
uv run python test_yolo.py

# Run YOLO-based detection
uv run python car_detector.py

# Use demo mode
uv run python car_detector.py --demo

# Specify custom webcam URL
uv run python car_detector.py --url http://192.168.1.100:8080
```

## iPhone Webcam Setup

Popular iPhone webcam apps and their typical endpoints:

1. **DroidCam** - Usually uses port 4747
   - Try: `http://PHONE_IP:4747/video`
   - Try: `http://PHONE_IP:4747/mjpegfeed`

2. **EpocCam** - Usually uses port 8080
   - Try: `http://PHONE_IP:8080/stream`

3. **iVCam** - Various ports
   - Check the app for the specific URL

4. **IP Webcam** - Usually port 8080
   - Try: `http://PHONE_IP:8080/video`

The script automatically tries common endpoints when you provide the base URL.

## Configuration

### Motion Detection Settings

Edit `motion_config.py` to adjust detection parameters:

```python
# Object size filters (in pixels)
"min_contour_area": 20,     # Minimum car size (lower = detect smaller cars)
"max_contour_area": 8000,   # Maximum car size

# Shape filters
"min_aspect_ratio": 0.1,    # Allow thin distant vehicles
"min_extent": 0.2,          # Allow partially occluded vehicles

# Performance settings
"target_fps": 0,            # 0 = unlimited, 30 = cap at 30fps
```

### Detection Presets

Four built-in presets optimized for different scenarios:

1. **`high_sensitivity`** (Default) - Best for distant traffic
2. **`distant_traffic`** - Maximum sensitivity for very small cars
3. **`balanced`** - Good noise/detection balance
4. **`low_noise`** - Minimal false positives

## Files Generated

### Motion Detection
- `motion_detection_TIMESTAMP.jpg` - Annotated frames with detections
- `motion_outputs/` - Directory for saved detection results

### YOLO Detection
- `test_input.jpg` - Original test image (from test_yolo.py)
- `test_detected.jpg` - Test image with car detection boxes
- `original_frame.jpg` - Original frame from webcam
- `detected_cars.jpg` - Frame with detected cars highlighted
- `outputs/TIMESTAMP/` - Timestamped detection results

## Dependencies

- `opencv-python` - Image processing and motion detection
- `ultralytics` - YOLO model (for legacy detection)
- `requests` - HTTP requests for webcam streaming
- `pillow` - Image handling
- `numpy` - Array operations

## Detection Methods

### Motion-Based Detection
- **Algorithm**: MOG2 Background Subtraction
- **Performance**: 30+ FPS real-time processing
- **Strengths**: Handles occlusion, small objects, no training required
- **Best for**: Fixed-camera traffic monitoring, bridge side views

### YOLO-Based Detection
- **Model**: YOLOv8 nano (`yolov8n.pt`) - downloads automatically
- **Performance**: 2-3 FPS processing
- **Detects**: Cars, motorcycles, buses, trucks, pedestrians
- **Best for**: Clear, unobstructed vehicle identification

## Troubleshooting

### Motion Detection Issues

1. **Cars not detected (showing as RED boxes in debug window)**
   - Lower `min_contour_area` in `motion_config.py` (try 15 or 10)
   - Switch to `distant_traffic` preset (press '4' key)
   - Ensure ROI covers the traffic area properly

2. **Too many false positives**
   - Switch to `low_noise` preset (press '3' key)
   - Increase `min_contour_area` and `min_extent` values
   - Adjust ROI to exclude non-traffic areas

3. **Cars detected but filtered out (ORANGE boxes)**
   - Lower `min_aspect_ratio` for thin distant cars
   - Lower `min_extent` for partially occluded cars
   - Check debug window for specific AR/EX values

### General Issues

1. **Can't connect to iPhone webcam**
   - Make sure iPhone and computer are on same WiFi network
   - Check the webcam app is running and showing the IP address
   - Try demo mode first to test the system

2. **Low frame rate**
   - Motion detector should run at 30+ FPS
   - YOLO detector runs at 2-3 FPS (expected)
   - Check `target_fps` setting in `motion_config.py`

3. **YOLO detection issues**
   - Try the test script first: `python test_yolo.py`
   - YOLO struggles with bridge side views (use motion detection instead)

## Documentation

- **RFD-000**: YOLO-based detection system (legacy)
- **RFD-001**: Motion-based detection system (current)

See `docs/` folder for detailed technical documentation.