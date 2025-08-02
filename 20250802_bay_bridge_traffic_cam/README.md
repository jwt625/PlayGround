# Bay Bridge Traffic Cam - Car Detection

A minimal Python script that grabs frames from an iPhone webcam and uses YOLO to detect cars.

## Setup

This project uses `uv` for dependency management:

```bash
# Dependencies are already installed via uv
uv run python car_detector.py --help
```

## Usage

### Test YOLO Detection
First, test that YOLO is working with a real traffic image:
```bash
uv run python test_yolo.py
```

### iPhone Webcam Detection
```bash
# Try to connect to iPhone webcam (auto-detects endpoints)
uv run python car_detector.py

# Use demo mode with synthetic image
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

## Files Generated

- `test_input.jpg` - Original test image (from test_yolo.py)
- `test_detected.jpg` - Test image with car detection boxes
- `original_frame.jpg` - Original frame from webcam
- `detected_cars.jpg` - Frame with detected cars highlighted
- `demo_input.jpg` - Synthetic demo image (in demo mode)

## Dependencies

- `opencv-python` - Image processing
- `ultralytics` - YOLO model
- `requests` - HTTP requests for webcam
- `pillow` - Image handling
- `numpy` - Array operations

## Model

Uses YOLOv8 nano model (`yolov8n.pt`) for fast inference. The model automatically downloads on first run.

Detects these vehicle types:
- Cars (class 2)
- Motorcycles (class 3)
- Buses (class 5)
- Trucks (class 7)

## Troubleshooting

1. **Can't connect to iPhone webcam**
   - Make sure iPhone and computer are on same WiFi network
   - Check the webcam app is running and showing the IP address
   - Try the demo mode first: `--demo`

2. **No cars detected**
   - Try the test script first: `python test_yolo.py`
   - Check if the image actually contains cars
   - Lower confidence threshold in the code if needed

3. **Slow detection**
   - YOLOv8n is already the fastest model
   - Consider reducing image resolution in the webcam app