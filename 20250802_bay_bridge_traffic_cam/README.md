# Bay Bridge Traffic Detection System

A comprehensive real-time traffic detection and monitoring system for Bay Bridge with **Prometheus + Grafana Cloud integration**:

## 🚀 Features

### Core Detection System
1. **Motion-Based Detection with Object Tracking** (Recommended) - Real-time motion tracking with persistent object IDs
2. **YOLO-Based Detection** (Legacy) - Object detection using YOLOv8 models

### 📊 Monitoring & Analytics (NEW)
- **Prometheus Metrics Collection** - Real-time traffic counting and system health
- **Grafana Cloud Integration** - Cloud-based dashboards and alerting
- **Traffic Flow Analytics** - Vehicles per minute by direction
- **System Health Monitoring** - Component status tracking
- **Performance Metrics** - FPS and processing time monitoring

## 🚀 Quick Start

### 1. Start Monitoring Infrastructure
```bash
# Start Prometheus and monitoring stack
./start.sh
```

### 2. Run Traffic Detection with Metrics
```bash
# Start the complete system with monitoring
python main.py

# Or run without metrics
python main.py --no-metrics
```

### 3. Access Monitoring
- **Metrics Endpoint**: http://localhost:9091/metrics
- **Prometheus UI**: http://localhost:9090
- **Grafana Cloud**: https://jwt625.grafana.net

## Setup

This project uses `uv` for dependency management:

```bash
# Install dependencies
uv sync

# Test the system
python test_metrics.py --validate
```

## 📊 Monitoring System

### Metrics Collection

The system automatically collects and exposes these metrics:

**Traffic Metrics:**
- `traffic_vehicles_total` - Total vehicles by direction (left/right)
- `traffic_flow_rate_per_minute` - Real-time traffic flow rate

**System Health:**
- `system_status` - Component health (webcam, detector, tracker)
- `motion_detector_fps` - Processing performance
- `tracked_objects_active` - Currently tracked objects

### Testing & Validation

```bash
# Validate metrics system
python test_metrics.py --validate

# Run performance benchmarks
python test_metrics.py --benchmark

# Start standalone metrics server for testing
python start_metrics_server.py --simulate
```

## Usage

### 🎯 Main System (Recommended)

**Complete traffic detection with monitoring:**

```bash
python main.py [OPTIONS]

Options:
  --debug          Enable debug output and visualization
  --no-metrics     Disable metrics collection
  --config FILE    Use custom configuration file
```

### Motion-Based Detection with Object Tracking

**Best for**: Bay Bridge side views, real-time monitoring, small/distant vehicles, traffic counting

**Features:**
- Real-time object tracking with persistent IDs
- Traffic counting with directional analysis
- ROI entry/exit counting for traffic flow analysis
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
- `x` - Reset ROI and traffic counters
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

## 🐳 Docker Infrastructure

The monitoring system uses Docker for Prometheus:

```bash
# Start monitoring stack
./start.sh

# Stop monitoring stack
docker-compose down

# View logs
docker logs prometheus
```

**Services:**
- **Prometheus** (port 9090) - Metrics collection and remote write to Grafana Cloud
- **Application** (port 9091) - Metrics HTTP server

## 📁 Project Structure

```
├── main.py                     # Main entry point with monitoring
├── prometheus_metrics.py       # Metrics collection engine
├── motion_detector.py          # Motion detection system
├── object_tracker.py          # Object tracking with metrics integration
├── start_metrics_server.py    # Standalone metrics server
├── test_metrics.py            # Testing and validation suite
├── docker-compose.yml         # Prometheus container
├── prometheus.yml.template    # Prometheus configuration template
├── generate-prometheus-config.sh # Auto-configuration script
├── start.sh                   # One-command startup
├── grafana-dashboard.json     # Ready-to-import dashboard
├── .env                       # Configuration (Grafana Cloud credentials)
└── docs/                      # Technical documentation
    ├── RFD-001-motion-detection.md
    ├── RFD-002-traffic-counting.md
    └── RFD-004-prometheus-grafana-monitoring.md
```

## Dependencies

### Core System
- `opencv-python>=4.12.0.88` - Image processing and motion detection
- `ultralytics>=8.3.173` - YOLO model (for legacy detection)
- `requests>=2.31.0` - HTTP requests for webcam streaming
- `pillow>=11.3.0` - Image handling
- `scipy>=1.11.0` - Scientific computing

### Monitoring System
- `prometheus_client>=0.22.1` - Metrics collection and HTTP server
- `python-dotenv>=1.1.1` - Environment configuration management

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

## ⚙️ Configuration

### Environment Variables (.env)

```bash
# Metrics Collection
METRICS_ENABLED=true
PROMETHEUS_HTTP_SERVER_PORT=9091

# Grafana Cloud Integration
PROMETHEUS_PUSH_GATEWAY_URL=https://prometheus-prod-XX-XXX.grafana.net/api/prom/push
PROMETHEUS_USERNAME=your_grafana_user_id
PROMETHEUS_API_KEY=your_grafana_api_token

# Application Settings
APP_NAME=bay-bridge-traffic-detector
GRAFANA_INSTANCE_URL=https://jwt625.grafana.net
```

### Grafana Cloud Setup

1. **Create Grafana Cloud Account** at https://grafana.com/
2. **Get Credentials** from your Grafana Cloud instance
3. **Update .env** with your credentials
4. **Import Dashboard** from `grafana-dashboard.json`

## Troubleshooting

### Monitoring System Issues

1. **Metrics endpoint not accessible**
   ```bash
   # Check if metrics server is running
   curl http://localhost:9091/metrics

   # Validate system
   python test_metrics.py --validate
   ```

2. **Prometheus not scraping metrics**
   ```bash
   # Check Prometheus targets
   curl http://localhost:9090/api/v1/targets

   # Regenerate configuration
   ./generate-prometheus-config.sh
   ```

3. **Grafana Cloud not receiving data**
   ```bash
   # Check Docker logs
   docker logs prometheus

   # Verify credentials in .env file
   ```

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

## 📚 Documentation

### Technical RFDs (Request for Discussion)
- **RFD-000**: YOLO-based detection system (legacy)
- **RFD-001**: Motion-based detection system (current)
- **RFD-002**: Traffic counting and direction detection
- **RFD-004**: Prometheus + Grafana Cloud monitoring system ⭐

### Key Metrics

**Performance Targets:**
- **Processing Speed**: 30+ FPS real-time detection
- **Detection Accuracy**: Vehicles as small as 20 pixels
- **Monitoring Overhead**: <1% CPU impact
- **Data Volume**: ~50 metrics per minute (free tier compliant)

**System Capabilities:**
- ✅ Real-time traffic counting by direction
- ✅ System health monitoring (webcam, detector, tracker)
- ✅ Cloud-based dashboards and alerting
- ✅ Historical traffic pattern analysis
- ✅ Performance optimization insights

## 🎯 Production Deployment

### Quick Deployment Checklist

1. **Configure Grafana Cloud credentials** in `.env`
2. **Start monitoring infrastructure**: `./start.sh`
3. **Run traffic detection**: `python main.py`
4. **Validate metrics**: `python test_metrics.py --validate`
5. **Import dashboard** to Grafana Cloud from `grafana-dashboard.json`

### Monitoring Endpoints

- **Application Metrics**: http://localhost:9091/metrics
- **Prometheus UI**: http://localhost:9090
- **Grafana Cloud**: https://jwt625.grafana.net

See `docs/` folder for detailed technical documentation and implementation guides.