#!/usr/bin/env python3
"""
Minimal car detection script using iPhone webcam and YOLO.
Grabs frames from iPhone webcam and detects cars using YOLOv8.
"""

import cv2
import requests
import numpy as np
from ultralytics import YOLO
from PIL import Image
import io
import time
import argparse
import os
from datetime import datetime
import config



def optimize_resolution(cap):
    """Set the highest resolution available for the video capture."""
    if not config.USE_HIGHEST_RESOLUTION:
        return

    # Common high resolutions to try (width, height)
    high_resolutions = [
        (3840, 2160),  # 4K
        (2560, 1440),  # 1440p
        (1920, 1080),  # 1080p
        (1280, 720),   # 720p
        (960, 540),    # 540p
        (640, 480),    # 480p
    ]

    print("Attempting to set highest resolution...")

    for width, height in high_resolutions:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        # Check what resolution was actually set
        actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if actual_width >= config.MIN_RESOLUTION[0] and actual_height >= config.MIN_RESOLUTION[1]:
            print(f"Set resolution to: {actual_width}x{actual_height}")
            return

    print("Using default resolution")

def process_frame_resolution(frame):
    """Process frame resolution according to config settings."""
    if frame is None:
        return None

    original_shape = frame.shape
    height, width = original_shape[:2]

    # Force specific resolution if configured
    if config.FORCE_RESOLUTION:
        target_width, target_height = config.FORCE_RESOLUTION
        frame = cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_CUBIC)
        print(f"Forced resolution: {original_shape[:2]} -> {frame.shape[:2]}")

    # Handle aspect ratio preference
    elif config.PREFERRED_ASPECT_RATIO == "landscape" and height > width:
        # Rotate portrait to landscape
        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        print(f"Rotated to landscape: {original_shape[:2]} -> {frame.shape[:2]}")
    elif config.PREFERRED_ASPECT_RATIO == "portrait" and width > height:
        # Rotate landscape to portrait
        frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        print(f"Rotated to portrait: {original_shape[:2]} -> {frame.shape[:2]}")

    # Apply small object enhancement if enabled
    if config.ENHANCE_SMALL_OBJECTS:
        height, width = frame.shape[:2]
        new_width = int(width * config.SMALL_OBJECT_ENHANCEMENT_FACTOR)
        new_height = int(height * config.SMALL_OBJECT_ENHANCEMENT_FACTOR)
        frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
        print(f"Enhanced for small objects: {width}x{height} -> {new_width}x{new_height}")

    return frame

def grab_frame_from_webcam(url):
    """Grab a single frame from the iPhone webcam stream with highest resolution."""
    try:
        print(f"Grabbing frame from: {url}")

        # Use OpenCV to capture from the video stream directly
        cap = cv2.VideoCapture(url)

        if not cap.isOpened():
            print("Failed to open video stream with OpenCV, trying requests...")
            # Fallback to requests method
            response = requests.get(url, stream=True, timeout=config.WEBCAM_TIMEOUT)
            if response.status_code == 200:
                bytes_data = b''
                for chunk in response.iter_content(chunk_size=1024):
                    bytes_data += chunk
                    # Look for JPEG markers
                    start = bytes_data.find(b'\xff\xd8')  # JPEG start
                    end = bytes_data.find(b'\xff\xd9')    # JPEG end

                    if start != -1 and end != -1 and end > start:
                        # Extract the JPEG frame
                        jpeg_data = bytes_data[start:end+2]

                        # Convert to PIL Image then to numpy array
                        image = Image.open(io.BytesIO(jpeg_data))
                        frame = np.array(image)

                        # Convert RGB to BGR for OpenCV
                        if len(frame.shape) == 3:
                            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

                        # Process resolution
                        frame = process_frame_resolution(frame)
                        return frame

                    # Keep only recent data to avoid memory issues
                    if len(bytes_data) > 100000:
                        bytes_data = bytes_data[-50000:]
            return None

        # Optimize resolution for OpenCV capture
        optimize_resolution(cap)

        # Read a frame using OpenCV
        ret, frame = cap.read()
        cap.release()

        if ret:
            print(f"Successfully captured frame: {frame.shape}")

            # Process resolution
            frame = process_frame_resolution(frame)
            return frame
        else:
            print("Failed to read frame from video stream")
            return None

    except Exception as e:
        print(f"Error grabbing frame: {e}")
        return None

def create_demo_image():
    """Create a demo image with cars for testing when webcam is not available."""
    print("Creating demo image with cars...")

    # Create a simple demo image using config settings
    width, height = config.DEMO_IMAGE_SIZE
    demo_frame = np.full((height, width, 3), config.DEMO_BACKGROUND_COLOR, dtype=np.uint8)

    # Draw some rectangles to simulate cars
    cars = [
        (100, 200, 180, 260),  # Car 1
        (300, 180, 400, 240),  # Car 2
        (450, 220, 550, 280),  # Car 3
    ]

    for i, (x1, y1, x2, y2) in enumerate(cars):
        # Draw car body (rectangle)
        cv2.rectangle(demo_frame, (x1, y1), (x2, y2), (0, 0, 200), -1)
        cv2.rectangle(demo_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

        # Draw wheels
        wheel_radius = 15
        cv2.circle(demo_frame, (x1+20, y2), wheel_radius, (50, 50, 50), -1)
        cv2.circle(demo_frame, (x2-20, y2), wheel_radius, (50, 50, 50), -1)

        # Add car label
        cv2.putText(demo_frame, f"Car {i+1}", (x1, y1-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Add some background elements
    cv2.putText(demo_frame, "DEMO: Bay Bridge Traffic", (50, 50),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    return demo_frame

def detect_objects(frame, model):
    """Run YOLO detection on the frame and filter for configured object classes."""
    if frame is None:
        return None, []

    # Get active detection classes from config
    target_classes = config.get_active_confidence_thresholds()

    # Run YOLO inference
    results = model(frame, verbose=config.VERBOSE_YOLO)

    detections = []
    detection_count = 0

    for result in results:
        boxes = result.boxes
        if boxes is not None:
            for box in boxes:
                if detection_count >= config.MAX_DETECTIONS_PER_FRAME:
                    break

                # Get class ID and confidence
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])

                # Check if it's a target class with sufficient confidence
                if class_id in target_classes:
                    class_info = target_classes[class_id]
                    if class_info['enabled'] and confidence > class_info['min_confidence']:
                        # Get bounding box coordinates
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        detections.append({
                            'bbox': (int(x1), int(y1), int(x2), int(y2)),
                            'confidence': confidence,
                            'class_id': class_id,
                            'class_name': class_info['name'],
                            'color': class_info['color']
                        })
                        detection_count += 1

    # Draw bounding boxes on the frame
    annotated_frame = frame.copy()
    for detection in detections:
        x1, y1, x2, y2 = detection['bbox']
        confidence = detection['confidence']
        class_name = detection['class_name']
        color = detection['color']

        # Draw rectangle
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, config.BOUNDING_BOX_THICKNESS)

        # Add label with background for better visibility
        label = f"{class_name}: {confidence:.2f}"
        label_size = cv2.getTextSize(label, config.LABEL_FONT, config.LABEL_FONT_SCALE, config.LABEL_FONT_THICKNESS)[0]
        cv2.rectangle(annotated_frame, (x1, y1-label_size[1]-10),
                     (x1+label_size[0], y1), color, -1)
        cv2.putText(annotated_frame, label, (x1, y1-5),
                   config.LABEL_FONT, config.LABEL_FONT_SCALE, config.LABEL_TEXT_COLOR, config.LABEL_FONT_THICKNESS)

    return annotated_frame, detections

def organize_outputs():
    """Create output directory structure."""
    timestamp = datetime.now().strftime(config.TIMESTAMP_FORMAT)
    output_dir = f"{config.OUTPUT_BASE_DIR}/{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

def main():
    """Main function to run object detection."""
    parser = argparse.ArgumentParser(description='Car and pedestrian detection using iPhone webcam and YOLO')
    parser.add_argument('--demo', action='store_true', help='Use demo mode with synthetic image')
    parser.add_argument('--url', default=f"{config.WEBCAM_BASE_URL}{config.WEBCAM_DEFAULT_ENDPOINT}", help='Webcam URL')
    parser.add_argument('--config-summary', action='store_true', help='Print configuration summary and exit')
    args = parser.parse_args()

    if args.config_summary:
        config.print_config_summary()
        return

    print("=== Bay Bridge Traffic Monitor ===")
    config.print_config_summary()

    print("Loading YOLO model...")
    model = YOLO(config.MODEL_NAME)
    print(f"Model loaded: {config.MODEL_NAME}")

    # Create output directory
    output_dir = organize_outputs()
    print(f"Output directory: {output_dir}")

    frame = None

    if args.demo:
        print("Running in demo mode...")
        frame = create_demo_image()
        demo_path = os.path.join(output_dir, "demo_input.jpg")
        cv2.imwrite(demo_path, frame)
        print(f"Demo frame created and saved as {demo_path}")
    else:
        # Grab a frame from the webcam
        print("Capturing frame from iPhone webcam...")
        frame = grab_frame_from_webcam(args.url)

        if frame is None:
            print("Failed to grab frame from webcam! Falling back to demo mode...")
            frame = create_demo_image()
            demo_path = os.path.join(output_dir, "demo_input.jpg")
            cv2.imwrite(demo_path, frame)
            print(f"Demo frame created and saved as {demo_path}")

    if frame is None:
        print("No frame available for processing!")
        return

    print(f"Frame shape: {frame.shape}")

    # Detect objects
    print("Running object detection (cars, pedestrians, vehicles)...")
    annotated_frame, detections = detect_objects(frame, model)

    if annotated_frame is not None:
        # Count detections by type
        detection_counts = {}
        for detection in detections:
            class_name = detection['class_name']
            detection_counts[class_name] = detection_counts.get(class_name, 0) + 1

        total_objects = len(detections)
        print(f"\n=== DETECTION RESULTS ===")
        print(f"Total objects detected: {total_objects}")
        for obj_type, count in detection_counts.items():
            print(f"  {obj_type}: {count}")

        # Print detailed detection info
        print(f"\nDetailed results:")
        for i, detection in enumerate(detections):
            bbox = detection['bbox']
            conf = detection['confidence']
            class_name = detection['class_name']
            print(f"  {i+1}. {class_name}: bbox={bbox}, confidence={conf:.3f}")

        # Save files with organized naming
        files_saved = []

        if config.SAVE_ORIGINAL_FRAME:
            original_path = os.path.join(output_dir, "original_frame.jpg")
            cv2.imwrite(original_path, frame)
            files_saved.append(f"Original frame: {original_path}")

        if config.SAVE_ANNOTATED_FRAME:
            annotated_path = os.path.join(output_dir, "detected_objects.jpg")
            cv2.imwrite(annotated_path, annotated_frame)
            files_saved.append(f"Annotated frame: {annotated_path}")

        print(f"\n=== FILES SAVED ===")
        for file_info in files_saved:
            print(file_info)

        print(f"\n=== SUMMARY ===")
        for obj_type, count in detection_counts.items():
            print(f"{obj_type}: {count}")
        if args.demo:
            print("Note: This was a demo run with synthetic data")

    else:
        print("Detection failed!")

if __name__ == "__main__":
    main()
