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

# Configuration
WEBCAM_BASE_URL = "http://192.168.12.6:4747"
MODEL_NAME = "yolov8n.pt"  # Nano model for speed

# Common iPhone webcam app endpoints to try
COMMON_ENDPOINTS = [
    "/video",
    "/video.mjpg",
    "/mjpeg",
    "/stream",
    "/cam.mjpg",
    "/video.cgi",
    "/videostream.cgi",
    "/axis-cgi/mjpg/video.cgi"
]

def try_webcam_endpoints(base_url):
    """Try different common endpoints to find the video stream."""
    print(f"Trying to find video stream at {base_url}...")

    for endpoint in COMMON_ENDPOINTS:
        url = base_url + endpoint
        print(f"Trying: {url}")

        try:
            response = requests.head(url, timeout=3)
            print(f"  Status: {response.status_code}")

            if response.status_code == 200:
                content_type = response.headers.get('content-type', '')
                print(f"  Content-Type: {content_type}")

                if 'mjpeg' in content_type.lower() or 'multipart' in content_type.lower():
                    print(f"  ✓ Found MJPEG stream at: {url}")
                    return url
                elif 'jpeg' in content_type.lower() or 'image' in content_type.lower():
                    print(f"  ✓ Found image endpoint at: {url}")
                    return url

        except Exception as e:
            print(f"  Error: {e}")

    print("No working video endpoints found!")
    return None

def grab_frame_from_webcam(url):
    """Grab a single frame from the iPhone webcam stream."""
    try:
        print(f"Grabbing frame from: {url}")

        # Use OpenCV to capture from the video stream directly
        cap = cv2.VideoCapture(url)

        if not cap.isOpened():
            print("Failed to open video stream with OpenCV, trying requests...")
            # Fallback to requests method
            response = requests.get(url, stream=True, timeout=10)
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

                        return frame

                    # Keep only recent data to avoid memory issues
                    if len(bytes_data) > 100000:
                        bytes_data = bytes_data[-50000:]
            return None

        # Read a frame using OpenCV
        ret, frame = cap.read()
        cap.release()

        if ret:
            print(f"Successfully captured frame: {frame.shape}")
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

    # Create a simple demo image (640x480, blue background)
    demo_frame = np.full((480, 640, 3), (100, 50, 0), dtype=np.uint8)

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
    """Run YOLO detection on the frame and filter for cars and pedestrians."""
    if frame is None:
        return None, []

    # Run YOLO inference
    results = model(frame, verbose=False)

    # Define object classes we're interested in
    target_classes = {
        0: {'name': 'person', 'color': (255, 0, 0), 'min_conf': 0.4},      # Blue for persons
        2: {'name': 'car', 'color': (0, 255, 0), 'min_conf': 0.5},         # Green for cars
        3: {'name': 'motorcycle', 'color': (0, 255, 255), 'min_conf': 0.5}, # Yellow for motorcycles
        5: {'name': 'bus', 'color': (255, 0, 255), 'min_conf': 0.5},       # Magenta for buses
        7: {'name': 'truck', 'color': (0, 128, 255), 'min_conf': 0.5}      # Orange for trucks
    }

    detections = []

    for result in results:
        boxes = result.boxes
        if boxes is not None:
            for box in boxes:
                # Get class ID and confidence
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])

                # Check if it's a target class with sufficient confidence
                if class_id in target_classes:
                    class_info = target_classes[class_id]
                    if confidence > class_info['min_conf']:
                        # Get bounding box coordinates
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        detections.append({
                            'bbox': (int(x1), int(y1), int(x2), int(y2)),
                            'confidence': confidence,
                            'class_id': class_id,
                            'class_name': class_info['name'],
                            'color': class_info['color']
                        })

    # Draw bounding boxes on the frame
    annotated_frame = frame.copy()
    for detection in detections:
        x1, y1, x2, y2 = detection['bbox']
        confidence = detection['confidence']
        class_name = detection['class_name']
        color = detection['color']

        # Draw rectangle
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)

        # Add label with background for better visibility
        label = f"{class_name}: {confidence:.2f}"
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
        cv2.rectangle(annotated_frame, (x1, y1-label_size[1]-10),
                     (x1+label_size[0], y1), color, -1)
        cv2.putText(annotated_frame, label, (x1, y1-5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    return annotated_frame, detections

def main():
    """Main function to run car detection."""
    parser = argparse.ArgumentParser(description='Car detection using iPhone webcam and YOLO')
    parser.add_argument('--demo', action='store_true', help='Use demo mode with synthetic image')
    parser.add_argument('--url', default="http://192.168.12.6:4747/video", help='Webcam URL')
    args = parser.parse_args()

    print("Loading YOLO model...")
    model = YOLO(MODEL_NAME)
    print(f"Model loaded: {MODEL_NAME}")

    frame = None

    if args.demo:
        print("Running in demo mode...")
        frame = create_demo_image()
        cv2.imwrite("demo_input.jpg", frame)
        print("Demo frame created and saved as demo_input.jpg")
    else:
        # Grab a frame from the webcam
        frame = grab_frame_from_webcam(args.url)

        if frame is None:
            print("Failed to grab frame from webcam! Falling back to demo mode...")
            frame = create_demo_image()
            cv2.imwrite("demo_input.jpg", frame)
            print("Demo frame created and saved as demo_input.jpg")

    if frame is None:
        print("No frame available for processing!")
        return

    print(f"Frame shape: {frame.shape}")

    # Detect cars
    print("Running car detection...")
    annotated_frame, detections = detect_cars(frame, model)

    if annotated_frame is not None:
        print(f"Found {len(detections)} cars!")

        # Print detection details
        for i, detection in enumerate(detections):
            bbox = detection['bbox']
            conf = detection['confidence']
            print(f"Car {i+1}: bbox={bbox}, confidence={conf:.3f}")

        # Save the annotated frame
        output_path = "detected_cars.jpg"
        cv2.imwrite(output_path, annotated_frame)
        print(f"Annotated frame saved to: {output_path}")

        # Also save the original frame for comparison
        cv2.imwrite("original_frame.jpg", frame)
        print("Original frame saved to: original_frame.jpg")

        print("\n=== SUMMARY ===")
        print(f"Cars detected: {len(detections)}")
        print(f"Output files: {output_path}, original_frame.jpg")
        if args.demo:
            print("Note: This was a demo run with synthetic data")

    else:
        print("Detection failed!")

if __name__ == "__main__":
    main()
