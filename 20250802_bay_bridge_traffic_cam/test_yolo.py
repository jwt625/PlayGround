#!/usr/bin/env python3
"""
Test YOLO car detection with a real traffic image from the internet.
"""

import cv2
import requests
import numpy as np
from ultralytics import YOLO
from PIL import Image
import io

def download_test_image():
    """Download a test traffic image from the internet."""
    # Use a sample traffic image URL (this is a public domain traffic image)
    test_urls = [
        "https://images.unsplash.com/photo-1449824913935-59a10b8d2000?w=800&h=600&fit=crop",  # Traffic
        "https://images.unsplash.com/photo-1544620347-c4fd4a3d5957?w=800&h=600&fit=crop",  # Highway
        "https://images.unsplash.com/photo-1502920917128-1aa500764cbd?w=800&h=600&fit=crop",  # Cars
    ]
    
    for i, url in enumerate(test_urls):
        try:
            print(f"Trying to download test image {i+1}...")
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                # Convert to PIL Image then to numpy array
                image = Image.open(io.BytesIO(response.content))
                frame = np.array(image)
                
                # Convert RGB to BGR for OpenCV
                if len(frame.shape) == 3:
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                
                print(f"Successfully downloaded image: {frame.shape}")
                return frame
                
        except Exception as e:
            print(f"Failed to download from {url}: {e}")
    
    print("All download attempts failed, creating a simple test image...")
    # Create a simple test image if downloads fail
    test_frame = np.full((480, 640, 3), (100, 150, 200), dtype=np.uint8)
    cv2.putText(test_frame, "TEST IMAGE", (200, 240), 
               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    return test_frame

def detect_cars(frame, model):
    """Run YOLO detection on the frame and filter for cars."""
    if frame is None:
        return None, []
    
    # Run YOLO inference
    results = model(frame, verbose=False)
    
    # Filter for cars and related vehicles
    vehicle_classes = [2, 3, 5, 7]  # car, motorcycle, bus, truck in COCO dataset
    vehicle_names = {2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck'}
    
    detections = []
    
    for result in results:
        boxes = result.boxes
        if boxes is not None:
            for box in boxes:
                # Get class ID and confidence
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                
                # Check if it's a vehicle and confidence is high enough
                if class_id in vehicle_classes and confidence > 0.3:
                    # Get bounding box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    detections.append({
                        'bbox': (int(x1), int(y1), int(x2), int(y2)),
                        'confidence': confidence,
                        'class': vehicle_names.get(class_id, f'class_{class_id}')
                    })
    
    # Draw bounding boxes on the frame
    annotated_frame = frame.copy()
    for detection in detections:
        x1, y1, x2, y2 = detection['bbox']
        confidence = detection['confidence']
        vehicle_type = detection['class']
        
        # Choose color based on vehicle type
        colors = {'car': (0, 255, 0), 'truck': (255, 0, 0), 'bus': (0, 0, 255), 'motorcycle': (255, 255, 0)}
        color = colors.get(vehicle_type, (0, 255, 0))
        
        # Draw rectangle
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
        
        # Add label
        label = f"{vehicle_type}: {confidence:.2f}"
        cv2.putText(annotated_frame, label, (x1, y1-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    return annotated_frame, detections

def main():
    """Test YOLO with a real traffic image."""
    print("=== YOLO Car Detection Test ===")
    
    print("Loading YOLO model...")
    model = YOLO("yolov8n.pt")
    print("Model loaded successfully!")
    
    print("Downloading test traffic image...")
    frame = download_test_image()
    
    if frame is None:
        print("Failed to get test image!")
        return
    
    # Save the original test image
    cv2.imwrite("test_input.jpg", frame)
    print(f"Test image saved as test_input.jpg (shape: {frame.shape})")
    
    # Run detection
    print("Running YOLO detection...")
    annotated_frame, detections = detect_cars(frame, model)
    
    if annotated_frame is not None:
        print(f"\n=== DETECTION RESULTS ===")
        print(f"Total vehicles detected: {len(detections)}")
        
        # Group by vehicle type
        vehicle_counts = {}
        for detection in detections:
            vehicle_type = detection['class']
            vehicle_counts[vehicle_type] = vehicle_counts.get(vehicle_type, 0) + 1
        
        for vehicle_type, count in vehicle_counts.items():
            print(f"  {vehicle_type}: {count}")
        
        # Print detailed detection info
        print(f"\nDetailed results:")
        for i, detection in enumerate(detections):
            bbox = detection['bbox']
            conf = detection['confidence']
            vehicle_type = detection['class']
            print(f"  {i+1}. {vehicle_type}: bbox={bbox}, confidence={conf:.3f}")
        
        # Save the annotated result
        cv2.imwrite("test_detected.jpg", annotated_frame)
        print(f"\nAnnotated result saved as test_detected.jpg")
        
        print(f"\n=== FILES CREATED ===")
        print(f"  test_input.jpg - Original test image")
        print(f"  test_detected.jpg - Image with detection boxes")
        
    else:
        print("Detection failed!")

if __name__ == "__main__":
    main()
