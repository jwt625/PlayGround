#!/usr/bin/env python3
"""
Test script for object tracking functionality.
Creates synthetic moving objects to test the tracking system.
"""

import cv2
import numpy as np
from object_tracker import ObjectTracker, TrafficCounter
import time

def create_test_contours(frame_num, frame_shape):
    """Create synthetic contours that move across the frame."""
    height, width = frame_shape[:2]
    contours = []
    
    # Create 3 moving objects
    for i in range(3):
        # Object moves from left to right at different speeds and heights
        x = int(50 + frame_num * (2 + i * 0.5)) % (width - 100)
        y = int(100 + i * 80)
        size = 30 + i * 10
        
        # Create a rectangular contour
        rect_points = np.array([
            [x, y],
            [x + size, y],
            [x + size, y + size],
            [x, y + size]
        ], dtype=np.int32)
        
        contours.append(rect_points)
    
    return contours

def test_tracking():
    """Test the object tracking system with synthetic data."""
    print("Testing Object Tracking System")
    print("=" * 40)
    
    # Initialize tracker
    tracker = ObjectTracker(
        max_distance=50,
        max_disappeared=5,
        min_detection_frames=2
    )
    
    # Initialize traffic counter with a counting line
    counter = TrafficCounter()
    counter.add_counting_line(200, 50, 200, 300)  # Vertical line at x=200
    
    # Create test frames
    frame_shape = (400, 640, 3)  # height, width, channels
    
    for frame_num in range(50):
        print(f"\nFrame {frame_num + 1}:")
        
        # Create synthetic contours
        contours = create_test_contours(frame_num, frame_shape)
        
        # Update tracker
        tracked_objects = tracker.update(contours)
        
        # Update counter
        counter.update(tracked_objects)
        
        # Print tracking results
        stats = tracker.get_statistics()
        counts = counter.get_counts()
        
        print(f"  Detections: {len(contours)}")
        print(f"  Tracked Objects: {stats['active_objects']}")
        print(f"  Confirmed Objects: {stats['confirmed_objects']}")
        print(f"  Traffic Count: {counts['total']}")
        
        # Print object details
        for obj in tracked_objects:
            print(f"    Object {obj.id}: pos={obj.centroid}, speed={obj.get_speed_pixels_per_second():.1f}px/s, dir={obj.direction}")
        
        # Small delay to simulate real-time processing
        time.sleep(0.1)
    
    print("\n" + "=" * 40)
    print("Test completed successfully!")
    
    # Final statistics
    final_stats = tracker.get_statistics()
    final_counts = counter.get_counts()
    
    print(f"Final Statistics:")
    print(f"  Total objects created: {final_stats['total_created']}")
    print(f"  Total objects confirmed: {final_stats['total_confirmed']}")
    print(f"  Total traffic count: {final_counts['total']}")
    print(f"  Left: {final_counts['left']}, Right: {final_counts['right']}")

def test_visualization():
    """Test the tracking visualization with a simple demo."""
    print("\nTesting Tracking Visualization")
    print("=" * 40)
    print("This will create a visual demo. Press 'q' to quit.")
    
    # Initialize tracker
    tracker = ObjectTracker(max_distance=50, max_disappeared=10, min_detection_frames=2)
    
    frame_num = 0
    
    while True:
        # Create a test frame
        frame = np.zeros((400, 640, 3), dtype=np.uint8)
        
        # Create moving contours
        contours = create_test_contours(frame_num, frame.shape)
        
        # Update tracker
        tracked_objects = tracker.update(contours)
        
        # Draw tracked objects
        for obj in tracked_objects:
            x, y, w, h = obj.bbox
            
            # Draw bounding box
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
            
            # Draw object ID
            cv2.putText(frame, f"ID:{obj.id}", (x, y-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            
            # Draw trajectory
            if len(obj.positions) > 1:
                for i in range(1, len(obj.positions)):
                    cv2.line(frame, obj.positions[i-1], obj.positions[i], (0, 255, 255), 1)
            
            # Draw speed and direction
            speed = obj.get_speed_pixels_per_second()
            cv2.putText(frame, f"{speed:.1f}px/s {obj.direction or ''}", (x, y+h+15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Draw statistics
        stats = tracker.get_statistics()
        cv2.putText(frame, f"Frame: {frame_num} | Active: {stats['active_objects']} | Confirmed: {stats['confirmed_objects']}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Show frame
        cv2.imshow("Tracking Test", frame)
        
        # Handle keyboard input
        key = cv2.waitKey(100) & 0xFF
        if key == ord('q'):
            break
        
        frame_num += 1
        
        # Reset after 100 frames
        if frame_num > 100:
            frame_num = 0
            tracker.reset()
    
    cv2.destroyAllWindows()
    print("Visualization test completed!")

if __name__ == "__main__":
    # Run tests
    test_tracking()
    
    # Ask user if they want to see visualization
    response = input("\nWould you like to see the visual tracking demo? (y/n): ")
    if response.lower() in ['y', 'yes']:
        test_visualization()
    
    print("\nAll tests completed successfully!")
