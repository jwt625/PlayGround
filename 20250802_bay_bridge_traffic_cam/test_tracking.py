#!/usr/bin/env python3
"""
Test script for object tracking functionality.
Creates synthetic moving objects to test the tracking system.
"""

import cv2
import numpy as np
from object_tracker import ObjectTracker, TrafficCounter, ROITracker
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

def test_roi_entry_exit():
    """Test ROI entry/exit counting functionality."""
    print("\nTesting ROI Entry/Exit Counting")
    print("=" * 40)

    # Initialize tracker and ROI tracker
    tracker = ObjectTracker(max_distance=50, max_disappeared=5, min_detection_frames=2)
    roi_tracker = ROITracker()

    # Set up ROI (center area of the frame)
    roi_rect = (200, 100, 240, 200)  # x, y, w, h - center area
    roi_tracker.set_roi(roi_rect)

    print(f"ROI set to: {roi_rect}")
    print("Testing objects moving through ROI...")

    frame_shape = (400, 640, 3)

    for frame_num in range(100):
        # Create objects that move through the ROI
        contours = []

        # Object 1: Moves from left to right, enters and exits ROI
        x1 = int(50 + frame_num * 6) % 600
        y1 = 150
        rect1 = np.array([[x1, y1], [x1+30, y1], [x1+30, y1+30], [x1, y1+30]], dtype=np.int32)
        contours.append(rect1)

        # Object 2: Moves from top to bottom, enters and exits ROI
        x2 = 300
        y2 = int(20 + frame_num * 4) % 350
        rect2 = np.array([[x2, y2], [x2+25, y2], [x2+25, y2+25], [x2, y2+25]], dtype=np.int32)
        contours.append(rect2)

        # Update tracker
        tracked_objects = tracker.update(contours)

        # Update ROI tracker
        roi_tracker.update(tracked_objects, frame_num)

        # Print status every 10 frames
        if frame_num % 10 == 0:
            roi_counts = roi_tracker.get_counts()
            print(f"Frame {frame_num:3d}: Entries: {roi_counts['entries']}, "
                  f"Exits: {roi_counts['exits']}, Current in ROI: {roi_counts['current_in_roi']}")

        time.sleep(0.05)  # Small delay

    # Final results
    final_counts = roi_tracker.get_counts()
    recent_stats = roi_tracker.get_recent_events(30)  # Last 30 seconds

    print("\n" + "=" * 40)
    print("ROI Entry/Exit Test Results:")
    print(f"  Total Entries: {final_counts['entries']}")
    print(f"  Total Exits: {final_counts['exits']}")
    print(f"  Current in ROI: {final_counts['current_in_roi']}")
    print(f"  Net Flow: {final_counts['net_flow']}")
    print(f"  Recent Entries (30s): {recent_stats['recent_entries']}")
    print(f"  Recent Exits (30s): {recent_stats['recent_exits']}")
    print(f"  Entry Rate: {recent_stats['entries_per_minute']:.1f}/min")
    print(f"  Exit Rate: {recent_stats['exits_per_minute']:.1f}/min")

    # Verify basic functionality
    if final_counts['entries'] > 0 and final_counts['exits'] > 0:
        print("✅ ROI entry/exit tracking working correctly!")
    else:
        print("❌ ROI entry/exit tracking may have issues")

    return final_counts

if __name__ == "__main__":
    # Run tests
    test_tracking()

    # Test ROI entry/exit
    test_roi_entry_exit()

    # Ask user if they want to see visualization
    response = input("\nWould you like to see the visual tracking demo? (y/n): ")
    if response.lower() in ['y', 'yes']:
        test_visualization()

    print("\nAll tests completed successfully!")
