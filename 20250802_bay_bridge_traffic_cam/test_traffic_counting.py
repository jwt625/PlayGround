#!/usr/bin/env python3
"""
Test script to demonstrate traffic counting functionality.
This creates synthetic traffic data to verify that L/R counting works correctly.
"""

import numpy as np
import cv2
from object_tracker import ObjectTracker, TrafficCounter

def create_synthetic_traffic():
    """Create synthetic traffic data moving left and right."""
    
    # Initialize tracker and counter
    tracker = ObjectTracker(max_distance=50, max_disappeared=10, min_detection_frames=3)
    
    # Set up a counting line in the middle of the frame
    counting_line = (400, 200, 400, 400)  # Vertical line at x=400
    counter = TrafficCounter(counting_lines=[counting_line])
    
    print("Testing Traffic Counting for Bridge Side View")
    print("=" * 50)
    print(f"Counting line: {counting_line}")
    print("Simulating vehicles moving left and right...")
    print()
    
    frame_count = 0
    
    # Simulate vehicles moving right (left to right across the counting line)
    print("Phase 1: Vehicles moving RIGHT (left to right)")
    for i in range(30):
        frame_count += 1
        
        # Create contours for vehicles moving right
        contours = []
        
        # Vehicle 1: starts at x=300, moves right
        x1 = 300 + i * 5
        if x1 < 600:  # Keep it in frame
            contour1 = np.array([[[x1, 250]], [[x1+30, 250]], [[x1+30, 280]], [[x1, 280]]], dtype=np.int32)
            contours.append(contour1)
        
        # Vehicle 2: starts at x=250, moves right (slightly behind)
        if i > 5:
            x2 = 250 + (i-5) * 4
            if x2 < 600:
                contour2 = np.array([[[x2, 320]], [[x2+25, 320]], [[x2+25, 345]], [[x2, 345]]], dtype=np.int32)
                contours.append(contour2)
        
        # Update tracker
        tracked_objects = tracker.update(contours)
        
        # Update counter
        counter.update(tracked_objects)
        
        # Print status every 10 frames
        if frame_count % 10 == 0:
            counts = counter.get_counts()
            print(f"Frame {frame_count:2d}: L:{counts['left']} R:{counts['right']} Total:{counts['total']}")
    
    print()
    print("Phase 2: Vehicles moving LEFT (right to left)")

    # DON'T reset tracker - keep the same counter to accumulate counts
    # But wait a bit for the previous vehicles to disappear
    for i in range(10):
        frame_count += 1
        tracked_objects = tracker.update([])  # Empty frame to let old objects disappear

    for i in range(30):
        frame_count += 1

        # Create contours for vehicles moving left
        contours = []

        # Vehicle 3: starts at x=500, moves left
        x3 = 500 - i * 5
        if x3 > 200:  # Keep it in frame
            contour3 = np.array([[[x3, 250]], [[x3+30, 250]], [[x3+30, 280]], [[x3, 280]]], dtype=np.int32)
            contours.append(contour3)

        # Vehicle 4: starts at x=550, moves left (slightly behind)
        if i > 5:
            x4 = 550 - (i-5) * 4
            if x4 > 200:
                contour4 = np.array([[[x4, 320]], [[x4+25, 320]], [[x4+25, 345]], [[x4, 345]]], dtype=np.int32)
                contours.append(contour4)

        # Update tracker
        tracked_objects = tracker.update(contours)

        # Update counter (same counter to accumulate counts)
        counter.update(tracked_objects)

        # Print status every 10 frames
        if frame_count % 10 == 0:
            counts = counter.get_counts()
            print(f"Frame {frame_count:2d}: L:{counts['left']} R:{counts['right']} Total:{counts['total']}")
    
    print()
    print("Final Results:")
    print("=" * 30)
    final_counts = counter.get_counts()
    print(f"Left (R→L):  {final_counts['left']} vehicles")
    print(f"Right (L→R): {final_counts['right']} vehicles")
    print(f"Total:       {final_counts['total']} vehicles")
    print()
    
    if final_counts['left'] > 0 and final_counts['right'] > 0:
        print("✅ SUCCESS: Traffic counting is working correctly!")
        print("   Both left and right directions are being counted.")
    else:
        print("❌ ISSUE: Some directions are not being counted.")
        print("   Check the direction calculation and line crossing logic.")

if __name__ == "__main__":
    create_synthetic_traffic()
