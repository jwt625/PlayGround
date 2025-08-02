#!/usr/bin/env python3
"""
Object Tracker for Motion-Based Traffic Detection

Implements centroid-based tracking with Hungarian algorithm for optimal assignment.
Designed for real-time traffic monitoring with small, distant objects.
"""

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from collections import OrderedDict
import time


class TrackedObject:
    """Represents a single tracked object with its history and properties."""
    
    def __init__(self, object_id, centroid, bbox, area):
        self.id = object_id
        self.centroid = centroid
        self.bbox = bbox  # (x, y, w, h)
        self.area = area
        
        # Tracking history
        self.positions = [centroid]
        self.bboxes = [bbox]
        self.areas = [area]
        self.timestamps = [time.time()]
        
        # State tracking
        self.disappeared_count = 0
        self.confirmed = False  # True after min_detection_frames
        self.detection_count = 1
        
        # Motion properties
        self.velocity = (0, 0)
        self.direction = None  # 'left', 'right', 'up', 'down', or None
        
    def update_position(self, centroid, bbox, area):
        """Update object position and calculate motion properties."""
        # Calculate velocity (pixels per frame)
        if len(self.positions) > 0:
            prev_centroid = self.positions[-1]
            self.velocity = (
                centroid[0] - prev_centroid[0],
                centroid[1] - prev_centroid[1]
            )
            
            # Determine primary direction
            if abs(self.velocity[0]) > abs(self.velocity[1]):
                self.direction = 'right' if self.velocity[0] > 0 else 'left'
            elif abs(self.velocity[1]) > 2:  # Threshold for vertical movement
                self.direction = 'down' if self.velocity[1] > 0 else 'up'
            else:
                self.direction = None
        
        # Update history
        self.centroid = centroid
        self.bbox = bbox
        self.area = area
        self.positions.append(centroid)
        self.bboxes.append(bbox)
        self.areas.append(area)
        self.timestamps.append(time.time())
        
        # Limit history size for memory efficiency
        max_history = 30
        if len(self.positions) > max_history:
            self.positions = self.positions[-max_history:]
            self.bboxes = self.bboxes[-max_history:]
            self.areas = self.areas[-max_history:]
            self.timestamps = self.timestamps[-max_history:]
        
        # Reset disappeared count and increment detection count
        self.disappeared_count = 0
        self.detection_count += 1
    
    def mark_disappeared(self):
        """Mark object as disappeared for this frame."""
        self.disappeared_count += 1
    
    def get_predicted_position(self):
        """Predict next position based on velocity."""
        if len(self.positions) >= 2:
            return (
                int(self.centroid[0] + self.velocity[0]),
                int(self.centroid[1] + self.velocity[1])
            )
        return self.centroid
    
    def get_speed_pixels_per_second(self):
        """Calculate speed in pixels per second."""
        if len(self.positions) >= 2 and len(self.timestamps) >= 2:
            time_diff = self.timestamps[-1] - self.timestamps[-2]
            if time_diff > 0:
                distance = np.sqrt(self.velocity[0]**2 + self.velocity[1]**2)
                return distance / time_diff
        return 0.0
    
    def get_trajectory_length(self):
        """Get total distance traveled."""
        if len(self.positions) < 2:
            return 0.0
        
        total_distance = 0.0
        for i in range(1, len(self.positions)):
            prev_pos = self.positions[i-1]
            curr_pos = self.positions[i]
            distance = np.sqrt((curr_pos[0] - prev_pos[0])**2 + 
                             (curr_pos[1] - prev_pos[1])**2)
            total_distance += distance
        
        return total_distance


class ObjectTracker:
    """
    Centroid-based object tracker with Hungarian algorithm assignment.
    
    Tracks objects across frames by associating detections based on proximity
    and maintains object IDs for consistent tracking.
    """
    
    def __init__(self, max_distance=50, max_disappeared=10, min_detection_frames=3):
        """
        Initialize the object tracker.
        
        Args:
            max_distance: Maximum distance for associating objects between frames
            max_disappeared: Maximum frames an object can be missing before removal
            min_detection_frames: Minimum frames before object is considered confirmed
        """
        self.max_distance = max_distance
        self.max_disappeared = max_disappeared
        self.min_detection_frames = min_detection_frames
        
        # Tracking state
        self.next_object_id = 0
        self.objects = OrderedDict()  # {id: TrackedObject}
        
        # Statistics
        self.total_objects_created = 0
        self.total_objects_confirmed = 0
        self.frame_count = 0
    
    def _calculate_centroid(self, contour):
        """Calculate centroid of a contour."""
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            return (cx, cy)
        else:
            # Fallback to bounding box center
            x, y, w, h = cv2.boundingRect(contour)
            return (x + w // 2, y + h // 2)
    
    def _calculate_distance_matrix(self, object_centroids, detection_centroids):
        """Calculate distance matrix between existing objects and new detections."""
        if len(object_centroids) == 0 or len(detection_centroids) == 0:
            return np.array([])
        
        distances = np.zeros((len(object_centroids), len(detection_centroids)))
        
        for i, obj_centroid in enumerate(object_centroids):
            for j, det_centroid in enumerate(detection_centroids):
                # Euclidean distance
                distance = np.sqrt((obj_centroid[0] - det_centroid[0])**2 + 
                                 (obj_centroid[1] - det_centroid[1])**2)
                distances[i, j] = distance
        
        return distances
    
    def update(self, contours):
        """
        Update tracker with new detections.
        
        Args:
            contours: List of contours from motion detection
            
        Returns:
            List of confirmed tracked objects
        """
        self.frame_count += 1
        
        # Extract detection information
        detections = []
        for contour in contours:
            centroid = self._calculate_centroid(contour)
            bbox = cv2.boundingRect(contour)
            area = cv2.contourArea(contour)
            detections.append({
                'centroid': centroid,
                'bbox': bbox,
                'area': area,
                'contour': contour
            })
        
        # If no existing objects, create new ones for all detections
        if len(self.objects) == 0:
            for detection in detections:
                self._create_new_object(detection)
        
        # If no detections, mark all objects as disappeared
        elif len(detections) == 0:
            for obj_id in list(self.objects.keys()):
                self.objects[obj_id].mark_disappeared()
                if self.objects[obj_id].disappeared_count > self.max_disappeared:
                    del self.objects[obj_id]
        
        # Associate detections with existing objects
        else:
            self._associate_detections(detections)
        
        # Return confirmed tracked objects
        return self._get_confirmed_objects()

    def _create_new_object(self, detection):
        """Create a new tracked object."""
        obj = TrackedObject(
            self.next_object_id,
            detection['centroid'],
            detection['bbox'],
            detection['area']
        )
        self.objects[self.next_object_id] = obj
        self.next_object_id += 1
        self.total_objects_created += 1

    def _associate_detections(self, detections):
        """Associate detections with existing objects using Hungarian algorithm."""
        # Get current object centroids (use predicted positions for better tracking)
        object_ids = list(self.objects.keys())
        object_centroids = []

        for obj_id in object_ids:
            obj = self.objects[obj_id]
            # Use predicted position for better association
            predicted_pos = obj.get_predicted_position()
            object_centroids.append(predicted_pos)

        # Get detection centroids
        detection_centroids = [det['centroid'] for det in detections]

        # Calculate distance matrix
        distance_matrix = self._calculate_distance_matrix(object_centroids, detection_centroids)

        if distance_matrix.size > 0:
            # Use Hungarian algorithm for optimal assignment
            row_indices, col_indices = linear_sum_assignment(distance_matrix)

            # Track which detections and objects were matched
            used_detection_indices = set()
            used_object_indices = set()

            # Process assignments
            for row, col in zip(row_indices, col_indices):
                # Check if distance is within threshold
                if distance_matrix[row, col] <= self.max_distance:
                    # Update existing object
                    obj_id = object_ids[row]
                    detection = detections[col]

                    self.objects[obj_id].update_position(
                        detection['centroid'],
                        detection['bbox'],
                        detection['area']
                    )

                    used_detection_indices.add(col)
                    used_object_indices.add(row)

            # Handle unmatched detections (create new objects)
            for i, detection in enumerate(detections):
                if i not in used_detection_indices:
                    self._create_new_object(detection)

            # Handle unmatched objects (mark as disappeared)
            for i, obj_id in enumerate(object_ids):
                if i not in used_object_indices:
                    self.objects[obj_id].mark_disappeared()

                    # Remove objects that have been missing too long
                    if self.objects[obj_id].disappeared_count > self.max_disappeared:
                        del self.objects[obj_id]
        else:
            # No existing objects or no detections - handle edge case
            for detection in detections:
                self._create_new_object(detection)

    def _get_confirmed_objects(self):
        """Get list of confirmed tracked objects."""
        confirmed_objects = []

        for obj in self.objects.values():
            # Confirm object if it has been detected enough times
            if not obj.confirmed and obj.detection_count >= self.min_detection_frames:
                obj.confirmed = True
                self.total_objects_confirmed += 1

            # Only return confirmed objects that are currently visible
            if obj.confirmed and obj.disappeared_count == 0:
                confirmed_objects.append(obj)

        return confirmed_objects

    def get_all_objects(self):
        """Get all tracked objects (confirmed and unconfirmed)."""
        return list(self.objects.values())

    def get_statistics(self):
        """Get tracking statistics."""
        active_objects = len([obj for obj in self.objects.values() if obj.disappeared_count == 0])
        confirmed_objects = len([obj for obj in self.objects.values() if obj.confirmed])

        return {
            'frame_count': self.frame_count,
            'active_objects': active_objects,
            'confirmed_objects': confirmed_objects,
            'total_created': self.total_objects_created,
            'total_confirmed': self.total_objects_confirmed,
            'tracking_objects': len(self.objects)
        }

    def reset(self):
        """Reset tracker state."""
        self.objects.clear()
        self.next_object_id = 0
        self.total_objects_created = 0
        self.total_objects_confirmed = 0
        self.frame_count = 0


class TrafficCounter:
    """
    Traffic counting functionality using tracked objects.

    Counts vehicles crossing virtual counting lines with direction detection.
    """

    def __init__(self, counting_lines=None):
        """
        Initialize traffic counter.

        Args:
            counting_lines: List of counting lines [(x1, y1, x2, y2), ...]
        """
        self.counting_lines = counting_lines or []
        self.counts = {'left': 0, 'right': 0, 'up': 0, 'down': 0, 'total': 0}
        self.counted_objects = set()  # Track which objects have been counted

    def add_counting_line(self, x1, y1, x2, y2):
        """Add a counting line."""
        self.counting_lines.append((x1, y1, x2, y2))

    def update(self, tracked_objects):
        """Update counts based on tracked objects crossing lines."""
        for obj in tracked_objects:
            if obj.id in self.counted_objects:
                continue

            # Check if object crosses any counting line
            for line in self.counting_lines:
                if self._crosses_line(obj, line):
                    self.counted_objects.add(obj.id)
                    self.counts['total'] += 1

                    # Count by direction
                    if obj.direction:
                        self.counts[obj.direction] += 1

                    break  # Only count once per object

    def _crosses_line(self, obj, line):
        """Check if object trajectory crosses a counting line."""
        if len(obj.positions) < 2:
            return False

        x1, y1, x2, y2 = line

        # Check last two positions
        prev_pos = obj.positions[-2]
        curr_pos = obj.positions[-1]

        # Simple line intersection check
        return self._line_intersection(prev_pos, curr_pos, (x1, y1), (x2, y2))

    def _line_intersection(self, p1, p2, p3, p4):
        """Check if line segments p1-p2 and p3-p4 intersect."""
        def ccw(A, B, C):
            return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

        return ccw(p1, p3, p4) != ccw(p2, p3, p4) and ccw(p1, p2, p3) != ccw(p1, p2, p4)

    def get_counts(self):
        """Get current counts."""
        return self.counts.copy()

    def reset_counts(self):
        """Reset all counts."""
        self.counts = {'left': 0, 'right': 0, 'up': 0, 'down': 0, 'total': 0}
        self.counted_objects.clear()
