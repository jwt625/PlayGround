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

        # ROI tracking
        self.roi_status = None  # 'inside', 'outside', or None (unknown)
        self.roi_entry_frame = None  # Frame number when entered ROI
        self.roi_exit_frame = None   # Frame number when exited ROI
        
    def update_position(self, centroid, bbox, area):
        """Update object position and calculate motion properties."""
        # Calculate velocity (pixels per frame)
        if len(self.positions) > 0:
            prev_centroid = self.positions[-1]
            self.velocity = (
                centroid[0] - prev_centroid[0],
                centroid[1] - prev_centroid[1]
            )

            # Determine primary direction using smoothed movement over multiple frames
            old_direction = self.direction
            self._update_direction_stable()

            # Debug: Print direction changes for tracking (only significant changes)
            # Uncomment the lines below for debugging direction changes
            # if old_direction != self.direction and len(self.positions) > 5:
            #     print(f"DEBUG: Object {self.id} direction STABILIZED from {old_direction} to {self.direction}")
            #     print(f"  Current velocity: {self.velocity}")
            #     print(f"  Overall movement: {self._get_overall_movement()}")
            #     print(f"  Position history: {self.positions[-3:]}")
        
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

    def _update_direction_stable(self):
        """Update direction using stable calculation over multiple frames.
        For bridge side view: only horizontal movement (left/right) is valid."""
        # Need at least 5 positions for direction calculation (reduced for faster response)
        if len(self.positions) < 5:
            return

        # Calculate overall movement over last 5 frames for stability
        overall_movement = self._get_overall_movement()

        # For bridge side view, only consider horizontal movement
        horizontal_movement = overall_movement[0]

        # Lower movement threshold for faster direction detection
        min_movement_threshold = 15  # pixels - reduced for faster response
        if abs(horizontal_movement) < min_movement_threshold:
            return

        # Once a direction is established, make it hard to change (highway traffic is consistent)
        if self.direction is not None:
            # Require stronger evidence to change direction on a highway
            direction_change_threshold = 30  # pixels - reduced for better responsiveness
            if abs(horizontal_movement) < direction_change_threshold:
                return

        # Determine direction based ONLY on horizontal movement (bridge side view)
        new_direction = None
        if abs(horizontal_movement) > min_movement_threshold:
            new_direction = 'right' if horizontal_movement > 0 else 'left'

        # Only change direction if we have a strong signal and it's different
        if new_direction and new_direction != self.direction:
            # Additional stability check: confirm direction over multiple frames
            if self._confirm_direction_change(new_direction):
                self.direction = new_direction

    def _get_overall_movement(self):
        """Get overall movement vector over the last several frames."""
        if len(self.positions) < 2:
            return (0, 0)

        # Use last 5 positions or all available positions
        num_frames = min(5, len(self.positions))
        start_pos = self.positions[-num_frames]
        end_pos = self.positions[-1]

        return (end_pos[0] - start_pos[0], end_pos[1] - start_pos[1])

    def _confirm_direction_change(self, new_direction):
        """Confirm direction change by checking consistency over recent frames.
        For bridge side view: only check horizontal movement."""
        if len(self.positions) < 3:
            return True  # Not enough data, allow change

        # Check if the new direction is consistent with recent horizontal movement
        recent_movements = []
        for i in range(max(1, len(self.positions) - 3), len(self.positions)):
            if i > 0:
                horizontal_movement = self.positions[i][0] - self.positions[i-1][0]
                recent_movements.append(horizontal_movement)

        # Count how many recent movements support the new direction (only horizontal)
        supporting_count = 0
        for movement in recent_movements:
            if new_direction == 'right' and movement > 1:  # Small threshold to ignore noise
                supporting_count += 1
            elif new_direction == 'left' and movement < -1:  # Small threshold to ignore noise
                supporting_count += 1

        # Require majority support for direction change (60% for faster response)
        return supporting_count >= len(recent_movements) * 0.6
    
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

    def update_roi_status(self, roi_rect, frame_number):
        """Update ROI status and track entry/exit events."""
        if roi_rect is None:
            return None

        x, y, w, h = roi_rect
        cx, cy = self.centroid

        # Check if centroid is inside ROI
        is_inside = (x <= cx <= x + w) and (y <= cy <= y + h)
        new_status = 'inside' if is_inside else 'outside'

        # Detect status changes (only for confirmed objects)
        if self.confirmed and self.roi_status is not None and self.roi_status != new_status:
            if self.roi_status == 'outside' and new_status == 'inside':
                # Object entered ROI
                self.roi_entry_frame = frame_number
                self.roi_status = new_status
                return 'entered'
            elif self.roi_status == 'inside' and new_status == 'outside':
                # Object exited ROI
                self.roi_exit_frame = frame_number
                self.roi_status = new_status
                return 'exited'

        # Update status (initialize for new objects)
        self.roi_status = new_status
        return None


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
        self.counts = {'left': 0, 'right': 0, 'total': 0}  # Only left/right for bridge side view
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

                    # Debug: Print crossing information
                    print(f"🚗 TRAFFIC COUNT: Object {obj.id} crossed counting line!")
                    print(f"  Direction: {obj.direction}")
                    print(f"  Overall movement: {obj._get_overall_movement() if hasattr(obj, '_get_overall_movement') else 'N/A'}")
                    print(f"  Position history: {obj.positions[-3:] if len(obj.positions) >= 3 else obj.positions}")

                    # Count by direction (only left/right for bridge side view)
                    if obj.direction in ['left', 'right']:
                        self.counts[obj.direction] += 1
                        print(f"  ✅ Added to {obj.direction} count. New counts: L:{self.counts['left']} R:{self.counts['right']} Total:{self.counts['total']}")
                    else:
                        print(f"  ⚠️ WARNING: Object direction '{obj.direction}' not valid for bridge side view! Not counted.")

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
        self.counts = {'left': 0, 'right': 0, 'total': 0}  # Only left/right for bridge side view
        self.counted_objects.clear()


class ROITracker:
    """
    ROI (Region of Interest) entry/exit tracker.

    Tracks objects entering and leaving the ROI area for traffic flow analysis.
    """

    def __init__(self):
        """Initialize ROI tracker."""
        self.entry_count = 0
        self.exit_count = 0
        self.objects_in_roi = set()  # Track which objects are currently in ROI
        self.entry_events = []  # List of (object_id, frame_number, timestamp)
        self.exit_events = []   # List of (object_id, frame_number, timestamp)
        self.roi_rect = None    # Current ROI rectangle (x, y, w, h)

    def set_roi(self, roi_rect):
        """Set the ROI rectangle."""
        self.roi_rect = roi_rect

    def update(self, tracked_objects, frame_number):
        """Update ROI tracking with current objects."""
        if self.roi_rect is None:
            return

        import time
        current_time = time.time()

        # Only process confirmed objects to avoid duplicate counting
        confirmed_objects = [obj for obj in tracked_objects if obj.confirmed]

        for obj in confirmed_objects:
            # Update object's ROI status and check for events
            event = obj.update_roi_status(self.roi_rect, frame_number)

            if event == 'entered':
                self.entry_count += 1
                self.objects_in_roi.add(obj.id)
                self.entry_events.append((obj.id, frame_number, current_time))

                # Limit event history to prevent memory growth
                if len(self.entry_events) > 1000:
                    self.entry_events = self.entry_events[-500:]

            elif event == 'exited':
                self.exit_count += 1
                self.objects_in_roi.discard(obj.id)
                self.exit_events.append((obj.id, frame_number, current_time))

                # Limit event history to prevent memory growth
                if len(self.exit_events) > 1000:
                    self.exit_events = self.exit_events[-500:]

        # Update current objects in ROI count based on actual positions
        current_in_roi = set()
        for obj in confirmed_objects:
            if obj.roi_status == 'inside':
                current_in_roi.add(obj.id)
        self.objects_in_roi = current_in_roi

    def get_counts(self):
        """Get current entry/exit counts."""
        return {
            'entries': self.entry_count,
            'exits': self.exit_count,
            'current_in_roi': len(self.objects_in_roi),
            'net_flow': self.entry_count - self.exit_count
        }

    def get_recent_events(self, seconds=60):
        """Get recent entry/exit events within the specified time window."""
        import time
        current_time = time.time()
        cutoff_time = current_time - seconds

        recent_entries = [e for e in self.entry_events if e[2] >= cutoff_time]
        recent_exits = [e for e in self.exit_events if e[2] >= cutoff_time]

        return {
            'recent_entries': len(recent_entries),
            'recent_exits': len(recent_exits),
            'entries_per_minute': len(recent_entries) * (60.0 / seconds),
            'exits_per_minute': len(recent_exits) * (60.0 / seconds)
        }

    def reset(self):
        """Reset all ROI tracking data."""
        self.entry_count = 0
        self.exit_count = 0
        self.objects_in_roi.clear()
        self.entry_events.clear()
        self.exit_events.clear()

    def get_statistics(self):
        """Get comprehensive ROI statistics."""
        counts = self.get_counts()
        recent = self.get_recent_events()

        return {
            **counts,
            **recent,
            'total_events': len(self.entry_events) + len(self.exit_events)
        }
