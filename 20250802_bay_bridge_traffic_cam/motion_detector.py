#!/usr/bin/env python3
"""
Minimal motion-based traffic detection for Bay Bridge side view.
Uses background subtraction to detect moving vehicles even when partially occluded.
"""

import cv2
import numpy as np
import requests
import time
from datetime import datetime
import os
import motion_config as config
from object_tracker import ObjectTracker, TrafficCounter, ROITracker

class MotionTrafficDetector:
    def __init__(self, webcam_url=None):
        self.webcam_url = webcam_url or config.WEBCAM_URL

        # Initialize background subtractor with config
        bg_config = config.BACKGROUND_SUBTRACTOR_CONFIG
        self.background_subtractor = cv2.createBackgroundSubtractorMOG2(
            detectShadows=bg_config["detectShadows"],
            varThreshold=bg_config["varThreshold"],
            history=bg_config["history"]
        )

        # Set shadow parameters if shadow detection is enabled
        if bg_config["detectShadows"]:
            self.background_subtractor.setShadowValue(bg_config["shadowValue"])
            self.background_subtractor.setShadowThreshold(bg_config["shadowThreshold"])

        # ROI for bridge deck (will be set interactively)
        self.roi = None
        self.roi_set = False

        # Motion detection parameters from config
        motion_config = config.MOTION_DETECTION
        self.min_contour_area = motion_config["min_contour_area"]
        self.max_contour_area = motion_config["max_contour_area"]
        self.min_aspect_ratio = motion_config["min_aspect_ratio"]
        self.max_aspect_ratio = motion_config["max_aspect_ratio"]
        self.min_extent = motion_config["min_extent"]

        # Traffic counting
        self.motion_objects = []
        self.frame_count = 0

        # Object tracking
        tracking_config = config.TRACKING_CONFIG
        self.tracker = ObjectTracker(
            max_distance=tracking_config["max_distance"],
            max_disappeared=tracking_config["max_disappeared"],
            min_detection_frames=tracking_config["min_detection_frames"]
        ) if tracking_config["enable_tracking"] else None

        self.traffic_counter = TrafficCounter(
            counting_lines=tracking_config["counting_lines"]
        ) if tracking_config["enable_counting"] else None

        self.roi_tracker = ROITracker() if tracking_config["enable_roi_tracking"] else None

        # PERFORMANCE OPTIMIZATION: Persistent video capture
        self.cap = None
        self.cap_initialized = False
        self.last_frame_time = time.time()
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.current_fps = 0

        # Pause/Resume functionality
        self.is_paused = False
        self.paused_frame = None
        self.paused_fg_mask = None
        self.paused_contours = []
        self.paused_tracked_objects = []
        self.pause_analysis_printed = False  # Track if we've printed analysis for current pause

        # Print configuration on startup
        config.print_config_summary()
        
    def get_frame_from_webcam(self):
        """Optimized frame capture with persistent connection."""
        try:
            # Initialize persistent video capture if needed
            if not self.cap_initialized:
                self.cap = cv2.VideoCapture(self.webcam_url)
                if self.cap.isOpened():
                    # Optimize capture settings for speed
                    self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer to get latest frame
                    self.cap.set(cv2.CAP_PROP_FPS, 60)        # Request higher FPS if available
                    self.cap_initialized = True
                    print("Persistent video capture initialized")
                else:
                    self.cap = None

            # Try to read from persistent connection
            if self.cap and self.cap.isOpened():
                ret, frame = self.cap.read()
                if ret:
                    return frame
                else:
                    # Connection lost, reinitialize
                    print("Video capture connection lost, reinitializing...")
                    self.cap.release()
                    self.cap_initialized = False
                    return self.get_frame_from_webcam()  # Retry once

            # Fallback to HTTP request method (only if OpenCV fails)
            response = requests.get(self.webcam_url, stream=True, timeout=config.WEBCAM_TIMEOUT)
            if response.status_code == 200:
                bytes_data = b''
                for chunk in response.iter_content(chunk_size=1024):
                    bytes_data += chunk
                    start = bytes_data.find(b'\xff\xd8')  # JPEG start
                    end = bytes_data.find(b'\xff\xd9')    # JPEG end

                    if start != -1 and end != -1 and end > start:
                        jpeg_data = bytes_data[start:end+2]
                        frame = cv2.imdecode(np.frombuffer(jpeg_data, np.uint8), cv2.IMREAD_COLOR)
                        return frame

        except Exception as e:
            print(f"Error capturing frame: {e}")

        return None

    def cleanup(self):
        """Clean up resources."""
        if self.cap:
            self.cap.release()
            self.cap = None
            self.cap_initialized = False
    
    def create_demo_frame(self):
        """Create a demo frame for testing when webcam is unavailable."""
        demo_config = config.DEMO_CONFIG
        width, height = demo_config["demo_frame_size"]
        frame = np.full((height, width, 3), demo_config["demo_background_color"], dtype=np.uint8)

        # Add some moving objects for demo
        t = time.time()
        colors = demo_config["demo_object_colors"]
        speed_min, speed_max = demo_config["demo_speed_range"]

        for i in range(demo_config["demo_object_count"]):
            speed = speed_min + (speed_max - speed_min) * (i / max(1, demo_config["demo_object_count"] - 1))
            x = int(100 + i * 150 + 50 * np.sin(t * speed))
            y = int(200 + i * 20)
            color = colors[i % len(colors)]

            cv2.rectangle(frame, (x, y), (x+40, y+40), color, -1)

        font_scale, thickness = self.get_scaled_font_params()
        cv2.putText(frame, "DEMO MODE - Moving objects", (10, 30),
                   config.DISPLAY_CONFIG["font"], font_scale,
                   config.DISPLAY_CONFIG["text_color"], thickness)

        return frame
    
    def set_roi_interactive(self, frame):
        """Let user select Region of Interest (bridge deck area)."""
        print("Select the bridge deck area by dragging a rectangle, then press ENTER")
        print("Or press ENTER without selecting to use the full frame")
        print("Press 'r' to reset selection, 'q' to quit")

        roi = cv2.selectROI("Select Bridge Deck ROI", frame, False, False)
        cv2.destroyWindow("Select Bridge Deck ROI")

        if roi[2] > 0 and roi[3] > 0:  # Valid ROI selection
            self.roi = roi
            self.roi_set = True
            print(f"ROI set: x={roi[0]}, y={roi[1]}, w={roi[2]}, h={roi[3]}")

            # Update ROI tracker
            if self.roi_tracker is not None:
                self.roi_tracker.set_roi(roi)
                print("ROI tracker updated with new ROI")
        else:
            # No ROI selected - use full frame
            print("No ROI selected, using full frame for motion detection")
            self.roi = None
            self.roi_set = True  # Mark as set so we don't ask again
            if self.roi_tracker is not None:
                self.roi_tracker.set_roi(None)
                print("ROI tracker set to use full frame")

    def set_counting_line_interactive(self, frame):
        """Let user set a counting line for traffic counting."""
        if self.traffic_counter is None:
            print("Traffic counting is disabled")
            return

        print("Click two points to set a counting line, then press ENTER")
        print("Press 'q' to cancel")

        points = []

        def mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                points.append((x, y))
                if len(points) == 2:
                    cv2.destroyWindow("Set Counting Line")

        cv2.namedWindow("Set Counting Line")
        cv2.setMouseCallback("Set Counting Line", mouse_callback)

        while len(points) < 2:
            display_frame = frame.copy()

            # Draw existing points
            for i, point in enumerate(points):
                cv2.circle(display_frame, point, 5, (0, 255, 0), -1)
                cv2.putText(display_frame, f"P{i+1}", (point[0]+10, point[1]),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            # Draw instruction
            cv2.putText(display_frame, f"Click point {len(points)+1} of 2", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            cv2.imshow("Set Counting Line", display_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                cv2.destroyWindow("Set Counting Line")
                return

        if len(points) == 2:
            # Adjust for ROI if set
            if self.roi_set and self.roi is not None:
                roi_x, roi_y, _, _ = self.roi
                adjusted_points = [(p[0] - roi_x, p[1] - roi_y) for p in points]
                line = (adjusted_points[0][0], adjusted_points[0][1],
                       adjusted_points[1][0], adjusted_points[1][1])
            else:
                line = (points[0][0], points[0][1], points[1][0], points[1][1])

            self.traffic_counter.add_counting_line(*line)
            print(f"Counting line added: {line}")

        cv2.destroyWindow("Set Counting Line")
    
    def detect_motion(self, frame):
        """Detect motion in the frame and return motion mask and contours."""
        if frame is None:
            return None, []

        # Apply ROI if set
        if self.roi_set and self.roi is not None:
            x, y, w, h = self.roi
            roi_frame = frame[y:y+h, x:x+w]
        else:
            roi_frame = frame

        # Apply background subtraction
        fg_mask = self.background_subtractor.apply(roi_frame)

        # Remove shadows if configured (convert grey shadow pixels to black)
        bg_config = config.BACKGROUND_SUBTRACTOR_CONFIG
        if bg_config["detectShadows"] and bg_config["removeShadows"]:
            # Remove shadow pixels (convert 127 to 0, keep 255 as 255)
            fg_mask[fg_mask == bg_config["shadowValue"]] = 0

        # Clean up the mask using config parameters
        motion_config = config.MOTION_DETECTION
        kernel_size = motion_config["morph_kernel_size"]
        iterations = motion_config["morph_iterations"]

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel, iterations=iterations)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel, iterations=iterations)

        # Find contours
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Filter contours by size and shape
        valid_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if self.min_contour_area <= area <= self.max_contour_area:
                # Additional shape filtering
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h if h > 0 else 0
                extent = area / (w * h) if w > 0 and h > 0 else 0

                if (self.min_aspect_ratio <= aspect_ratio <= self.max_aspect_ratio and
                    extent >= self.min_extent):
                    valid_contours.append(contour)

        # Apply object tracking if enabled
        tracked_objects = []
        if self.tracker is not None:
            tracked_objects = self.tracker.update(valid_contours)

            # Update traffic counter if enabled
            if self.traffic_counter is not None:
                self.traffic_counter.update(tracked_objects)

            # Update ROI tracker if enabled
            if self.roi_tracker is not None:
                self.roi_tracker.update(tracked_objects, self.frame_count)

        return fg_mask, valid_contours, tracked_objects
    
    def draw_detections(self, frame, fg_mask, contours, tracked_objects=None):
        """Draw motion detections and tracked objects on the frame."""
        display_frame = frame.copy()
        display_config = config.DISPLAY_CONFIG
        tracking_config = config.TRACKING_CONFIG

        # Draw ROI if set
        if self.roi_set and self.roi is not None:
            x, y, w, h = self.roi
            cv2.rectangle(display_frame, (x, y), (x+w, y+h),
                         display_config["roi_color"], display_config["roi_thickness"])
            small_font_scale, small_thickness = self.get_scaled_font_params("small_font_scale", "small_font_thickness")
            cv2.putText(display_frame, "Bridge Deck ROI", (x, y-10),
                       display_config["font"], small_font_scale,
                       display_config["roi_color"], small_thickness)

        # Draw tracked objects if available, otherwise draw regular contours
        motion_count = 0

        if tracked_objects is not None and tracking_config["enable_tracking"]:
            # Draw tracked objects with IDs and trajectories
            for obj in tracked_objects:
                x, y, w, h = obj.bbox

                # Adjust coordinates if ROI is set
                if self.roi_set and self.roi is not None:
                    roi_x, roi_y, _, _ = self.roi
                    x += roi_x
                    y += roi_y

                # Draw bounding box with different color for tracked objects
                color = (0, 255, 255)  # Yellow for tracked objects
                cv2.rectangle(display_frame, (x, y), (x+w, y+h), color, 2)

                # Draw object ID if enabled
                if tracking_config["show_object_ids"]:
                    cv2.putText(display_frame, f"ID:{obj.id}", (x, y-25),
                               display_config["font"], display_config["small_font_scale"],
                               color, display_config["small_font_thickness"])

                # Draw trajectory if enabled
                if tracking_config["show_trajectories"] and len(obj.positions) > 1:
                    trajectory_points = obj.positions[-tracking_config["trajectory_length"]:]

                    # Adjust trajectory points for ROI
                    if self.roi_set and self.roi is not None:
                        roi_x, roi_y, _, _ = self.roi
                        trajectory_points = [(px + roi_x, py + roi_y) for px, py in trajectory_points]

                    # Draw trajectory line
                    for i in range(1, len(trajectory_points)):
                        cv2.line(display_frame, trajectory_points[i-1], trajectory_points[i],
                                color, 1)

                # Show speed and direction info
                speed = obj.get_speed_pixels_per_second()
                if speed > 0:
                    cv2.putText(display_frame, f"{speed:.1f}px/s", (x, y-5),
                               display_config["font"], display_config["small_font_scale"],
                               color, display_config["small_font_thickness"])

                if obj.direction:
                    cv2.putText(display_frame, obj.direction, (x+w-30, y-5),
                               display_config["font"], display_config["small_font_scale"],
                               color, display_config["small_font_thickness"])

                motion_count += 1
        else:
            # Draw regular contours (fallback when tracking is disabled)
            for contour in contours:
                # Get bounding box
                x, y, w, h = cv2.boundingRect(contour)
                area = cv2.contourArea(contour)

                # Adjust coordinates if ROI is set
                if self.roi_set and self.roi is not None:
                    roi_x, roi_y, _, _ = self.roi
                    x += roi_x
                    y += roi_y

                # Draw bounding box
                cv2.rectangle(display_frame, (x, y), (x+w, y+h),
                             display_config["detection_color"], display_config["detection_thickness"])

                # Draw contour
                if self.roi_set and self.roi is not None:
                    # Adjust contour coordinates
                    adjusted_contour = contour + [self.roi[0], self.roi[1]]
                    cv2.drawContours(display_frame, [adjusted_contour], -1,
                                   display_config["contour_color"], display_config["contour_thickness"])
                else:
                    cv2.drawContours(display_frame, [contour], -1,
                                   display_config["contour_color"], display_config["contour_thickness"])

                # Add size info for small objects
                if area < 200:  # Show area for small detections
                    cv2.putText(display_frame, f"{int(area)}", (x, y-5),
                               display_config["font"], display_config["small_font_scale"],
                               display_config["text_color"], display_config["small_font_thickness"])

                motion_count += 1

        # Draw counting lines if enabled
        if (self.traffic_counter is not None and
            tracking_config["show_counting_lines"] and
            len(self.traffic_counter.counting_lines) > 0):

            for line in self.traffic_counter.counting_lines:
                x1, y1, x2, y2 = line
                # Adjust for ROI if set
                if self.roi_set and self.roi is not None:
                    roi_x, roi_y, _, _ = self.roi
                    x1 += roi_x
                    y1 += roi_y
                    x2 += roi_x
                    y2 += roi_y

                cv2.line(display_frame, (x1, y1), (x2, y2), (255, 0, 255), 2)  # Magenta line
                cv2.putText(display_frame, "COUNT LINE", (x1, y1-10),
                           display_config["font"], display_config["small_font_scale"],
                           (255, 0, 255), display_config["small_font_thickness"])

        # Add info text with FPS, tracking stats, and pause status
        current_config = config.get_current_config()

        # Status line with pause indicator
        status_text = f"Frame: {self.frame_count} | FPS: {self.current_fps:.1f}"
        if self.is_paused:
            status_text += " | PAUSED"

        # Get scaled font parameters
        font_scale, thickness = self.get_scaled_font_params()
        small_font_scale, small_thickness = self.get_scaled_font_params("small_font_scale", "small_font_thickness")

        # Main detection count
        cv2.putText(display_frame, f"Motion Objects: {motion_count}", (10, 30),
                   display_config["font"], font_scale,
                   display_config["text_color"], thickness)

        # Tracking statistics
        if self.tracker is not None:
            stats = self.tracker.get_statistics()
            tracking_text = f"Tracked: {stats['active_objects']} | Confirmed: {stats['confirmed_objects']}"
            cv2.putText(display_frame, tracking_text, (10, 60),
                       display_config["font"], small_font_scale,
                       display_config["text_color"], small_thickness)

            # Traffic counts
            if self.traffic_counter is not None:
                counts = self.traffic_counter.get_counts()
                count_text = f"Traffic Count: {counts['total']} (L:{counts['left']} R:{counts['right']})"
                cv2.putText(display_frame, count_text, (10, 80),
                           display_config["font"], small_font_scale,
                           display_config["text_color"], small_thickness)
                y_offset = 100
            else:
                y_offset = 80

            # ROI entry/exit counts
            if self.roi_tracker is not None and tracking_config["show_roi_counts"]:
                roi_counts = self.roi_tracker.get_counts()
                roi_text = f"ROI: In:{roi_counts['entries']} Out:{roi_counts['exits']} Current:{roi_counts['current_in_roi']}"
                cv2.putText(display_frame, roi_text, (10, y_offset),
                           display_config["font"], display_config["small_font_scale"],
                           display_config["text_color"], display_config["small_font_thickness"])

                # Show recent activity if enabled
                if tracking_config["show_recent_events"]:
                    recent = self.roi_tracker.get_recent_events(60)  # Last 60 seconds
                    recent_text = f"Recent: {recent['recent_entries']}/min in, {recent['recent_exits']}/min out"
                    cv2.putText(display_frame, recent_text, (10, y_offset + 20),
                               display_config["font"], display_config["small_font_scale"],
                               display_config["text_color"], display_config["small_font_thickness"])
                    y_offset += 40
                else:
                    y_offset += 20
        else:
            y_offset = 60

        cv2.putText(display_frame, status_text, (10, y_offset),
                   display_config["font"], font_scale,
                   display_config["text_color"], thickness)
        cv2.putText(display_frame, f"Min Size: {current_config['min_object_size']} px", (10, y_offset + 30),
                   display_config["font"], small_font_scale,
                   display_config["text_color"], small_thickness)
        cv2.putText(display_frame, "Press 'q' to quit, 'r' to reset ROI, 'c' to change preset", (10, y_offset + 50),
                   display_config["font"], small_font_scale,
                   display_config["text_color"], small_thickness)
        cv2.putText(display_frame, "Press SPACE to pause/resume, 't' to toggle tracking", (10, y_offset + 70),
                   display_config["font"], small_font_scale,
                   display_config["text_color"], small_thickness)
        cv2.putText(display_frame, "Press 'l' to set counting line, 'x' to reset ROI counters", (10, y_offset + 90),
                   display_config["font"], small_font_scale,
                   display_config["text_color"], small_thickness)

        return display_frame

    def analyze_detections(self, fg_mask):
        """Analyze all detections and return detailed information."""
        if fg_mask is None:
            return []

        # Find ALL contours for analysis
        all_contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detection_analysis = []

        for i, contour in enumerate(all_contours):
            area = cv2.contourArea(contour)
            x, y, w, h = cv2.boundingRect(contour)

            # Calculate shape metrics
            aspect_ratio = w / h if h > 0 else 0
            extent = area / (w * h) if w > 0 and h > 0 else 0

            # Determine status and reason
            if area < self.min_contour_area:
                status = "REJECTED"
                reason = "TOO_SMALL"
                color_name = "RED"
            elif area > self.max_contour_area:
                status = "REJECTED"
                reason = "TOO_LARGE"
                color_name = "BLUE"
            elif aspect_ratio < self.min_aspect_ratio:
                status = "REJECTED"
                reason = "ASPECT_RATIO_TOO_LOW"
                color_name = "ORANGE"
            elif aspect_ratio > self.max_aspect_ratio:
                status = "REJECTED"
                reason = "ASPECT_RATIO_TOO_HIGH"
                color_name = "ORANGE"
            elif extent < self.min_extent:
                status = "REJECTED"
                reason = "EXTENT_TOO_LOW"
                color_name = "ORANGE"
            else:
                status = "VALID"
                reason = "PASSED_ALL_FILTERS"
                color_name = "GREEN"

            # Adjust coordinates if ROI is set
            display_x, display_y = x, y
            if self.roi_set and self.roi is not None:
                roi_x, roi_y, _, _ = self.roi
                display_x += roi_x
                display_y += roi_y

            detection_info = {
                'id': i + 1,
                'status': status,
                'reason': reason,
                'color_name': color_name,
                'area': int(area),
                'bbox': (display_x, display_y, w, h),
                'aspect_ratio': round(aspect_ratio, 3),
                'extent': round(extent, 3),
                'contour': contour
            }

            detection_analysis.append(detection_info)

        return detection_analysis

    def print_detection_analysis(self, detection_analysis):
        """Print detailed detection analysis to terminal."""
        print("\n" + "="*80)
        print("PAUSED FRAME DETECTION ANALYSIS")
        print("="*80)

        if not detection_analysis:
            print("No motion detected in this frame.")
            return

        # Count by status
        valid_count = sum(1 for d in detection_analysis if d['status'] == 'VALID')
        rejected_count = len(detection_analysis) - valid_count

        print(f"Total Objects Detected: {len(detection_analysis)}")
        print(f"Valid Detections: {valid_count}")
        print(f"Rejected Detections: {rejected_count}")
        print()

        # Print each detection
        for detection in detection_analysis:
            status_symbol = "✅" if detection['status'] == 'VALID' else "❌"
            print(f"{status_symbol} Object #{detection['id']:2d} | {detection['color_name']:6s} | {detection['status']:8s}")
            print(f"   Reason: {detection['reason']}")
            print(f"   Area: {detection['area']:4d} pixels")
            print(f"   BBox: ({detection['bbox'][0]:3d},{detection['bbox'][1]:3d}) {detection['bbox'][2]:3d}x{detection['bbox'][3]:3d}")
            print(f"   Aspect Ratio: {detection['aspect_ratio']:5.3f} (range: {self.min_aspect_ratio}-{self.max_aspect_ratio})")
            print(f"   Extent: {detection['extent']:5.3f} (min: {self.min_extent})")
            print()

        # Print filter thresholds
        print("CURRENT FILTER SETTINGS:")
        print(f"  Size Range: {self.min_contour_area} - {self.max_contour_area} pixels")
        print(f"  Aspect Ratio: {self.min_aspect_ratio} - {self.max_aspect_ratio}")
        print(f"  Min Extent: {self.min_extent}")
        print("="*80)

    def create_debug_motion_mask(self, frame, fg_mask, _contours):
        """Create enhanced motion mask with bounding boxes for debugging."""
        # Create 3-channel mask for colored overlays
        if self.roi_set and self.roi is not None:
            # Full frame mask with ROI area filled
            mask_display = np.zeros_like(frame)
            x, y, w, h = self.roi

            # Convert single-channel mask to 3-channel
            fg_mask_3ch = cv2.cvtColor(fg_mask, cv2.COLOR_GRAY2BGR)
            mask_display[y:y+h, x:x+w] = fg_mask_3ch

            # Draw ROI boundary
            cv2.rectangle(mask_display, (x, y), (x+w, y+h), (255, 255, 0), 2)
            cv2.putText(mask_display, "ROI", (x, y-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        else:
            # Full frame mask
            mask_display = cv2.cvtColor(fg_mask, cv2.COLOR_GRAY2BGR)

        # Get detailed analysis
        detection_analysis = self.analyze_detections(fg_mask)

        # Draw all detections with analysis
        for detection in detection_analysis:
            x, y, w, h = detection['bbox']

            # Color mapping
            color_map = {
                'GREEN': (0, 255, 0),
                'RED': (0, 0, 255),
                'BLUE': (255, 0, 0),
                'ORANGE': (0, 165, 255)
            }
            color = color_map[detection['color_name']]

            # Draw bounding box
            cv2.rectangle(mask_display, (x, y), (x+w, y+h), color, 2)

            # Create detailed label
            if detection['status'] == 'VALID':
                label = f"#{detection['id']} VALID: {detection['area']}"
            else:
                label = f"#{detection['id']} {detection['reason']}: {detection['area']}"

            # Draw label
            cv2.putText(mask_display, label, (x, y-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

            # Draw contour outline
            if self.roi_set and self.roi is not None:
                # Adjust contour coordinates for display
                adjusted_contour = detection['contour'] + [self.roi[0], self.roi[1]]
                cv2.drawContours(mask_display, [adjusted_contour], -1, color, 1)
            else:
                cv2.drawContours(mask_display, [detection['contour']], -1, color, 1)

        # Add legend and analysis summary with scaled fonts
        legend_y = 30
        multiplier = config.DISPLAY_CONFIG["text_scale_multiplier"]
        legend_scale = 0.6 * multiplier
        legend_thickness = max(1, int(2 * multiplier))
        small_scale = 0.4 * multiplier
        small_thick = max(1, int(1 * multiplier))

        cv2.putText(mask_display, "LEGEND:", (10, legend_y),
                   cv2.FONT_HERSHEY_SIMPLEX, legend_scale, (255, 255, 255), legend_thickness)
        cv2.putText(mask_display, "GREEN = Valid detection", (10, legend_y + 25),
                   cv2.FONT_HERSHEY_SIMPLEX, small_scale, (0, 255, 0), small_thick)
        cv2.putText(mask_display, "RED = Too small", (10, legend_y + 45),
                   cv2.FONT_HERSHEY_SIMPLEX, small_scale, (0, 0, 255), small_thick)
        cv2.putText(mask_display, "BLUE = Too large", (10, legend_y + 65),
                   cv2.FONT_HERSHEY_SIMPLEX, small_scale, (255, 0, 0), small_thick)
        cv2.putText(mask_display, "ORANGE = Failed shape filter", (10, legend_y + 85),
                   cv2.FONT_HERSHEY_SIMPLEX, small_scale, (0, 165, 255), small_thick)

        # Add pause-specific information
        if self.is_paused:
            valid_count = sum(1 for d in detection_analysis if d['status'] == 'VALID')
            total_count = len(detection_analysis)
            pause_scale = 0.5 * multiplier
            pause_thick = max(1, int(2 * multiplier))
            cv2.putText(mask_display, f"PAUSED - Analysis: {valid_count}/{total_count} valid",
                       (10, legend_y + 110), cv2.FONT_HERSHEY_SIMPLEX, pause_scale, (255, 255, 0), pause_thick)

            # Print analysis to terminal (only once per pause)
            if not self.pause_analysis_printed:
                self.print_detection_analysis(detection_analysis)
                self.pause_analysis_printed = True

        # Add current filter settings
        cv2.putText(mask_display, f"Min Size: {self.min_contour_area} | Max Size: {self.max_contour_area}",
                   (10, legend_y + 135), cv2.FONT_HERSHEY_SIMPLEX, small_scale, (255, 255, 255), small_thick)
        cv2.putText(mask_display, f"Aspect Ratio: {self.min_aspect_ratio}-{self.max_aspect_ratio} | Min Extent: {self.min_extent}",
                   (10, legend_y + 155), cv2.FONT_HERSHEY_SIMPLEX, small_scale, (255, 255, 255), small_thick)

        return mask_display

    def create_combined_display(self, display_frame, fg_mask, contours, original_frame):
        """Create a combined display with main view and smaller subplots for mask and background."""
        # Get original frame dimensions
        main_height, main_width = display_frame.shape[:2]

        # Calculate subplot dimensions (half size)
        sub_height = main_height // 2
        sub_width = main_width // 2

        # Create combined canvas
        # Layout: [Main Detection View    ]
        #         [Motion Mask | Background]
        combined_height = main_height + sub_height
        combined_width = main_width
        combined_display = np.zeros((combined_height, combined_width, 3), dtype=np.uint8)

        # Place main detection view at top
        combined_display[0:main_height, 0:main_width] = display_frame

        # Create and place motion mask subplot (bottom left)
        if fg_mask is not None and config.DISPLAY_CONFIG["show_motion_mask"]:
            mask_display = self.create_debug_motion_mask(original_frame, fg_mask, contours)
            mask_resized = cv2.resize(mask_display, (sub_width, sub_height))
            combined_display[main_height:combined_height, 0:sub_width] = mask_resized

            # Add label with scaled font
            label_scale = 0.6 * config.DISPLAY_CONFIG["text_scale_multiplier"]
            label_thickness = max(1, int(2 * config.DISPLAY_CONFIG["text_scale_multiplier"]))
            cv2.putText(combined_display, "Motion Mask", (5, main_height + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, label_scale, (255, 255, 255), label_thickness)

        # Create and place background model subplot (bottom right)
        if config.DISPLAY_CONFIG["show_background_model"]:
            background_image = self.background_subtractor.getBackgroundImage()
            if background_image is not None:
                # Apply ROI to background if set
                if self.roi_set and self.roi is not None:
                    x, y, w, h = self.roi
                    bg_display = background_image.copy()
                    cv2.rectangle(bg_display, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    cv2.putText(bg_display, "ROI", (x, y-10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                else:
                    bg_display = background_image

                bg_resized = cv2.resize(bg_display, (sub_width, sub_height))
                combined_display[main_height:combined_height, sub_width:main_width] = bg_resized

                # Add label with scaled font
                cv2.putText(combined_display, "Background Model", (sub_width + 5, main_height + 20),
                           cv2.FONT_HERSHEY_SIMPLEX, label_scale, (255, 255, 255), label_thickness)

        return combined_display

    def get_scaled_font_params(self, font_scale_key="font_scale", thickness_key="font_thickness"):
        """Get font parameters scaled by the text_scale_multiplier."""
        display_config = config.DISPLAY_CONFIG
        multiplier = display_config["text_scale_multiplier"]

        scaled_font_scale = display_config[font_scale_key] * multiplier
        scaled_thickness = max(1, int(display_config[thickness_key] * multiplier))

        return scaled_font_scale, scaled_thickness

    def update_fps(self):
        """Update FPS calculation."""
        current_time = time.time()
        self.fps_counter += 1

        # Calculate FPS every second
        if current_time - self.fps_start_time >= 1.0:
            self.current_fps = self.fps_counter / (current_time - self.fps_start_time)
            self.fps_counter = 0
            self.fps_start_time = current_time
    
    def run(self):
        """Main detection loop."""
        print("Starting Motion-Based Traffic Detection with Object Tracking")
        print("Controls:")
        print("  'q' - Quit")
        print("  'r' - Reset/Set ROI")
        print("  's' - Save current frame")
        print("  'c' - Cycle through detection presets")
        print("  '1-4' - Switch to specific preset")
        print("  't' - Toggle object tracking")
        print("  'l' - Set counting line")
        print("  'x' - Reset ROI and traffic counters")
        print("  SPACE - Pause/Resume")
        print()

        webcam_available = False
        preset_names = list(config.DETECTION_PRESETS.keys())
        current_preset_index = preset_names.index(config.ACTIVE_PRESET)

        while True:
            # Handle pause/resume logic
            if not self.is_paused:
                # Get new frame only when not paused
                frame = self.get_frame_from_webcam()
                if frame is None:
                    if not webcam_available:
                        print("Webcam not available, using demo mode")
                        webcam_available = False
                    frame = self.create_demo_frame()
                else:
                    if not webcam_available:
                        print("Webcam connected successfully!")
                        webcam_available = True

                self.frame_count += 1

                # Update FPS calculation
                self.update_fps()

                # Set ROI on first frame or when requested
                if not self.roi_set and webcam_available:
                    self.set_roi_interactive(frame)

                # Skip processing if frame skipping is enabled
                if config.PERFORMANCE["frame_skip"] > 1 and self.frame_count % config.PERFORMANCE["frame_skip"] != 0:
                    continue

                # Detect motion
                fg_mask, contours, tracked_objects = self.detect_motion(frame)

                # Store current frame data for pause functionality
                self.paused_frame = frame.copy()
                self.paused_fg_mask = fg_mask.copy() if fg_mask is not None else None
                self.paused_contours = contours.copy()
                self.paused_tracked_objects = tracked_objects.copy() if tracked_objects else []
            else:
                # Use paused frame data
                frame = self.paused_frame
                fg_mask = self.paused_fg_mask
                contours = self.paused_contours
                tracked_objects = self.paused_tracked_objects

            # Draw results (always update display even when paused)
            display_frame = self.draw_detections(frame, fg_mask, contours, tracked_objects)

            # Create combined display with subplots
            combined_display = self.create_combined_display(display_frame, fg_mask, contours, frame)
            cv2.imshow("Motion Traffic Detection - Combined View", combined_display)

            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord(' '):  # Spacebar for pause/resume
                self.is_paused = not self.is_paused
                if self.is_paused:
                    print("PAUSED - Press SPACE to resume")
                    self.pause_analysis_printed = False  # Reset analysis flag for new pause
                else:
                    print("RESUMED")
            elif key == ord('r'):
                self.roi_set = False
                print("ROI reset - will be set on next frame")
            elif key == ord('s'):
                timestamp = datetime.now().strftime(config.OUTPUT_CONFIG["timestamp_format"])
                filename = f"motion_detection_{timestamp}.jpg"
                cv2.imwrite(filename, display_frame)
                print(f"Saved frame: {filename}")
            elif key == ord('c'):
                # Cycle through presets
                current_preset_index = (current_preset_index + 1) % len(preset_names)
                new_preset = preset_names[current_preset_index]
                config.apply_preset(new_preset)
                self._update_detection_params()
                print(f"Switched to preset: {new_preset}")
            elif key in [ord('1'), ord('2'), ord('3'), ord('4')]:
                # Direct preset selection
                preset_index = int(chr(key)) - 1
                if 0 <= preset_index < len(preset_names):
                    current_preset_index = preset_index
                    new_preset = preset_names[preset_index]
                    config.apply_preset(new_preset)
                    self._update_detection_params()
                    print(f"Switched to preset: {new_preset}")
            elif key == ord('t'):
                # Toggle tracking
                if self.tracker is not None:
                    config.TRACKING_CONFIG["enable_tracking"] = not config.TRACKING_CONFIG["enable_tracking"]
                    if not config.TRACKING_CONFIG["enable_tracking"]:
                        self.tracker.reset()
                    print(f"Tracking {'enabled' if config.TRACKING_CONFIG['enable_tracking'] else 'disabled'}")
            elif key == ord('l'):
                # Set counting line
                if frame is not None:
                    self.set_counting_line_interactive(frame)
            elif key == ord('x'):
                # Reset ROI counters
                if self.roi_tracker is not None:
                    self.roi_tracker.reset()
                    print("ROI entry/exit counters reset")
                if self.traffic_counter is not None:
                    self.traffic_counter.reset_counts()
                    print("Traffic counters reset")

            # Adaptive frame rate control (only when not paused)
            if not self.is_paused:
                target_fps = config.PERFORMANCE["target_fps"]
                if target_fps > 0:
                    frame_time = time.time() - self.last_frame_time
                    target_frame_time = 1.0 / target_fps
                    if frame_time < target_frame_time:
                        time.sleep(target_frame_time - frame_time)
                self.last_frame_time = time.time()
            else:
                # When paused, just a small delay to prevent excessive CPU usage
                time.sleep(0.05)

        # Cleanup
        self.cleanup()
        cv2.destroyAllWindows()
        print(f"Motion detection stopped. Final FPS: {self.current_fps:.1f}")

    def _update_detection_params(self):
        """Update detection parameters from config after preset change."""
        motion_config = config.MOTION_DETECTION
        self.min_contour_area = motion_config["min_contour_area"]
        self.max_contour_area = motion_config["max_contour_area"]

        # Update background subtractor parameters
        bg_config = config.BACKGROUND_SUBTRACTOR_CONFIG
        self.background_subtractor.setVarThreshold(bg_config["varThreshold"])

        # Update tracking parameters
        if self.tracker is not None:
            tracking_config = config.TRACKING_CONFIG
            self.tracker.max_distance = tracking_config["max_distance"]
            self.tracker.max_disappeared = tracking_config["max_disappeared"]
            self.tracker.min_detection_frames = tracking_config["min_detection_frames"]

def main():
    detector = MotionTrafficDetector()
    detector.run()

if __name__ == "__main__":
    main()
