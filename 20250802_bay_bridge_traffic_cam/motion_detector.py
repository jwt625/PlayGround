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

        # Print configuration on startup
        config.print_config_summary()
        
    def get_frame_from_webcam(self):
        """Capture frame from iPhone webcam with fallback methods."""
        try:
            # Try OpenCV first
            cap = cv2.VideoCapture(self.webcam_url)
            if cap.isOpened():
                ret, frame = cap.read()
                cap.release()
                if ret:
                    return frame

            # Fallback to HTTP request method
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

        cv2.putText(frame, "DEMO MODE - Moving objects", (10, 30),
                   config.DISPLAY_CONFIG["font"], config.DISPLAY_CONFIG["font_scale"],
                   config.DISPLAY_CONFIG["text_color"], config.DISPLAY_CONFIG["font_thickness"])

        return frame
    
    def set_roi_interactive(self, frame):
        """Let user select Region of Interest (bridge deck area)."""
        print("Select the bridge deck area by dragging a rectangle, then press ENTER")
        print("Press 'r' to reset selection, 'q' to quit")
        
        roi = cv2.selectROI("Select Bridge Deck ROI", frame, False, False)
        cv2.destroyWindow("Select Bridge Deck ROI")
        
        if roi[2] > 0 and roi[3] > 0:  # Valid selection
            self.roi = roi
            self.roi_set = True
            print(f"ROI set: x={roi[0]}, y={roi[1]}, w={roi[2]}, h={roi[3]}")
        else:
            print("No ROI selected, using full frame")
            self.roi = None
    
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

        return fg_mask, valid_contours
    
    def draw_detections(self, frame, fg_mask, contours):
        """Draw motion detections on the frame."""
        display_frame = frame.copy()
        display_config = config.DISPLAY_CONFIG

        # Draw ROI if set
        if self.roi_set and self.roi is not None:
            x, y, w, h = self.roi
            cv2.rectangle(display_frame, (x, y), (x+w, y+h),
                         display_config["roi_color"], display_config["roi_thickness"])
            cv2.putText(display_frame, "Bridge Deck ROI", (x, y-10),
                       display_config["font"], display_config["small_font_scale"],
                       display_config["roi_color"], display_config["small_font_thickness"])

        # Draw motion contours
        motion_count = 0
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

        # Add info text
        current_config = config.get_current_config()
        cv2.putText(display_frame, f"Motion Objects: {motion_count}", (10, 30),
                   display_config["font"], display_config["font_scale"],
                   display_config["text_color"], display_config["font_thickness"])
        cv2.putText(display_frame, f"Frame: {self.frame_count}", (10, 60),
                   display_config["font"], display_config["font_scale"],
                   display_config["text_color"], display_config["font_thickness"])
        cv2.putText(display_frame, f"Min Size: {current_config['min_object_size']} px", (10, 90),
                   display_config["font"], display_config["small_font_scale"],
                   display_config["text_color"], display_config["small_font_thickness"])
        cv2.putText(display_frame, "Press 'q' to quit, 'r' to reset ROI, 'c' to change preset", (10, 120),
                   display_config["font"], display_config["small_font_scale"],
                   display_config["text_color"], display_config["small_font_thickness"])

        return display_frame
    
    def run(self):
        """Main detection loop."""
        print("Starting Motion-Based Traffic Detection")
        print("Controls:")
        print("  'q' - Quit")
        print("  'r' - Reset/Set ROI")
        print("  's' - Save current frame")
        print("  'c' - Cycle through detection presets")
        print("  '1-4' - Switch to specific preset")
        print()

        webcam_available = False
        preset_names = list(config.DETECTION_PRESETS.keys())
        current_preset_index = preset_names.index(config.ACTIVE_PRESET)

        while True:
            # Get frame
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

            # Set ROI on first frame or when requested
            if not self.roi_set and webcam_available:
                self.set_roi_interactive(frame)

            # Detect motion
            fg_mask, contours = self.detect_motion(frame)

            # Draw results
            display_frame = self.draw_detections(frame, fg_mask, contours)

            # Show frames
            cv2.imshow("Motion Traffic Detection", display_frame)

            if fg_mask is not None and config.DISPLAY_CONFIG["show_motion_mask"]:
                # Resize mask for display
                if self.roi_set and self.roi is not None:
                    mask_display = np.zeros_like(frame[:,:,0])
                    x, y, w, h = self.roi
                    mask_display[y:y+h, x:x+w] = fg_mask
                else:
                    mask_display = fg_mask
                cv2.imshow("Motion Mask", mask_display)

            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
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

            # Small delay to prevent excessive CPU usage
            target_fps = config.PERFORMANCE["target_fps"]
            time.sleep(1.0 / target_fps if target_fps > 0 else 0.03)

        cv2.destroyAllWindows()
        print("Motion detection stopped")

    def _update_detection_params(self):
        """Update detection parameters from config after preset change."""
        motion_config = config.MOTION_DETECTION
        self.min_contour_area = motion_config["min_contour_area"]
        self.max_contour_area = motion_config["max_contour_area"]

        # Update background subtractor parameters
        bg_config = config.BACKGROUND_SUBTRACTOR_CONFIG
        self.background_subtractor.setVarThreshold(bg_config["varThreshold"])

def main():
    detector = MotionTrafficDetector()
    detector.run()

if __name__ == "__main__":
    main()
