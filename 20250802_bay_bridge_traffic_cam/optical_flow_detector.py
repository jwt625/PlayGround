#!/usr/bin/env python3
"""
Fresh Optical Flow-based Traffic Detection for Bay Bridge
Designed specifically for heavily occluded bridge traffic where 80% of vehicles may be hidden.
Uses dense optical flow to detect motion patterns of partially visible vehicles.
"""

import cv2
import numpy as np
import requests
import time
from datetime import datetime

class OpticalFlowTrafficDetector:
    def __init__(self, webcam_url="http://192.168.12.6:4747/video"):
        self.webcam_url = webcam_url
        
        # Optical flow parameters - tuned for bridge traffic
        self.flow_threshold = 5.0  # Minimum flow magnitude for motion
        self.min_motion_area = 200  # Minimum area of motion region
        self.max_motion_area = 3000  # Maximum area to avoid noise
        self.coherence_threshold = 0.7  # Motion direction consistency
        
        # Dense optical flow parameters (Farneback)
        self.flow_params = {
            'pyr_scale': 0.5,
            'levels': 3,
            'winsize': 15,
            'iterations': 3,
            'poly_n': 5,
            'poly_sigma': 1.2,
            'flags': 0
        }
        
        # Frame processing
        self.prev_gray = None
        self.frame_count = 0
        self.roi = None
        
        # Traffic counting
        self.counting_line = None
        self.vehicle_count = 0
        self.tracked_objects = {}
        self.next_id = 1

        # Debug visualization
        self.show_debug = True
        self.debug_windows = {
            'flow_vectors': True,
            'motion_mask': True,
            'magnitude': True,
            'coherence': True
        }
        
        # Performance
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.current_fps = 0

        # Persistent video capture for better performance
        self.cap = None
        self.cap_initialized = False
        
    def get_frame(self):
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
                    return self.get_frame()  # Retry once

            # Fallback to HTTP request method (only if OpenCV fails)
            response = requests.get(self.webcam_url, stream=True, timeout=5)
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
    
    def set_roi_interactive(self, frame):
        """Set region of interest for bridge deck."""
        print("Select the bridge deck area by dragging a rectangle, then press ENTER")
        roi = cv2.selectROI("Select Bridge Deck ROI", frame, False, False)
        cv2.destroyWindow("Select Bridge Deck ROI")
        
        if roi[2] > 0 and roi[3] > 0:
            self.roi = roi
            print(f"ROI set: x={roi[0]}, y={roi[1]}, w={roi[2]}, h={roi[3]}")
        else:
            print("No ROI selected, using full frame")
            self.roi = None
    
    def set_counting_line_interactive(self, frame):
        """Set counting line for traffic counting."""
        print("Click two points to set a counting line")
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
            for i, point in enumerate(points):
                cv2.circle(display_frame, point, 5, (0, 255, 0), -1)
            cv2.putText(display_frame, f"Click point {len(points)+1} of 2", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.imshow("Set Counting Line", display_frame)
            cv2.waitKey(1)
        
        if len(points) == 2:
            self.counting_line = points
            print(f"Counting line set: {points}")
        
        cv2.destroyWindow("Set Counting Line")
    
    def detect_motion_flow(self, frame):
        """Detect motion using optical flow."""
        if frame is None:
            return [], np.zeros_like(frame[:,:,0]) if frame is not None else None
        
        # Apply ROI if set
        if self.roi:
            x, y, w, h = self.roi
            roi_frame = frame[y:y+h, x:x+w]
        else:
            roi_frame = frame
        
        # Convert to grayscale
        gray = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)
        
        # Initialize on first frame
        if self.prev_gray is None:
            self.prev_gray = gray.copy()
            return [], np.zeros_like(gray)
        
        # Calculate optical flow
        flow = cv2.calcOpticalFlowFarneback(
            self.prev_gray, gray, None,
            self.flow_params['pyr_scale'],
            self.flow_params['levels'],
            self.flow_params['winsize'],
            self.flow_params['iterations'],
            self.flow_params['poly_n'],
            self.flow_params['poly_sigma'],
            self.flow_params['flags']
        )
        
        # Calculate flow magnitude
        magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        
        # Create motion mask
        motion_mask = (magnitude > self.flow_threshold).astype(np.uint8) * 255
        
        # Clean up mask with morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_OPEN, kernel)
        motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_CLOSE, kernel)
        
        # Find motion regions
        contours, _ = cv2.findContours(motion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours by size and motion coherence
        valid_motions = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if self.min_motion_area <= area <= self.max_motion_area:
                # Check motion coherence
                mask = np.zeros_like(motion_mask)
                cv2.drawContours(mask, [contour], -1, 255, -1)
                
                # Get flow vectors in this region
                region_flow_x = flow[..., 0][mask > 0]
                region_flow_y = flow[..., 1][mask > 0]
                
                if len(region_flow_x) > 0:
                    # Calculate direction consistency
                    directions = np.arctan2(region_flow_y, region_flow_x)
                    mean_direction = np.arctan2(np.mean(region_flow_y), np.mean(region_flow_x))
                    
                    # Calculate coherence
                    angle_diffs = np.abs(directions - mean_direction)
                    angle_diffs = np.minimum(angle_diffs, 2*np.pi - angle_diffs)
                    coherence = 1.0 - (np.mean(angle_diffs) / np.pi)
                    
                    if coherence >= self.coherence_threshold:
                        # Adjust contour coordinates if ROI is set
                        if self.roi:
                            contour = contour + [self.roi[0], self.roi[1]]
                        valid_motions.append({
                            'contour': contour,
                            'area': area,
                            'coherence': coherence,
                            'flow': (np.mean(region_flow_x), np.mean(region_flow_y))
                        })
        
        self.prev_gray = gray.copy()

        # Store debug data
        if self.show_debug:
            self.debug_data = {
                'flow': flow,
                'magnitude': magnitude,
                'angle': angle,
                'motion_mask': motion_mask,
                'valid_motions': valid_motions,
                'roi_frame': roi_frame
            }

        return valid_motions, motion_mask
    
    def simple_tracking(self, motions):
        """Simple tracking of motion regions."""
        current_objects = {}
        
        for motion in motions:
            contour = motion['contour']
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                centroid = (cx, cy)
                
                # Find closest existing object
                min_dist = float('inf')
                closest_id = None
                
                for obj_id, obj_data in self.tracked_objects.items():
                    dist = np.sqrt((centroid[0] - obj_data['centroid'][0])**2 + 
                                 (centroid[1] - obj_data['centroid'][1])**2)
                    if dist < min_dist and dist < 100:  # Max association distance
                        min_dist = dist
                        closest_id = obj_id
                
                if closest_id:
                    # Update existing object
                    self.tracked_objects[closest_id]['centroid'] = centroid
                    self.tracked_objects[closest_id]['contour'] = contour
                    self.tracked_objects[closest_id]['age'] += 1
                    current_objects[closest_id] = self.tracked_objects[closest_id]
                else:
                    # Create new object
                    new_id = self.next_id
                    self.next_id += 1
                    current_objects[new_id] = {
                        'centroid': centroid,
                        'contour': contour,
                        'age': 1,
                        'counted': False
                    }
        
        # Update tracked objects (remove old ones)
        self.tracked_objects = current_objects
        return list(current_objects.values())
    
    def check_line_crossing(self, objects):
        """Check if objects cross the counting line."""
        if not self.counting_line:
            return
        
        line_start, line_end = self.counting_line
        
        for obj_id, obj_data in self.tracked_objects.items():
            if obj_data['counted'] or obj_data['age'] < 3:
                continue
            
            centroid = obj_data['centroid']
            
            # Simple line crossing check (crossing vertical line)
            if abs(centroid[0] - line_start[0]) < 10:  # Near the line
                self.tracked_objects[obj_id]['counted'] = True
                self.vehicle_count += 1
                print(f"🚗 Vehicle {obj_id} crossed counting line! Total: {self.vehicle_count}")
    
    def update_fps(self):
        """Update FPS calculation."""
        self.fps_counter += 1
        current_time = time.time()
        if current_time - self.fps_start_time >= 1.0:
            self.current_fps = self.fps_counter / (current_time - self.fps_start_time)
            self.fps_counter = 0
            self.fps_start_time = current_time
    
    def draw_results(self, frame, motions, objects):
        """Draw detection results on frame."""
        display_frame = frame.copy()
        
        # Draw ROI
        if self.roi:
            x, y, w, h = self.roi
            cv2.rectangle(display_frame, (x, y), (x+w, y+h), (255, 255, 0), 2)
        
        # Draw counting line
        if self.counting_line:
            cv2.line(display_frame, self.counting_line[0], self.counting_line[1], (255, 0, 255), 3)
            cv2.putText(display_frame, "COUNT LINE", 
                       (self.counting_line[0][0], self.counting_line[0][1]-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
        
        # Draw tracked objects
        for obj in objects:
            centroid = obj['centroid']
            contour = obj['contour']
            
            # Draw bounding box
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Draw centroid
            cv2.circle(display_frame, centroid, 5, (0, 0, 255), -1)
        
        # Draw info
        cv2.putText(display_frame, f"Frame: {self.frame_count} | FPS: {self.current_fps:.1f}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(display_frame, f"Objects: {len(objects)} | Count: {self.vehicle_count}", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(display_frame, "Press 'q' to quit, 'r' for ROI, 'l' for line", 
                   (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return display_frame

    def create_flow_visualization(self, flow, magnitude):
        """Create optical flow visualization with arrows."""
        h, w = flow.shape[:2]

        # Create HSV image for flow direction
        hsv = np.zeros((h, w, 3), dtype=np.uint8)
        hsv[..., 1] = 255  # Full saturation

        # Map angle to hue, magnitude to value
        magnitude_norm, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        hsv[..., 0] = angle * 180 / np.pi / 2
        hsv[..., 2] = cv2.normalize(magnitude_norm, None, 0, 255, cv2.NORM_MINMAX)

        # Convert to BGR
        flow_bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        # Draw flow vectors as arrows (subsample for visibility)
        step = 20
        for y in range(0, h, step):
            for x in range(0, w, step):
                if magnitude[y, x] > self.flow_threshold:
                    # Draw arrow
                    dx, dy = flow[y, x]
                    end_x = int(x + dx * 3)  # Scale for visibility
                    end_y = int(y + dy * 3)

                    # Ensure arrow end is within bounds
                    end_x = max(0, min(w-1, end_x))
                    end_y = max(0, min(h-1, end_y))

                    cv2.arrowedLine(flow_bgr, (x, y), (end_x, end_y), (255, 255, 255), 1, tipLength=0.3)

        return flow_bgr

    def create_magnitude_visualization(self, magnitude):
        """Create magnitude visualization."""
        # Normalize magnitude for display
        mag_norm = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        # Apply colormap
        mag_colored = cv2.applyColorMap(mag_norm, cv2.COLORMAP_JET)

        # Add threshold line
        threshold_mask = (magnitude > self.flow_threshold).astype(np.uint8) * 255
        mag_colored[threshold_mask > 0] = [0, 255, 0]  # Green for above threshold

        return mag_colored

    def create_motion_regions_visualization(self, frame, valid_motions):
        """Visualize detected motion regions with detailed info."""
        debug_frame = frame.copy()

        for i, motion in enumerate(valid_motions):
            contour = motion['contour']
            area = motion['area']
            coherence = motion['coherence']
            flow_vec = motion['flow']

            # Draw contour
            cv2.drawContours(debug_frame, [contour], -1, (0, 255, 0), 2)

            # Get bounding box
            x, y, w, h = cv2.boundingRect(contour)

            # Draw bounding box
            cv2.rectangle(debug_frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

            # Add detailed labels
            cv2.putText(debug_frame, f"ID:{i}", (x, y-40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            cv2.putText(debug_frame, f"Area:{int(area)}", (x, y-25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
            cv2.putText(debug_frame, f"Coh:{coherence:.2f}", (x, y-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
            cv2.putText(debug_frame, f"Flow:({flow_vec[0]:.1f},{flow_vec[1]:.1f})", (x, y+h+15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)

            # Draw flow vector arrow from center
            center_x = x + w // 2
            center_y = y + h // 2
            end_x = int(center_x + flow_vec[0] * 5)
            end_y = int(center_y + flow_vec[1] * 5)
            cv2.arrowedLine(debug_frame, (center_x, center_y), (end_x, end_y), (0, 0, 255), 2, tipLength=0.3)

        # Add summary info
        cv2.putText(debug_frame, f"Valid Motion Regions: {len(valid_motions)}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(debug_frame, f"Flow Threshold: {self.flow_threshold}", (10, 55),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(debug_frame, f"Coherence Threshold: {self.coherence_threshold}", (10, 75),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        return debug_frame

    def show_debug_windows(self):
        """Show all debug visualization windows."""
        if not self.show_debug or not hasattr(self, 'debug_data'):
            return

        debug_data = self.debug_data

        # Flow vectors visualization
        if self.debug_windows['flow_vectors']:
            flow_vis = self.create_flow_visualization(debug_data['flow'], debug_data['magnitude'])
            cv2.imshow("Debug: Flow Vectors", flow_vis)

        # Magnitude visualization
        if self.debug_windows['magnitude']:
            mag_vis = self.create_magnitude_visualization(debug_data['magnitude'])
            cv2.imshow("Debug: Flow Magnitude", mag_vis)

        # Motion mask
        if self.debug_windows['motion_mask']:
            cv2.imshow("Debug: Motion Mask", debug_data['motion_mask'])

        # Motion regions with details
        if self.debug_windows['coherence']:
            regions_vis = self.create_motion_regions_visualization(debug_data['roi_frame'], debug_data['valid_motions'])
            cv2.imshow("Debug: Motion Regions", regions_vis)

    def run(self):
        """Main detection loop."""
        print("Starting Optical Flow Traffic Detection")
        print("Controls:")
        print("  'q' - Quit")
        print("  'r' - Set ROI")
        print("  'l' - Set counting line")
        print("  's' - Save frame")
        print("  'd' - Toggle debug windows")
        print("  '1' - Toggle flow vectors")
        print("  '2' - Toggle magnitude")
        print("  '3' - Toggle motion mask")
        print("  '4' - Toggle motion regions")
        print()

        # Get first frame and set ROI
        frame = self.get_frame()
        if frame is not None:
            self.set_roi_interactive(frame)
            self.set_counting_line_interactive(frame)

        while True:
            frame = self.get_frame()
            if frame is None:
                print("No frame received, retrying...")
                time.sleep(1)
                continue

            self.frame_count += 1
            self.update_fps()

            # Detect motion using optical flow
            motions, motion_mask = self.detect_motion_flow(frame)

            # Track objects
            objects = self.simple_tracking(motions)

            # Check line crossings
            self.check_line_crossing(objects)

            # Draw results
            display_frame = self.draw_results(frame, motions, objects)

            # Show debug windows
            self.show_debug_windows()

            # Show main display
            cv2.imshow("Optical Flow Traffic Detection", display_frame)

            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                self.set_roi_interactive(frame)
            elif key == ord('l'):
                self.set_counting_line_interactive(frame)
            elif key == ord('s'):
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"optical_flow_detection_{timestamp}.jpg"
                cv2.imwrite(filename, display_frame)
                print(f"Saved frame: {filename}")
            elif key == ord('d'):
                self.show_debug = not self.show_debug
                print(f"Debug windows: {'ON' if self.show_debug else 'OFF'}")
                if not self.show_debug:
                    cv2.destroyWindow("Debug: Flow Vectors")
                    cv2.destroyWindow("Debug: Flow Magnitude")
                    cv2.destroyWindow("Debug: Motion Mask")
                    cv2.destroyWindow("Debug: Motion Regions")
            elif key == ord('1'):
                self.debug_windows['flow_vectors'] = not self.debug_windows['flow_vectors']
                print(f"Flow vectors: {'ON' if self.debug_windows['flow_vectors'] else 'OFF'}")
                if not self.debug_windows['flow_vectors']:
                    cv2.destroyWindow("Debug: Flow Vectors")
            elif key == ord('2'):
                self.debug_windows['magnitude'] = not self.debug_windows['magnitude']
                print(f"Magnitude: {'ON' if self.debug_windows['magnitude'] else 'OFF'}")
                if not self.debug_windows['magnitude']:
                    cv2.destroyWindow("Debug: Flow Magnitude")
            elif key == ord('3'):
                self.debug_windows['motion_mask'] = not self.debug_windows['motion_mask']
                print(f"Motion mask: {'ON' if self.debug_windows['motion_mask'] else 'OFF'}")
                if not self.debug_windows['motion_mask']:
                    cv2.destroyWindow("Debug: Motion Mask")
            elif key == ord('4'):
                self.debug_windows['coherence'] = not self.debug_windows['coherence']
                print(f"Motion regions: {'ON' if self.debug_windows['coherence'] else 'OFF'}")
                if not self.debug_windows['coherence']:
                    cv2.destroyWindow("Debug: Motion Regions")

        self.cleanup()
        cv2.destroyAllWindows()
        print(f"Detection stopped. Final count: {self.vehicle_count}")

def main():
    detector = OpticalFlowTrafficDetector()
    detector.run()

if __name__ == "__main__":
    main()
