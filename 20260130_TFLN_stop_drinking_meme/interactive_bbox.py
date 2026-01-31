#!/usr/bin/env python3
"""
Interactive bbox editor for caption segments.

For each segment:
- Shows frame with detected bbox (red)
- Press ENTER to accept current bbox
- Or click and drag to draw your own bbox (green)
- Press 'r' to reset to detected bbox
- Press 'q' to quit and save

Final bboxes saved to final_bboxes.json
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.widgets import RectangleSelector
from PIL import Image
import json
import numpy as np
from pathlib import Path

# Load segments from v3 detection
SEGMENTS_FILE = "segments_v3.txt"
FRAMES_DIR = Path("debug_frames_v3")
OUTPUT_FILE = "final_bboxes.json"

def load_segments():
    """Load detected segments."""
    segments = []
    with open(SEGMENTS_FILE) as f:
        for line in f:
            if line.startswith('#'):
                continue
            parts = line.strip().split(',')
            if len(parts) == 6:
                segments.append({
                    'start_time': float(parts[0]),
                    'end_time': float(parts[1]),
                    'x_min': int(parts[2]),
                    'y_min': int(parts[3]),
                    'x_max': int(parts[4]),
                    'y_max': int(parts[5]),
                })
    return segments

def get_frame_for_segment(seg):
    """Get middle frame for a segment."""
    mid_time = (seg['start_time'] + seg['end_time']) / 2
    frame_num = int(mid_time * 30) + 1  # 1-indexed
    frame_path = FRAMES_DIR / f"frame_{frame_num:05d}.jpg"
    return frame_path, frame_num

class BBoxEditor:
    def __init__(self, segments):
        self.segments = segments
        self.current_idx = 0
        self.final_bboxes = []
        self.current_bbox = None
        self.user_bbox = None

        # Setup figure
        self.fig, self.ax = plt.subplots(1, 1, figsize=(12, 9))
        self.fig.canvas.manager.set_window_title('Caption BBox Editor')

        # Rectangle selector for drawing
        self.selector = RectangleSelector(
            self.ax, self.on_select,
            useblit=True,
            button=[1],
            minspanx=5, minspany=5,
            spancoords='pixels',
            interactive=True
        )

        # Key bindings
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)

        # Load first segment
        self.load_segment(0)

    def load_segment(self, idx):
        """Load and display a segment."""
        if idx >= len(self.segments):
            self.save_and_quit()
            return

        self.current_idx = idx
        seg = self.segments[idx]

        # Get frame
        frame_path, frame_num = get_frame_for_segment(seg)
        img = Image.open(frame_path)

        # Clear and show image
        self.ax.clear()
        self.ax.imshow(img)
        self.ax.set_title(
            f"Segment {idx+1}/{len(self.segments)} | "
            f"Time: {seg['start_time']:.2f}s - {seg['end_time']:.2f}s\n"
            f"ENTER=accept | Click+drag=draw new | R=reset | Q=quit+save",
            fontsize=12
        )

        # Store current detected bbox
        self.current_bbox = (seg['x_min'], seg['y_min'], seg['x_max'], seg['y_max'])
        self.user_bbox = None

        # Draw detected bbox in red
        self.draw_bbox(self.current_bbox, 'red', 'Detected')

        self.fig.canvas.draw()

    def draw_bbox(self, bbox, color, label):
        """Draw a bounding box."""
        x1, y1, x2, y2 = bbox
        rect = patches.Rectangle(
            (x1, y1), x2-x1, y2-y1,
            linewidth=2, edgecolor=color, facecolor='none',
            label=label
        )
        self.ax.add_patch(rect)
        self.ax.legend(loc='upper right')

    def on_select(self, eclick, erelease):
        """Handle rectangle selection."""
        x1, y1 = int(eclick.xdata), int(eclick.ydata)
        x2, y2 = int(erelease.xdata), int(erelease.ydata)

        # Ensure proper ordering
        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)

        self.user_bbox = (x1, y1, x2, y2)

        # Redraw with both boxes
        seg = self.segments[self.current_idx]
        frame_path, _ = get_frame_for_segment(seg)
        img = Image.open(frame_path)

        self.ax.clear()
        self.ax.imshow(img)
        self.ax.set_title(
            f"Segment {self.current_idx+1}/{len(self.segments)} | "
            f"Time: {seg['start_time']:.2f}s - {seg['end_time']:.2f}s\n"
            f"ENTER=accept GREEN | R=reset to RED | Q=quit+save",
            fontsize=12
        )

        # Draw detected in red (dimmed)
        self.draw_bbox(self.current_bbox, 'red', 'Detected')
        # Draw user bbox in green
        self.draw_bbox(self.user_bbox, 'lime', 'YOUR BBOX')

        self.fig.canvas.draw()

    def on_key(self, event):
        """Handle key presses."""
        if event.key == 'enter':
            # Accept current bbox (user's if drawn, else detected)
            bbox = self.user_bbox if self.user_bbox else self.current_bbox
            seg = self.segments[self.current_idx]

            self.final_bboxes.append({
                'segment': self.current_idx + 1,
                'start_time': seg['start_time'],
                'end_time': seg['end_time'],
                'x_min': bbox[0],
                'y_min': bbox[1],
                'x_max': bbox[2],
                'y_max': bbox[3],
                'source': 'user' if self.user_bbox else 'detected'
            })

            print(f"Segment {self.current_idx+1}: Accepted {'USER' if self.user_bbox else 'DETECTED'} bbox: {bbox}")

            # Next segment
            self.load_segment(self.current_idx + 1)

        elif event.key == 'r':
            # Reset to detected bbox
            self.user_bbox = None
            self.load_segment(self.current_idx)

        elif event.key == 'q':
            self.save_and_quit()

    def save_and_quit(self):
        """Save results and quit."""
        # Save any remaining segments with detected bbox
        while self.current_idx < len(self.segments):
            seg = self.segments[self.current_idx]
            self.final_bboxes.append({
                'segment': self.current_idx + 1,
                'start_time': seg['start_time'],
                'end_time': seg['end_time'],
                'x_min': seg['x_min'],
                'y_min': seg['y_min'],
                'x_max': seg['x_max'],
                'y_max': seg['y_max'],
                'source': 'detected'
            })
            self.current_idx += 1

        # Save to file
        with open(OUTPUT_FILE, 'w') as f:
            json.dump(self.final_bboxes, f, indent=2)

        print(f"\nSaved {len(self.final_bboxes)} bboxes to {OUTPUT_FILE}")
        plt.close()

    def run(self):
        """Run the editor."""
        plt.show()

def main():
    print("Loading segments...")
    segments = load_segments()
    print(f"Found {len(segments)} segments")

    print("\nControls:")
    print("  ENTER  - Accept current bbox (green if drawn, else red)")
    print("  Click+Drag - Draw your own bbox")
    print("  R      - Reset to detected bbox")
    print("  Q      - Quit and save all\n")

    editor = BBoxEditor(segments)
    editor.run()

if __name__ == "__main__":
    main()
