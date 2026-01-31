#!/usr/bin/env python3
"""
Verify/edit saved bboxes.

Controls:
- ENTER = Accept (green if drawn, else current) and go next
- Click+Drag = Draw new bbox
- R = Reset to saved bbox
- LEFT = Go back
- Q = Quit and save
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.widgets import RectangleSelector
from PIL import Image
import json
from pathlib import Path

BBOXES_FILE = "final_bboxes.json"
FRAMES_DIR = Path("debug_frames_v3")

class BBoxVerifier:
    def __init__(self):
        with open(BBOXES_FILE) as f:
            self.bboxes = json.load(f)

        self.current_idx = 0
        self.user_bbox = None

        self.fig, self.ax = plt.subplots(1, 1, figsize=(12, 9))
        self.fig.canvas.manager.set_window_title('BBox Verification')

        # Always active rectangle selector
        self.selector = RectangleSelector(
            self.ax, self.on_select,
            useblit=True, button=[1],
            minspanx=5, minspany=5,
            spancoords='pixels', interactive=True
        )

        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        self.load_segment(0)

    def get_frame_path(self, seg):
        mid_time = (seg['start_time'] + seg['end_time']) / 2
        frame_num = int(mid_time * 30) + 1
        return FRAMES_DIR / f"frame_{frame_num:05d}.jpg"

    def load_segment(self, idx):
        if idx < 0:
            idx = 0
        if idx >= len(self.bboxes):
            self.save_and_quit()
            return

        self.current_idx = idx
        self.user_bbox = None

        seg = self.bboxes[idx]
        frame_path = self.get_frame_path(seg)
        img = Image.open(frame_path)

        self.ax.clear()
        self.ax.imshow(img)

        self.current_bbox = (seg['x_min'], seg['y_min'], seg['x_max'], seg['y_max'])
        source = seg.get('source', 'detected')

        self.draw_bbox(self.current_bbox, 'red', f"Current ({source})")

        self.ax.set_title(
            f"Segment {idx+1}/{len(self.bboxes)} | "
            f"{seg['start_time']:.2f}s - {seg['end_time']:.2f}s\n"
            f"ENTER=accept | Click+drag=draw | R=reset | ←=back | Q=quit+save",
            fontsize=11
        )

        self.fig.canvas.draw()

    def draw_bbox(self, bbox, color, label):
        x1, y1, x2, y2 = bbox
        rect = patches.Rectangle(
            (x1, y1), x2-x1, y2-y1,
            linewidth=2, edgecolor=color, facecolor='none', label=label
        )
        self.ax.add_patch(rect)
        self.ax.legend(loc='upper right')

    def on_select(self, eclick, erelease):
        x1, y1 = int(eclick.xdata), int(eclick.ydata)
        x2, y2 = int(erelease.xdata), int(erelease.ydata)
        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)

        self.user_bbox = (x1, y1, x2, y2)

        # Redraw
        seg = self.bboxes[self.current_idx]
        frame_path = self.get_frame_path(seg)
        img = Image.open(frame_path)

        self.ax.clear()
        self.ax.imshow(img)

        self.draw_bbox(self.current_bbox, 'red', 'Current')
        self.draw_bbox(self.user_bbox, 'lime', 'YOUR BBOX')

        self.ax.set_title(
            f"Segment {self.current_idx+1}/{len(self.bboxes)} | "
            f"{seg['start_time']:.2f}s - {seg['end_time']:.2f}s\n"
            f"ENTER=accept GREEN | R=reset | Q=quit+save",
            fontsize=11
        )

        self.fig.canvas.draw()

    def on_key(self, event):
        if event.key == 'enter':
            bbox = self.user_bbox if self.user_bbox else self.current_bbox
            seg = self.bboxes[self.current_idx]
            seg['x_min'], seg['y_min'], seg['x_max'], seg['y_max'] = bbox
            seg['source'] = 'user' if self.user_bbox else seg.get('source', 'detected')

            print(f"Seg {self.current_idx+1}: {bbox}")
            self.load_segment(self.current_idx + 1)

        elif event.key == 'left':
            self.load_segment(self.current_idx - 1)

        elif event.key == 'r':
            self.user_bbox = None
            self.load_segment(self.current_idx)

        elif event.key == 'q':
            self.save_and_quit()

    def save_and_quit(self):
        with open(BBOXES_FILE, 'w') as f:
            json.dump(self.bboxes, f, indent=2)
        print(f"\nSaved {len(self.bboxes)} bboxes to {BBOXES_FILE}")
        plt.close()

    def run(self):
        plt.show()

def main():
    print("Controls:")
    print("  ENTER      = Accept (green if drawn, else red) → next")
    print("  Click+Drag = Draw new bbox")
    print("  R          = Reset to current")
    print("  LEFT       = Go back")
    print("  Q          = Quit and save\n")

    BBoxVerifier().run()

if __name__ == "__main__":
    main()
