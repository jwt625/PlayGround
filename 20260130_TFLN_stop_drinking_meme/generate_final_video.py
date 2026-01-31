#!/usr/bin/env python3
"""
Generate TFLN meme video v2.
- 5px padding on bbox
- Extend timing by 0.1s before and after
- Font size = exact bbox height (single-line) or bbox_height/2 (two-line)
"""

import subprocess
import json

INPUT_VIDEO = "source_video.mp4"
OUTPUT_VIDEO = "final.mp4"
BBOXES_FILE = "final_bboxes.json"

FONT = "/System/Library/Fonts/Supplemental/Times New Roman Bold.ttf"
TEXT_COLOR = "#ecef8a"
SHADOW_COLOR = "#2a2a2a"

PADDING = 5
TIME_EXTEND = 0.1

CAPTIONS = [
    "PYROELECTRIC",           # 1
    "GRIIRA",                 # 2
    "DRIFT",                  # 3
    "SHIELDING",              # 4
    "DEFECTS",                # 5
    "MgO DOPING!",            # 6
    "HEAT CHIP",              # 7
    "BLUE LED",               # 8
    "DIRECT CONTACT",         # 9
    "EXTRA\nDIELECTRIC",      # 10
    "ANNEALING",              # 11
    "STABLE\nBIAS",           # 12
    "LOW Vpi",                # 13
    "HIGH POWER",             # 14 (split from merged segment)
    "LOW\nLOSS",              # 15 (split - two-line to match original COLLECT MONEY)
    "HIGH BANDWIDTH",         # 16
    "CMOS DRIVEN",            # 17
    "WAFER SCALE",            # 18
    "FOUNDRY READY",          # 19
    "PI LOVES\nHIM",          # 20
    "NATURE PAPER",           # 21
    "RELIABLE",               # 22
    "REVIEW ARTICLE",         # 23
    "INTEGRATED\nLASER",      # 24
    "PHOTONIC AI",            # 25
    "REVOLUTIONIZE\nTELECOM", # 26
    "JUST DOPE\nMgO !!",      # 27
]

def load_bboxes():
    with open(BBOXES_FILE) as f:
        return json.load(f)

def build_filter(bboxes):
    parts = []
    cur = "0:v"

    for i, seg in enumerate(bboxes):
        # Original bbox (before padding)
        orig_x1, orig_y1 = seg['x_min'], seg['y_min']
        orig_x2, orig_y2 = seg['x_max'], seg['y_max']
        orig_h = orig_y2 - orig_y1

        # Padded bbox for blur
        x1 = max(0, orig_x1 - PADDING)
        y1 = max(0, orig_y1 - PADDING)
        x2 = min(480, orig_x2 + PADDING)
        y2 = min(360, orig_y2 + PADDING)
        bw, bh = x2 - x1, y2 - y1

        # Extended timing
        start = max(0, seg['start_time'] - TIME_EXTEND)
        end = min(30.77, seg['end_time'] + TIME_EXTEND)

        caption = CAPTIONS[i]
        lines = caption.split('\n')
        is_two_line = len(lines) > 1

        # Font size = original bbox height (for single), or half (for two-line)
        if is_two_line:
            fontsize = orig_h // 2
        else:
            fontsize = orig_h

        # Shadow offset
        sh_down = max(2, fontsize // 10)
        sh_right = max(1, sh_down // 7)

        enable = f"between(t,{start},{end})"

        # Light blur on padded bbox
        blur_out = f"b{i}"
        parts.append(
            f"[{cur}]split[m{i}][r{i}];"
            f"[r{i}]crop={bw}:{bh}:{x1}:{y1},boxblur=8:2[bl{i}];"
            f"[m{i}][bl{i}]overlay={x1}:{y1}:enable='{enable}'[{blur_out}]"
        )
        cur = blur_out

        # Text position - center in ORIGINAL bbox area
        if is_two_line:
            line_gap = 2
            total_text_h = fontsize * 2 + line_gap
            ty1 = orig_y1 + (orig_h - total_text_h) // 2
            ty2 = ty1 + fontsize + line_gap

            # Shadow line 1
            parts.append(
                f"[{cur}]drawtext=fontfile='{FONT}':text='{lines[0]}':"
                f"fontcolor={SHADOW_COLOR}:fontsize={fontsize}:"
                f"x=(w-text_w)/2+{sh_right}:y={ty1}+{sh_down}:"
                f"enable='{enable}'[s1_{i}]"
            )
            # Shadow line 2
            parts.append(
                f"[s1_{i}]drawtext=fontfile='{FONT}':text='{lines[1]}':"
                f"fontcolor={SHADOW_COLOR}:fontsize={fontsize}:"
                f"x=(w-text_w)/2+{sh_right}:y={ty2}+{sh_down}:"
                f"enable='{enable}'[s2_{i}]"
            )
            # Main line 1
            parts.append(
                f"[s2_{i}]drawtext=fontfile='{FONT}':text='{lines[0]}':"
                f"fontcolor={TEXT_COLOR}:fontsize={fontsize}:"
                f"x=(w-text_w)/2:y={ty1}:"
                f"enable='{enable}'[t1_{i}]"
            )
            # Main line 2
            parts.append(
                f"[t1_{i}]drawtext=fontfile='{FONT}':text='{lines[1]}':"
                f"fontcolor={TEXT_COLOR}:fontsize={fontsize}:"
                f"x=(w-text_w)/2:y={ty2}:"
                f"enable='{enable}'[t{i}]"
            )
            cur = f"t{i}"
        else:
            ty = orig_y1 + (orig_h - fontsize) // 2

            # Shadow
            parts.append(
                f"[{cur}]drawtext=fontfile='{FONT}':text='{caption}':"
                f"fontcolor={SHADOW_COLOR}:fontsize={fontsize}:"
                f"x=(w-text_w)/2+{sh_right}:y={ty}+{sh_down}:"
                f"enable='{enable}'[sh{i}]"
            )
            # Main
            parts.append(
                f"[sh{i}]drawtext=fontfile='{FONT}':text='{caption}':"
                f"fontcolor={TEXT_COLOR}:fontsize={fontsize}:"
                f"x=(w-text_w)/2:y={ty}:"
                f"enable='{enable}'[t{i}]"
            )
            cur = f"t{i}"

    return ";".join(parts), cur

def main():
    print("Loading bboxes...")
    bboxes = load_bboxes()
    print(f"  {len(bboxes)} segments")
    print(f"  Padding: {PADDING}px, Time extend: ±{TIME_EXTEND}s")

    print("\nCaption mapping:")
    for i, (seg, cap) in enumerate(zip(bboxes, CAPTIONS)):
        h = seg['y_max'] - seg['y_min']
        is_two = '\n' in cap
        fs = h // 2 if is_two else h
        print(f"  {i+1:2d}. h={h:2d}px → fontsize={fs:2d} | {cap.replace(chr(10), ' / ')}")

    print("\nBuilding filter...")
    filt, final = build_filter(bboxes)

    print("Running ffmpeg...")
    cmd = [
        "ffmpeg", "-i", INPUT_VIDEO,
        "-filter_complex", filt,
        "-map", f"[{final}]", "-map", "0:a",
        "-c:v", "libx264", "-preset", "medium", "-crf", "18",
        "-c:a", "copy", "-y", OUTPUT_VIDEO
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode == 0:
        print(f"\nSuccess! Output: {OUTPUT_VIDEO}")
    else:
        print(f"\nError:\n{result.stderr[-2000:]}")

if __name__ == "__main__":
    main()
