"""Read generated PNGs and write frame/anchor metadata; never modify source image pixels.

Generated nominal 4x4 atlases can have uneven row spacing. Find dark gutters,
measure the visible machine and register its stationary front foundation tip.
The result is a presentation mapping, not a collision/port definition.
"""
from pathlib import Path
from statistics import median
import json
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]

def gutters(counts):
    edges = [0]
    for i in range(1, 4):
        center = round(len(counts) * i / 4)
        start, end = center - 65, center + 50
        values = counts[start:end]
        threshold = max(5, min(values) + 3)
        edges.append(round(median(start + j for j, n in enumerate(values) if n < threshold)))
    return edges + [len(counts)]

result = {}
for name in ['extractor-cycle-v1', 'assembler-cycle-v1']:
    im = Image.open(ROOT / 'animations' / f'{name}.png').convert('RGB')
    width, height = im.size
    pixels = list(im.get_flattened_data() if hasattr(im, 'get_flattened_data') else im.getdata())
    bright = [max(p) > 85 for p in pixels]
    row_edges = gutters([sum(bright[y * width:(y + 1) * width]) for y in range(height)])
    frames = []
    for row in range(4):
        y0, y1 = row_edges[row:row + 2]
        col_edges = gutters([sum(bright[y * width + x] for y in range(y0, y1)) for x in range(width)])
        for col in range(4):
            x0, x1 = col_edges[col:col + 2]
            points = [(x, y) for y in range(y0, y1) for x in range(x0, x1) if bright[y * width + x]]
            left = max(x0, min(x for x, y in points) - 5)
            top = max(y0, min(y for x, y in points) - 5)
            right = min(x1, max(x for x, y in points) + 6)
            bottom_tip = max(y for x, y in points)
            bottom = min(y1, bottom_tip + 6)
            # The last bright rows contain the stationary front corner of the base.
            anchor_x = median(x for x, y in points if y >= bottom_tip - 4)
            frames.append({'source': [left, top, right - left, bottom - top],
                           'offset': [round(170 - (anchor_x - left), 2), 310 - (bottom_tip - top)]})
    result[name] = {'logicalSize': 340, 'frames': frames, 'method': 'dark gutters; visible foundation-tip registration; source PNG unmodified'}
(ROOT / 'animations' / 'frame-map.json').write_text(json.dumps(result, indent=2) + '\n')
print('Recorded 32 source rectangles and foundation anchors; PNGs unchanged.')
