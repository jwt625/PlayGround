#!/bin/bash

# Create simple placeholder icons using ImageMagick or Python PIL
# If neither is available, we'll create minimal valid PNG files

create_icon() {
  size=$1
  output=$2
  
  # Try with ImageMagick first
  if command -v convert &> /dev/null; then
    convert -size ${size}x${size} xc:none -fill "#0066cc" -draw "circle $((size/2)),$((size/2)) $((size/2)),5" -fill white -gravity center -pointsize $((size/2)) -annotate +0+0 "▶" "$output"
    return 0
  fi
  
  # Try with Python PIL
  if command -v python3 &> /dev/null; then
    python3 << PYEOF
from PIL import Image, ImageDraw
import sys

size = $size
img = Image.new('RGBA', (size, size), (0, 102, 204, 255))
draw = ImageDraw.Draw(img)

# Draw a simple play triangle
points = [
    (size * 0.3, size * 0.2),
    (size * 0.3, size * 0.8),
    (size * 0.75, size * 0.5)
]
draw.polygon(points, fill=(255, 255, 255, 255))

img.save('$output')
PYEOF
    return 0
  fi
  
  # Fallback: create minimal valid PNG (1x1 transparent pixel)
  printf '\x89\x50\x4e\x47\x0d\x0a\x1a\x0a\x00\x00\x00\x0d\x49\x48\x44\x52\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\x0a\x49\x44\x41\x54\x78\x9c\x63\x00\x01\x00\x00\x05\x0 \x01\x0d\x0a\x2d\xb4\x00\x00\x00\x00\x49\x45\x4e\x44\xae\x42\x60\x82' > "$output"
}

create_icon 16 icon16.png
create_icon 48 icon48.png
create_icon 128 icon128.png

echo "Icons created!"
