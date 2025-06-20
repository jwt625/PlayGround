#!/usr/bin/env python3
"""
Generate placeholder icons for the Chrome extension
Creates simple colored squares with "GPT" text
"""

from PIL import Image, ImageDraw, ImageFont
import os

# Create icons directory if it doesn't exist
icons_dir = os.path.dirname(os.path.abspath(__file__))

# Define icon sizes
sizes = [16, 48, 128]

for size in sizes:
    # Create a new image with a green background
    img = Image.new('RGBA', (size, size), color=(16, 163, 127, 255))  # #10a37f
    
    # Create a drawing context
    draw = ImageDraw.Draw(img)
    
    # Draw a simple "G" in white
    # For simplicity, just draw a circle with a line
    margin = size // 8
    draw.ellipse([margin, margin, size-margin, size-margin], 
                 outline='white', width=max(1, size//16))
    
    # Add a small horizontal line for the "G"
    mid = size // 2
    draw.rectangle([mid, mid, size-margin, mid+max(1, size//16)], 
                   fill='white')
    
    # Save the icon
    filename = f'icon{size}.png'
    img.save(os.path.join(icons_dir, filename))
    print(f"Created {filename}")

print("\nIcons created successfully!")
print("If you don't have PIL installed, you can:")
print("1. Use the create_icons.html file in a browser")
print("2. Or create simple colored squares in any image editor")