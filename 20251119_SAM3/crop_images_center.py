#!/usr/bin/env python3
"""
Crop the center portion of images to reduce lens distortion effects.
This is useful when images have heavy barrel/pincushion distortion at edges.
"""

import cv2
import numpy as np
from pathlib import Path
import argparse
import shutil

def crop_center(image, crop_fraction=0.5):
    """
    Crop the center portion of an image.
    
    Args:
        image: Input image (numpy array)
        crop_fraction: Fraction of image to keep (0.5 = keep center 50% width and height)
    
    Returns:
        Cropped image
    """
    h, w = image.shape[:2]
    
    # Calculate crop dimensions
    new_h = int(h * crop_fraction)
    new_w = int(w * crop_fraction)
    
    # Calculate crop coordinates (centered)
    start_y = (h - new_h) // 2
    start_x = (w - new_w) // 2
    end_y = start_y + new_h
    end_x = start_x + new_w
    
    # Crop
    cropped = image[start_y:end_y, start_x:end_x]
    
    return cropped

def process_images(input_dir, output_dir, crop_fraction=0.5, image_extensions=None):
    """
    Process all images in a directory, cropping the center portion.
    
    Args:
        input_dir: Path to input images
        output_dir: Path to save cropped images
        crop_fraction: Fraction of image to keep (default 0.5 = 50%)
        image_extensions: List of image extensions to process
    """
    if image_extensions is None:
        image_extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
    
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Find all images
    image_files = []
    for ext in image_extensions:
        image_files.extend(input_path.glob(f'*{ext}'))
    
    image_files = sorted(image_files)
    
    if not image_files:
        print(f"No images found in {input_dir}")
        return
    
    print(f"Found {len(image_files)} images")
    print(f"Cropping to center {crop_fraction*100:.0f}% (width and height)")
    
    # Process each image
    for i, img_file in enumerate(image_files, 1):
        print(f"[{i}/{len(image_files)}] Processing {img_file.name}...", end=' ')
        
        # Read image
        img = cv2.imread(str(img_file))
        if img is None:
            print(f"FAILED to read")
            continue
        
        original_shape = img.shape[:2]
        
        # Crop center
        cropped = crop_center(img, crop_fraction)
        
        cropped_shape = cropped.shape[:2]
        
        # Save cropped image
        output_file = output_path / img_file.name
        cv2.imwrite(str(output_file), cropped)
        
        print(f"OK ({original_shape[1]}x{original_shape[0]} -> {cropped_shape[1]}x{cropped_shape[0]})")
    
    print(f"\nDone! Cropped images saved to: {output_dir}")

def main():
    parser = argparse.ArgumentParser(
        description='Crop center portion of images to reduce lens distortion effects'
    )
    parser.add_argument(
        'input_dir',
        help='Directory containing input images'
    )
    parser.add_argument(
        'output_dir',
        help='Directory to save cropped images'
    )
    parser.add_argument(
        '--crop-fraction',
        type=float,
        default=0.5,
        help='Fraction of image to keep (default: 0.5 = center 50%%)'
    )
    
    args = parser.parse_args()
    
    # Validate crop fraction
    if not 0 < args.crop_fraction <= 1.0:
        print("Error: crop_fraction must be between 0 and 1")
        return
    
    process_images(args.input_dir, args.output_dir, args.crop_fraction)

if __name__ == '__main__':
    main()

