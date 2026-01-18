#!/bin/bash

# Script to normalize pyramid images:
# 1. Trim transparent backgrounds
# 2. Resize to consistent max dimension (512px)
# 3. Remove any floating backgrounds

PYRAMID_DIR="webapp/public/images/pyramid"
BACKUP_DIR="webapp/public/images/pyramid_backup"
MAX_SIZE=512

# Create backup
echo "Creating backup..."
mkdir -p "$BACKUP_DIR"
cp -r "$PYRAMID_DIR"/*.webp "$BACKUP_DIR/"

echo "Processing images..."
for img in "$PYRAMID_DIR"/*.webp; do
    filename=$(basename "$img")
    echo "Processing: $filename"
    
    # Process: trim transparent areas, resize to max dimension, maintain aspect ratio
    magick "$img" \
        -fuzz 1% \
        -trim \
        +repage \
        -resize "${MAX_SIZE}x${MAX_SIZE}>" \
        -background none \
        -gravity center \
        "$img"
done

echo "Done! Backup saved to $BACKUP_DIR"
echo "Processed $(ls -1 "$PYRAMID_DIR"/*.webp | wc -l) images"

