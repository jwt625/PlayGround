#!/bin/bash
# Fix duplicate LC_RPATH in libopenblas that prevents loading on macOS
# This is a workaround for conda packaging bug on macOS ARM64

set -e

LIBOPENBLAS="$HOME/miniforge3/envs/peacock-qt5/lib/libopenblas.0.dylib"

echo "Fixing duplicate LC_RPATH in libopenblas..."
echo "Target: $LIBOPENBLAS"

# Check if file exists
if [ ! -f "$LIBOPENBLAS" ]; then
    echo "Error: $LIBOPENBLAS not found"
    exit 1
fi

# Backup original
cp "$LIBOPENBLAS" "${LIBOPENBLAS}.backup"
echo "Backup created: ${LIBOPENBLAS}.backup"

# Remove the duplicate RPATH entries
# We'll delete both and add back a single correct one
install_name_tool -delete_rpath "@loader_path" "$LIBOPENBLAS" 2>/dev/null || true
install_name_tool -delete_rpath "@loader_path/" "$LIBOPENBLAS" 2>/dev/null || true

# Add back a single RPATH
install_name_tool -add_rpath "@loader_path" "$LIBOPENBLAS"

echo "Fixed! Verifying..."
otool -l "$LIBOPENBLAS" | grep -A 2 LC_RPATH

echo ""
echo "Done! libopenblas RPATH has been fixed."

