#!/bin/bash
# Fix duplicate LC_RPATH in all affected libraries in peacock-qt5 environment
# This is a workaround for conda packaging bug on macOS ARM64

set -e

ENV_LIB="$HOME/miniforge3/envs/peacock-qt5/lib"

# List of critical libraries that need fixing for Peacock to run
LIBS=(
    "libopenblas.0.dylib"
    "libgfortran.5.dylib"
    "libquadmath.0.dylib"
)

echo "========================================="
echo "Fixing duplicate LC_RPATH entries"
echo "Environment: peacock-qt5"
echo "========================================="
echo ""

for lib in "${LIBS[@]}"; do
    LIBPATH="$ENV_LIB/$lib"
    
    echo "Processing: $lib"
    
    # Check if file exists
    if [ ! -f "$LIBPATH" ]; then
        echo "  WARNING: $LIBPATH not found, skipping"
        echo ""
        continue
    fi
    
    # Check current RPATH count
    RPATH_COUNT=$(otool -l "$LIBPATH" 2>/dev/null | grep -c "LC_RPATH" || echo "0")
    echo "  Current RPATH entries: $RPATH_COUNT"
    
    if [ "$RPATH_COUNT" -le 1 ]; then
        echo "  OK: Already has single or no RPATH entry, skipping"
        echo ""
        continue
    fi
    
    # Backup if not already backed up
    if [ ! -f "${LIBPATH}.backup" ]; then
        cp "$LIBPATH" "${LIBPATH}.backup"
        echo "  Backup created: ${lib}.backup"
    else
        echo "  Backup already exists, skipping backup"
    fi
    
    # Remove all RPATH entries (may need multiple attempts for duplicates)
    echo "  Removing duplicate RPATH entries..."
    for i in {1..5}; do
        install_name_tool -delete_rpath "@loader_path" "$LIBPATH" 2>/dev/null || true
        install_name_tool -delete_rpath "@loader_path/" "$LIBPATH" 2>/dev/null || true
    done
    
    # Add back a single RPATH
    echo "  Adding single RPATH entry..."
    install_name_tool -add_rpath "@loader_path" "$LIBPATH" 2>&1 | grep -v "warning: changes being made" || true
    
    # Verify
    NEW_COUNT=$(otool -l "$LIBPATH" 2>/dev/null | grep -c "LC_RPATH" || echo "0")
    echo "  New RPATH entries: $NEW_COUNT"
    
    if [ "$NEW_COUNT" -eq 1 ]; then
        echo "  ✓ SUCCESS: Fixed!"
    else
        echo "  ✗ ERROR: Still has $NEW_COUNT RPATH entries"
    fi
    
    echo ""
done

echo "========================================="
echo "Summary"
echo "========================================="
echo "Fixed libraries:"
for lib in "${LIBS[@]}"; do
    LIBPATH="$ENV_LIB/$lib"
    if [ -f "$LIBPATH" ]; then
        COUNT=$(otool -l "$LIBPATH" 2>/dev/null | grep -c "LC_RPATH" || echo "0")
        echo "  $lib: $COUNT RPATH entries"
    fi
done
echo ""
echo "Done! You can now try running Peacock."

