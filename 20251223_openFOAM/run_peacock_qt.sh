#!/bin/bash
# Wrapper script to run the original Peacock (PyQt5-based GUI)
# This is the traditional desktop GUI for MOOSE, as opposed to peacock-trame (web-based)

# Set up environment
export PYTHONPATH=~/peacock-work/moose/python:$PYTHONPATH
export PATH=~/miniforge3/envs/moose/bin:$PATH
export MOOSE_DIR=~/peacock-work/moose

# Set environment variables for building MOOSE applications
export LIBMESH_DIR=~/miniforge3/envs/moose/libmesh
export WASP_DIR=~/miniforge3/envs/moose/wasp
export PETSC_DIR=~/miniforge3/envs/moose
export MOOSE_NO_CODESIGN=1  # Required on macOS to avoid code signing errors

# Force Qt5 to avoid conflicts with Qt6
export QT_API=pyqt5
# Filter out Qt6 from library path by explicitly setting Qt5 paths first
export DYLD_LIBRARY_PATH=~/miniforge3/envs/moose/lib/python3.10/site-packages/PyQt5/Qt5/lib:~/miniforge3/envs/moose/lib:$DYLD_LIBRARY_PATH
# Tell Qt to use Qt5 plugins
export QT_PLUGIN_PATH=~/miniforge3/envs/moose/lib/python3.10/site-packages/PyQt5/Qt5/plugins
# Disable Qt6 completely
export QT_QPA_PLATFORM=

# Create timestamp for log file
timestamp=$(date +%Y%m%d_%H%M%S)
log_file=~/Documents/GitHub/PlayGround/20251223_openFOAM/peacock_qt_${timestamp}.log

# Get the input file from arguments, or use default
INPUT_FILE="${1:-~/peacock-work/moose/examples/ex08_materials/ex08.i}"

# Expand tilde to absolute path
INPUT_FILE=$(eval echo "$INPUT_FILE")

# Check if input file exists
if [ ! -f "$INPUT_FILE" ]; then
    echo "Error: Input file not found: $INPUT_FILE"
    exit 1
fi

# Redirect all output (stdout and stderr) to both terminal and log file
{
    echo "Starting Peacock (PyQt5 GUI)..."
    echo "Input file: $INPUT_FILE"
    echo "Log file: $log_file"
    echo ""
    echo "⚠️  WARNING: Qt5/Qt6 library conflict detected in this environment!"
    echo "   This may cause segmentation faults due to both Qt5 and Qt6 being present."
    echo ""
    echo "   RECOMMENDED: Use Peacock-Trame (web-based) instead:"
    echo "   ./run_peacock.sh ~/peacock-work/moose/examples/ex08_materials ex08.i"
    echo ""
    echo "   Or create a dedicated environment for Peacock Qt5 (see README.md)"
    echo ""
    echo "Attempting to run anyway..."
    echo ""

    # Run the original Peacock
    ~/miniforge3/envs/moose/bin/python ~/peacock-work/moose/python/peacock/peacock -i "$INPUT_FILE"

    exit_code=$?
    if [ $exit_code -ne 0 ]; then
        echo ""
        echo "❌ Peacock exited with error code: $exit_code"
        echo ""
        echo "This is likely due to Qt5/Qt6 conflicts. Please use Peacock-Trame instead:"
        echo "  ./run_peacock.sh ~/peacock-work/moose/examples/ex08_materials ex08.i"
        echo ""
    fi
} 2>&1 | tee "$log_file"

