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
    echo "Note: This is the original desktop GUI. For the web-based version, use run_peacock.sh"
    echo ""

    # Run the original Peacock
    ~/miniforge3/envs/moose/bin/python ~/peacock-work/moose/python/peacock/peacock -i "$INPUT_FILE"
} 2>&1 | tee "$log_file"

