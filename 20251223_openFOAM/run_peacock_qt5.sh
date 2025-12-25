#!/bin/bash

# Get the input file from command line argument
INPUT_FILE="${1:-~/peacock-work/moose/examples/ex08_materials/ex08.i}"

# Expand tilde to full path
INPUT_FILE="${INPUT_FILE/#\~/$HOME}"

# Check if input file exists
if [ ! -f "$INPUT_FILE" ]; then
    echo "Error: Input file not found: $INPUT_FILE"
    echo "Usage: $0 <path_to_input_file.i>"
    exit 1
fi

# Create logs directory if it doesn't exist
mkdir -p logs

# Generate timestamp for log file
timestamp=$(date +"%Y%m%d_%H%M%S")
log_file="logs/peacock_qt5_${timestamp}.log"

# Set up environment for peacock-qt5 conda environment
export PYTHONPATH=~/peacock-work/moose/python:$PYTHONPATH
export MOOSE_DIR=~/peacock-work/moose

# Set environment variables for building MOOSE applications
export LIBMESH_DIR=~/miniforge3/envs/peacock-qt5/libmesh
export WASP_DIR=~/miniforge3/envs/peacock-qt5/wasp
export PETSC_DIR=~/miniforge3/envs/peacock-qt5
export MOOSE_NO_CODESIGN=1

# Fix library path for numpy/gfortran
export DYLD_LIBRARY_PATH=~/miniforge3/envs/peacock-qt5/lib:$DYLD_LIBRARY_PATH

# Redirect all output (stdout and stderr) to both terminal and log file
{
    echo "Starting Peacock (PyQt5 GUI) from peacock-qt5 environment..."
    echo "Input file: $INPUT_FILE"
    echo "Log file: $log_file"
    echo ""
    echo "Environment: peacock-qt5 (Qt5-only, VTK 9.2.6)"
    echo ""

    # Run Peacock using the peacock-qt5 environment
    ~/miniforge3/envs/peacock-qt5/bin/python ~/peacock-work/moose/python/peacock/peacock -i "$INPUT_FILE"
    
    exit_code=$?
    if [ $exit_code -ne 0 ]; then
        echo ""
        echo "Peacock exited with error code: $exit_code"
        echo ""
    else
        echo ""
        echo "Peacock closed successfully."
        echo ""
    fi
} 2>&1 | tee "$log_file"

echo "Log saved to: $log_file"

