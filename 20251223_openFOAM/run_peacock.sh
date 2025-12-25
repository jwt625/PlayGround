#!/bin/bash
# Wrapper script to run peacock-trame with proper environment and monkey patch
# This script fixes the compatibility issue between peacock-trame and current MOOSE version
# by adding the missing hit.explode() function

# Set up environment
export PYTHONPATH=~/peacock-work/moose/framework/contrib/hit:~/peacock-work/moose/python:$PYTHONPATH
export PATH=~/miniforge3/envs/moose/bin:$PATH

# Create timestamp for log file
timestamp=$(date +%Y%m%d_%H%M%S)
log_file=~/Documents/GitHub/PlayGround/20251223_openFOAM/peacock_${timestamp}.log

# Get the example directory and input file from arguments, or use defaults
# Expand tilde to absolute path
EXAMPLE_DIR=$(eval echo "${1:-~/peacock-work/moose/examples/ex01_inputfile}")
INPUT_FILE="${2:-ex01.i}"

# Change to the example directory
cd "$EXAMPLE_DIR" || { echo "Error: Cannot change to directory $EXAMPLE_DIR"; exit 1; }

# Export INPUT_FILE as environment variable for Python to access
export PEACOCK_INPUT_FILE="$INPUT_FILE"

# Redirect all output (stdout and stderr) to both terminal and log file
{
    echo "Starting peacock-trame GUI..."
    echo "Example directory: $EXAMPLE_DIR"
    echo "Input file: $INPUT_FILE"
    echo "Log file: $log_file"
    echo "GUI will be available at: http://localhost:8080/"
    echo ""

    # Run peacock-trame with the monkey patch
    ~/miniforge3/envs/moose/bin/python -c "
import sys
import os

# Add paths for MOOSE modules
sys.path.insert(0, os.path.expanduser('~/peacock-work/moose/framework/contrib/hit'))
sys.path.insert(0, os.path.expanduser('~/peacock-work/moose/python'))

# Import and patch hit module
import hit
def explode(node):
    '''Monkey patch: hit.explode() was removed from MOOSE but peacock-trame still expects it'''
    pass
hit.explode = explode

# Now run peacock-trame
from peacock_trame.app.main import main
input_file = os.environ.get('PEACOCK_INPUT_FILE', 'ex01.i')
sys.argv = ['peacock-trame', '-I', './' + input_file]
main()
"
} 2>&1 | tee "$log_file"

