#!/bin/bash
# Wrapper script to run peacock-trame with proper environment and monkey patch
# This script fixes the compatibility issue between peacock-trame and current MOOSE version
# by adding the missing hit.explode() function
#
# Usage:
#   ./run_peacock.sh                              # Uses default example (ex08_materials/ex08.i)
#   ./run_peacock.sh <example_dir> <input_file>   # Uses custom directory and input file
#
# Examples:
#   ./run_peacock.sh ~/peacock-work/moose/examples/ex01_inputfile ex01.i
#   ./run_peacock.sh ~/peacock-work/moose/examples/ex08_materials ex08.i

# Set up environment
export PYTHONPATH=~/peacock-work/moose/framework/contrib/hit:~/peacock-work/moose/python:$PYTHONPATH
export PATH=~/miniforge3/envs/moose/bin:$PATH

# Create timestamp for log file
timestamp=$(date +%Y%m%d_%H%M%S)
log_file=~/Documents/GitHub/PlayGround/20251223_openFOAM/peacock_${timestamp}.log

# Get the example directory and input file from arguments, or use defaults
# Expand tilde to absolute path
EXAMPLE_DIR=$(eval echo "${1:-~/peacock-work/moose/examples/ex08_materials}")
INPUT_FILE="${2:-ex08.i}"

# Port configuration (default 8082 to avoid conflict with Grafana on 8080)
PORT="${3:-8082}"

# Change to the example directory
cd "$EXAMPLE_DIR" || { echo "Error: Cannot change to directory $EXAMPLE_DIR"; exit 1; }

# Export INPUT_FILE and PORT as environment variables for Python to access
export PEACOCK_INPUT_FILE="$INPUT_FILE"
export PEACOCK_PORT="$PORT"

# Redirect all output (stdout and stderr) to both terminal and log file
{
    echo "========================================"
    echo "Starting peacock-trame GUI..."
    echo "========================================"
    echo "Working directory: $EXAMPLE_DIR"
    echo "Input file: $INPUT_FILE"
    echo "Port: $PORT"
    echo "Log file: $log_file"
    echo ""
    echo "GUI will be available at: http://localhost:$PORT/"
    echo "========================================"
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

# Now run peacock-trame with custom port
from peacock_trame.app.main import main
input_file = os.environ.get('PEACOCK_INPUT_FILE', 'ex01.i')
port = int(os.environ.get('PEACOCK_PORT', '8082'))
sys.argv = ['peacock-trame', '-I', './' + input_file, '--port', str(port)]
main()
"
} 2>&1 | tee "$log_file"

