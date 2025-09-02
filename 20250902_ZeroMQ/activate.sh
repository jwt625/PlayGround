#!/bin/bash

# ZeroMQ Learning Environment Activation Script
# Source this file to activate the virtual environment and set up the environment

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}Activating ZeroMQ Learning Environment...${NC}"

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Activate the virtual environment
if [ -f "$SCRIPT_DIR/.venv/bin/activate" ]; then
    source "$SCRIPT_DIR/.venv/bin/activate"
    echo -e "${GREEN}✓ Virtual environment activated${NC}"
else
    echo "❌ Virtual environment not found. Please run: uv venv .venv --python 3.11"
    return 1
fi

# Set up environment variables
export PYTHONPATH="$SCRIPT_DIR:$PYTHONPATH"
export ZMQ_LEARNING_ROOT="$SCRIPT_DIR"

# Add useful aliases
alias zmq-test="python test_setup.py"
alias zmq-req-rep="python 01-zmq-basics/req_rep.py"
alias zmq-push-pull="python 01-zmq-basics/push_pull.py"
alias zmq-pub-sub="python 01-zmq-basics/pub_sub.py"
alias zmq-router-dealer="python 01-zmq-basics/router_dealer.py"
alias zmq-coordinator="python 02-cluster-patterns/coordinator.py"
alias zmq-worker="python 02-cluster-patterns/worker.py"
alias zmq-monitor="python 02-cluster-patterns/monitor.py"
alias zmq-heartbeat="python 02-cluster-patterns/heartbeat.py"
alias zmq-cluster="./docker/simulate-cluster.sh"

echo -e "${GREEN}✓ Environment variables and aliases set${NC}"
echo ""
echo "Available commands:"
echo "  zmq-test           - Test the setup"
echo "  zmq-req-rep        - Run REQ-REP examples"
echo "  zmq-push-pull      - Run PUSH-PULL examples"
echo "  zmq-pub-sub        - Run PUB-SUB examples"
echo "  zmq-router-dealer  - Run ROUTER-DEALER examples"
echo "  zmq-coordinator    - Run cluster coordinator"
echo "  zmq-worker         - Run cluster worker"
echo "  zmq-monitor        - Run cluster monitor"
echo "  zmq-heartbeat      - Run heartbeat examples"
echo "  zmq-cluster        - Docker cluster simulation"
echo ""
echo "Learning path:"
echo "  1. Start with: zmq-test"
echo "  2. Try basic patterns in 01-zmq-basics/"
echo "  3. Explore cluster patterns in 02-cluster-patterns/"
echo "  4. Follow docs/learning-notes.md"
echo ""
echo -e "${GREEN}Happy learning! 🚀${NC}"
