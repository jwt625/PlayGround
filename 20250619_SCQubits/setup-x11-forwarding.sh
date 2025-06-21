#!/bin/bash

echo "🖥️  Setting up X11 forwarding for Docker GUI on macOS..."

# Check if XQuartz is installed
if ! command -v xquartz &> /dev/null; then
    echo "❌ XQuartz not found. Installing via Homebrew..."
    if ! command -v brew &> /dev/null; then
        echo "❌ Homebrew not found. Please install Homebrew first:"
        echo "   /bin/bash -c \"\$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\""
        exit 1
    fi
    brew install --cask xquartz
    echo "✅ XQuartz installed. Please log out and log back in, then run this script again."
    exit 0
fi

# Start XQuartz if not running
if ! pgrep -x "XQuartz" > /dev/null; then
    echo "🚀 Starting XQuartz..."
    open -a XQuartz
    sleep 3
fi

# Configure XQuartz for network connections
echo "🔧 Configuring XQuartz for Docker..."
defaults write org.xquartz.X11 nolisten_tcp -bool false
defaults write org.xquartz.X11 no_auth -bool false

# Allow connections from localhost
echo "🔐 Setting up X11 authentication..."
xhost +localhost

# Get the display number
DISPLAY_NUM=$(echo $DISPLAY | sed 's/.*:\([0-9]*\).*/\1/')
if [ -z "$DISPLAY_NUM" ]; then
    DISPLAY_NUM=0
fi

echo "✅ X11 forwarding setup complete!"
echo ""
echo "📋 Docker run command with GUI support:"
echo "docker run -it --rm \\"
echo "  -e DISPLAY=host.docker.internal:$DISPLAY_NUM \\"
echo "  -v /tmp/.X11-unix:/tmp/.X11-unix \\"
echo "  -v \$(pwd):/workspace/host \\"
echo "  qiskit-metal-container"
echo ""
echo "🔧 To build and run the container:"
echo "1. Build: docker build -t qiskit-metal-container ."
echo "2. Run:   docker run -it --rm -e DISPLAY=host.docker.internal:$DISPLAY_NUM -v /tmp/.X11-unix:/tmp/.X11-unix -v \$(pwd):/workspace/host qiskit-metal-container"
echo ""
echo "📝 Note: If you get permission errors, you may need to run:"
echo "   xhost +local:docker"
