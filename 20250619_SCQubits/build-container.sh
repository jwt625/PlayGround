#!/bin/bash

echo "🔨 Building optimized qiskit-metal Docker container..."
echo "📦 This will install all packages during build time for faster startup"
echo ""

# Remove old container if it exists
if docker image inspect qiskit-metal-container >/dev/null 2>&1; then
    echo "🗑️  Removing old container..."
    docker rmi qiskit-metal-container
fi

# Build new optimized container
echo "🚀 Building new container with pre-installed packages..."
echo "⏳ This will take 5-10 minutes but only needs to be done once..."

docker build --platform linux/x86_64 -t qiskit-metal-container .

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Container built successfully!"
    echo "🚀 Now you can run: ./run-qiskit-metal.sh"
    echo "⚡ Startup will be much faster since packages are pre-installed"
else
    echo ""
    echo "❌ Container build failed"
    exit 1
fi
