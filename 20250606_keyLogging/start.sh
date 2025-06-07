#!/bin/bash

# Keystroke Tracker Launcher
# Starts both Swift helper and Go tracker together

echo "🚀 Starting Keystroke Tracker..."
echo "Press Ctrl+C to stop both processes"
echo ""

# Check if Go is installed
echo "🔧 Checking Go installation..."
if command -v go >/dev/null 2>&1; then
    GO_VERSION=$(go version | awk '{print $3}')
    echo "   ✅ Go is installed: $GO_VERSION"
else
    echo "   ❌ Go is not installed. Installing via Homebrew..."
    
    # Check if Homebrew is installed
    if command -v brew >/dev/null 2>&1; then
        echo "   🍺 Installing Go with Homebrew..."
        brew install go
        if [ $? -eq 0 ]; then
            echo "   ✅ Go installed successfully"
            GO_VERSION=$(go version | awk '{print $3}')
            echo "   ℹ️  Installed: $GO_VERSION"
        else
            echo "   ❌ Failed to install Go via Homebrew"
            echo "   💡 Please install Go manually from: https://golang.org/dl/"
            exit 1
        fi
    else
        echo "   ❌ Homebrew not found. Please install Go manually:"
        echo "   💡 Visit: https://golang.org/dl/"
        echo "   💡 Or install Homebrew first: https://brew.sh/"
        exit 1
    fi
fi
echo ""

# Check if Docker containers are running
echo "🐳 Checking Docker containers..."
if docker-compose ps | grep -q "Up"; then
    echo "   ✅ Prometheus & Grafana containers already running"
else
    echo "   🔄 Starting Prometheus & Grafana containers..."
    docker-compose up -d
    if [ $? -eq 0 ]; then
        echo "   ✅ Containers started successfully"
        # Give containers time to initialize
        sleep 3
    else
        echo "   ❌ Failed to start containers. Please check Docker Desktop is running."
        exit 1
    fi
fi
echo ""

# Function to cleanup processes on exit
cleanup() {
    echo ""
    echo "🛑 Stopping processes..."
    kill $SWIFT_PID $GO_PID 2>/dev/null
    echo "✅ Stopped"
    exit 0
}

# Set trap to cleanup on exit
trap cleanup SIGINT SIGTERM

# Start Swift helper in background
echo "📱 Starting Swift app detector helper..."
swift app-detector-helper.swift &
SWIFT_PID=$!
echo "   Swift helper started (PID: $SWIFT_PID)"

# Wait a moment for Swift to initialize
sleep 1

# Start Go tracker in background
echo "⌨️  Starting Go keystroke tracker..."
go run main.go &
GO_PID=$!
echo "   Go tracker started (PID: $GO_PID)"

echo ""
echo "🎯 Both processes running!"
echo "   • Swift helper: PID $SWIFT_PID"
echo "   • Go tracker: PID $GO_PID"
echo "   • Metrics: http://localhost:8080/metrics"
echo "   • Grafana: http://localhost:3001"
echo ""

# Wait for both processes
wait $SWIFT_PID $GO_PID