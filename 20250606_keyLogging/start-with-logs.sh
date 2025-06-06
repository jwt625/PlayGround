#!/bin/bash

# Keystroke Tracker Launcher with Log Display
# Starts both processes and shows their output with colored prefixes

echo "🚀 Starting Keystroke Tracker with logs..."
echo "Press Ctrl+C to stop both processes"
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

# Colors for log prefixes
SWIFT_COLOR='\033[34m'  # Blue
GO_COLOR='\033[32m'     # Green
RESET_COLOR='\033[0m'   # Reset

# Function to cleanup processes on exit
cleanup() {
    echo ""
    echo "🛑 Stopping processes..."
    jobs -p | xargs kill 2>/dev/null
    echo "✅ Stopped"
    exit 0
}

# Set trap to cleanup on exit
trap cleanup SIGINT SIGTERM

# Start Swift helper with log prefix
echo "📱 Starting Swift app detector helper..."
(swift app-detector-helper.swift 2>&1 | while IFS= read -r line; do
    echo -e "${SWIFT_COLOR}[swift]${RESET_COLOR} $line"
done) &

# Wait a moment for Swift to initialize
sleep 1

# Start Go tracker with log prefix
echo "⌨️  Starting Go keystroke tracker..."
(go run main.go 2>&1 | while IFS= read -r line; do
    echo -e "${GO_COLOR}[go]${RESET_COLOR} $line"
done) &

echo ""
echo "🎯 Both processes running with logs below:"
echo "   • Metrics: http://localhost:8080/metrics"
echo "   • Grafana: http://localhost:3001"
echo ""

# Wait for all background jobs
wait