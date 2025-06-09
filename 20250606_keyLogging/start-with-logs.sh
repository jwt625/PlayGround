#!/bin/bash

# Keystroke Tracker Launcher with Log Display
# Starts both processes and shows their output with colored prefixes

echo "🚀 Starting Keystroke Tracker with logs..."
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

# Generate Prometheus config from template and .env
echo "⚙️  Generating Prometheus configuration..."
if [ -f "./generate-prometheus-config.sh" ]; then
    ./generate-prometheus-config.sh
    if [ $? -ne 0 ]; then
        echo "   ❌ Failed to generate Prometheus config. Please check .env file."
        exit 1
    fi
else
    echo "   ⚠️  generate-prometheus-config.sh not found, using existing prometheus.yml"
fi
echo ""

# Check if Docker containers are running
echo "🐳 Checking Docker containers..."
if docker-compose ps | grep -q "Up"; then
    echo "   ✅ Prometheus & Grafana containers already running"
    echo "   🔄 Restarting Prometheus to pick up config changes..."
    docker-compose restart prometheus
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

# Check if binary exists or is outdated, build if needed
echo "🔨 Checking Go binaries..."
if [ ! -f "./keystroke-tracker" ] || [ "main.go" -nt "./keystroke-tracker" ]; then
    echo "   🔄 Building keystroke tracker binary..."
    go build
    if [ $? -eq 0 ]; then
        echo "   ✅ Main binary built successfully"
    else
        echo "   ❌ Failed to build main binary. Please check Go installation and code."
        exit 1
    fi
else
    echo "   ✅ Main binary is up to date"
fi

echo ""

# Colors for log prefixes
SWIFT_COLOR='\033[34m'  # Blue
GO_COLOR='\033[32m'     # Green
CHROME_COLOR='\033[35m' # Magenta
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

# Start Swift unified tracker with log prefix
echo "📱 Starting Swift unified tracker (app + trackpad)..."
(swift swift/tracker.swift 2>&1 | while IFS= read -r line; do
    echo -e "${SWIFT_COLOR}[swift]${RESET_COLOR} $line"
done) &

# Wait a moment for Swift to initialize
sleep 1


# Start Go tracker with log prefix
echo "⌨️  Starting Go keystroke tracker..."
(./keystroke-tracker 2>&1 | while IFS= read -r line; do
    echo -e "${GO_COLOR}[go]${RESET_COLOR} $line"
done) &

echo ""
echo "🎯 All processes running with logs below:"
echo "   • Metrics: http://localhost:8080/metrics"
echo "   • Grafana: http://localhost:3001"
echo ""

# Wait for all background jobs
wait