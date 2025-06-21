#!/bin/bash

echo "🚀 Starting qiskit-metal Docker environment with GUI support..."

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to wait for XQuartz to start
wait_for_xquartz() {
    echo "⏳ Waiting for XQuartz to start..."
    local count=0
    while [ ! -S /tmp/.X11-unix/X0 ] && [ $count -lt 30 ]; do
        sleep 1
        count=$((count + 1))
    done
    
    if [ -S /tmp/.X11-unix/X0 ]; then
        echo "✅ XQuartz is running"
        return 0
    else
        echo "❌ XQuartz failed to start"
        return 1
    fi
}

# Check prerequisites
echo "🔍 Checking prerequisites..."

if ! command_exists docker; then
    echo "❌ Docker not found. Please install Docker Desktop for Mac."
    exit 1
fi

if ! command_exists xhost; then
    echo "❌ XQuartz not found. Please install XQuartz."
    echo "   brew install --cask xquartz"
    exit 1
fi

# Step 1: Setup XQuartz
echo "🖥️  Setting up XQuartz..."

# Kill any existing XQuartz processes
killall XQuartz 2>/dev/null || true
sleep 2

# Configure XQuartz preferences
echo "🔧 Configuring XQuartz preferences..."
defaults write org.xquartz.X11 nolisten_tcp -bool false
defaults write org.xquartz.X11 no_auth -bool false

# Start XQuartz
echo "🚀 Starting XQuartz..."
open -a XQuartz

# Wait for XQuartz to start
if ! wait_for_xquartz; then
    echo "❌ Failed to start XQuartz. Please start it manually and try again."
    exit 1
fi

# Set DISPLAY variable
export DISPLAY=:0
echo "📺 DISPLAY set to: $DISPLAY"

# Allow localhost connections
echo "🔐 Configuring X11 access control..."
xhost +localhost

# Verify X11 socket exists
if [ -S /tmp/.X11-unix/X0 ]; then
    echo "✅ X11 socket found: /tmp/.X11-unix/X0"
else
    echo "❌ X11 socket not found. XQuartz may not be running properly."
    exit 1
fi

# Step 2: Build Docker container if it doesn't exist
echo "🐳 Checking Docker container..."

if ! docker image inspect qiskit-metal-container >/dev/null 2>&1; then
    echo "🔨 Building Docker container (this may take a few minutes)..."
    docker build --platform linux/x86_64 -t qiskit-metal-container .
    
    if [ $? -ne 0 ]; then
        echo "❌ Failed to build Docker container"
        exit 1
    fi
    echo "✅ Docker container built successfully"
else
    echo "✅ Docker container already exists"
fi

# Step 3: Run the container
echo "🚀 Starting Docker container..."

docker run --platform linux/x86_64 -it --rm \
  -e DISPLAY=host.docker.internal:0 \
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  -v "$(pwd)":/workspace/host \
  qiskit-metal-container bash -c "
echo '🐧 Inside Docker container...'
echo '📦 Setting up Python environment...'

# Activate virtual environment
source venv/bin/activate

# Check if packages are already installed
if python -c 'import PySide2' 2>/dev/null; then
    echo '✅ PySide2 already installed'
else
    echo '📦 Installing PySide2 from wheel...'
    pip install https://files.pythonhosted.org/packages/c2/9a/78ca8bada6cf4d2798e0c823c025c590517d74445837f4eb50bfddce8737/PySide2-5.15.2.1-5.15.2-cp35.cp36.cp37.cp38.cp39.cp310-abi3-manylinux1_x86_64.whl
fi

if python -c 'import qiskit_metal' 2>/dev/null; then
    echo '✅ qiskit-metal already installed'
else
    echo '📦 Installing remaining packages...'
    pip install -r /workspace/requirements-docker.txt
    pip install qiskit-metal
fi

echo '🧪 Testing installation...'
python -c \"
try:
    import qiskit_metal
    import scqubits
    import PySide2
    print('✅ All packages imported successfully!')
    print(f'qiskit-metal version: {qiskit_metal.__version__}')
    print(f'scqubits version: {scqubits.__version__}')
    print(f'PySide2 version: {PySide2.__version__}')
except ImportError as e:
    print(f'❌ Import error: {e}')
    exit(1)
\"

echo '🖥️  Testing X11 connection...'
apt-get update >/dev/null 2>&1
apt-get install -y x11-apps >/dev/null 2>&1

echo '👀 Testing with xeyes (close the window to continue)...'
timeout 10s xeyes || echo '⚠️  xeyes test timed out or failed'

echo '🎨 Testing PySide2 GUI...'
python -c \"
from PySide2.QtWidgets import QApplication, QLabel, QPushButton, QVBoxLayout, QWidget
import sys

app = QApplication(sys.argv)

# Create a simple window
window = QWidget()
window.setWindowTitle('qiskit-metal Docker Environment')
window.setGeometry(100, 100, 300, 200)

layout = QVBoxLayout()

label = QLabel('🎉 Success! qiskit-metal is ready!')
label.setStyleSheet('font-size: 14px; padding: 10px;')

info_label = QLabel('GUI forwarding is working!\\nYou can now use qiskit-metal with full GUI support.')
info_label.setStyleSheet('font-size: 12px; padding: 10px;')

button = QPushButton('Close')
button.clicked.connect(app.quit)

layout.addWidget(label)
layout.addWidget(info_label)
layout.addWidget(button)

window.setLayout(layout)
window.show()

print('🎉 GUI window should appear on your macOS desktop!')
print('Close the window to continue...')
app.exec_()
\"

echo ''
echo '🎉 Setup complete! You are now in the qiskit-metal environment.'
echo ''
echo '📚 Quick start examples:'
echo '  # Basic qiskit-metal usage:'
echo '  python -c \"import qiskit_metal as qmetal; print(qmetal.__version__)\"'
echo ''
echo '  # Launch qiskit-metal GUI:'
echo '  python -c \"'
echo '  import qiskit_metal as qmetal'
echo '  from qiskit_metal import designs'
echo '  design = designs.DesignPlanar()'
echo '  gui = design.launch_gui()\"'
echo ''
echo '  # Basic scqubits usage:'
echo '  python -c \"'
echo '  import scqubits as scq'
echo '  tmon = scq.Transmon(EJ=25.0, EC=0.2, ng=0.0, ncut=30)'
echo '  print(f\\\"Ground state: {tmon.eigenvals(evals_count=2)[0]:.3f} GHz\\\")\"'
echo ''
echo '💡 Your host directory is mounted at /workspace/host'
echo '🐚 You are now in an interactive bash shell. Type \"exit\" to leave.'
echo ''

# Start interactive shell
exec bash
"

echo "👋 Exited Docker container. XQuartz is still running for future use."
