#!/bin/bash
set -e

echo "🚀 Installing qiskit-metal with PySide2 in Ubuntu Docker container..."

# Activate virtual environment
source /workspace/venv/bin/activate

# Install base requirements
echo "📦 Installing base requirements..."
pip install -r /workspace/requirements-docker.txt

# Try to install PySide2 from wheel first, fallback to source build
echo "🔨 Attempting to install PySide2..."

# First try: install from wheel (faster)
echo "📦 Trying PySide2 wheel installation..."
if pip install PySide2==5.15.2.1; then
    echo "✅ PySide2 installed from wheel!"
else
    echo "⚠️  Wheel installation failed, building from source..."

    # Download and build PySide2 from source
    cd /tmp

    # Download PySide2 source
    wget https://download.qt.io/official_releases/QtForPython/pyside2/PySide2-5.15.8-src/pyside-setup-opensource-src-5.15.8.tar.xz
    tar -xf pyside-setup-opensource-src-5.15.8.tar.xz
    cd pyside-setup-opensource-src-5.15.8

    # Set environment variables for building
    export PATH="/usr/lib/llvm-10/bin:$PATH"
    export LLVM_INSTALL_DIR="/usr/lib/llvm-10"
    export QT_SELECT=qt5

    # Build PySide2 with correct environment
    echo "🔧 Building shiboken2..."
    python setup.py build --qmake=qmake --parallel=4

    echo "📦 Installing PySide2..."
    python setup.py install
fi

# Install qiskit-metal
echo "🎯 Installing qiskit-metal..."
cd /workspace
pip install qiskit-metal

# Test installation
echo "🧪 Testing installation..."
python -c "
import qiskit_metal
import scqubits
import PySide2
print('✅ All packages imported successfully!')
print(f'qiskit-metal version: {qiskit_metal.__version__}')
print(f'scqubits version: {scqubits.__version__}')
print(f'PySide2 version: {PySide2.__version__}')
"

echo "🎉 Installation complete! qiskit-metal is ready to use."
echo ""
echo "To run with GUI support, make sure X11 forwarding is enabled:"
echo "  docker run -it --rm -e DISPLAY=host.docker.internal:0 -v /tmp/.X11-unix:/tmp/.X11-unix qiskit-metal-container"
