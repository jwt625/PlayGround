# qiskit-metal + scqubits Docker Environment

A complete Docker-based solution for running qiskit-metal and scqubits with full GUI support on macOS (Apple Silicon).

## 🎯 What This Provides

- ✅ **qiskit-metal 0.1.5** - Quantum device design and simulation
- ✅ **scqubits 3.1.0** - Superconducting qubit analysis
- ✅ **PySide2 5.15.2.1** - Full GUI support with X11 forwarding
- ✅ **Complete scientific stack** - NumPy, matplotlib, pandas, scipy
- ✅ **Native macOS integration** - Qt GUIs appear on your desktop

## 🚀 Quick Start

### Prerequisites
- Docker Desktop for Mac
- XQuartz (for GUI support)

### 1. Setup X11 Forwarding
```bash
./setup-x11-forwarding.sh
```

### 2. Build the Container
```bash
docker build --platform linux/x86_64 -t qiskit-metal-container .
```

### 3. Run the Container
```bash
docker run --platform linux/x86_64 -it --rm \
  -e DISPLAY=host.docker.internal:0 \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v $(pwd):/workspace/host \
  qiskit-metal-container bash
```

### 4. Install Packages (Inside Container)
```bash
# Activate virtual environment
source venv/bin/activate

# Install PySide2 from wheel
pip install https://files.pythonhosted.org/packages/c2/9a/78ca8bada6cf4d2798e0c823c025c590517d74445837f4eb50bfddce8737/PySide2-5.15.2.1-5.15.2-cp35.cp36.cp37.cp38.cp39.cp310-abi3-manylinux1_x86_64.whl

# Install remaining packages
pip install -r /workspace/requirements-docker.txt
pip install qiskit-metal

# Test installation
python -c "
import qiskit_metal
import scqubits
import PySide2
print('✅ All packages imported successfully!')
print(f'qiskit-metal version: {qiskit_metal.__version__}')
print(f'scqubits version: {scqubits.__version__}')
print(f'PySide2 version: {PySide2.__version__}')
"
```

## 🖥️ GUI Testing

Test that GUI applications work:

```bash
python -c "
from PySide2.QtWidgets import QApplication, QLabel
import sys

app = QApplication(sys.argv)
label = QLabel('Hello from Docker PySide2!')
label.show()
print('GUI window should appear on your macOS desktop!')
app.exec_()
"
```

## 📁 File Structure

```
20250619_SCQubits/
├── Dockerfile                    # Ubuntu 20.04 container definition
├── requirements-docker.txt       # Python package requirements
├── setup-x11-forwarding.sh      # macOS X11 configuration script
├── install-qiskit-metal.sh      # Automated installation (alternative)
├── test_installation.py         # Installation verification script
├── DevLog.md                    # Complete development journey
└── README.md                    # This file
```

## 🔧 Technical Details

### Why Docker + x86_64 Emulation?
- **Architecture Compatibility**: PySide2 wheels only available for x86_64
- **Consistent Environment**: Ubuntu 20.04 with compatible Qt5/LLVM versions
- **GUI Support**: X11 forwarding enables native Qt applications on macOS
- **Reproducible**: Same environment every time, no host system conflicts

### Package Versions
- **Python**: 3.9
- **NumPy**: 1.20.3 (compatible with all packages)
- **Qt**: 5.12.8 (Ubuntu 20.04 default)
- **LLVM**: 10 (compatible with PySide2)

## 🐛 Troubleshooting

### GUI Not Appearing?
1. Ensure XQuartz is running: `open -a XQuartz`
2. Allow localhost connections: `xhost +localhost`
3. Check DISPLAY variable: `echo $DISPLAY`

### Container Build Issues?
1. Ensure Docker has sufficient resources (4GB+ RAM)
2. Check internet connection for package downloads
3. Try building without cache: `docker build --no-cache ...`

### Import Errors?
1. Verify virtual environment is activated: `source venv/bin/activate`
2. Check package installation: `pip list | grep -E "(qiskit|scqubits|PySide2)"`
3. Test individual imports to isolate issues

## 📚 Usage Examples

### Basic qiskit-metal Usage
```python
import qiskit_metal as qmetal
from qiskit_metal import designs, draw
from qiskit_metal.qlibrary.qubits.transmon_pocket import TransmonPocket

# Create a design
design = designs.DesignPlanar()

# Add a transmon qubit
q1 = TransmonPocket(design, 'Q1', options=dict(pos_x='0mm', pos_y='0mm'))

# Launch GUI (will appear on macOS desktop)
gui = design.launch_gui()
```

### Basic scqubits Usage
```python
import scqubits as scq
import numpy as np

# Create a transmon qubit
tmon = scq.Transmon(
    EJ=25.0,    # Josephson energy in GHz
    EC=0.2,     # Charging energy in GHz
    ng=0.0,     # Offset charge
    ncut=30     # Charge cutoff
)

# Calculate energy spectrum
eigenvals = tmon.eigenvals(evals_count=10)
print(f"Ground state energy: {eigenvals[0]:.3f} GHz")
print(f"First excited state: {eigenvals[1]:.3f} GHz")
```

## 🎉 Success!

You now have a complete quantum device design environment with:
- Full qiskit-metal functionality for device design
- Complete scqubits library for qubit analysis  
- GUI support for interactive design
- Reproducible Docker environment
- Native macOS integration

Happy quantum computing! 🚀

---

*For detailed development history and troubleshooting journey, see [DevLog.md](DevLog.md)*
