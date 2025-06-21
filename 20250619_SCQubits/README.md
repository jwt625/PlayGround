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

### 1. One-Time Setup: Build Optimized Container
```bash
# Build container with all packages pre-installed (takes 5-10 minutes)
./build-container.sh
```

### 2. Daily Usage: Run Container
```bash
# Fast startup with everything ready (takes ~5 seconds)
./run-qiskit-metal.sh
```

That's it! The script will:
- ✅ Configure XQuartz automatically
- ✅ Start the optimized container
- ✅ Test GUI functionality
- ✅ Drop you into a working environment

### 3. Alternative Manual Setup
If you prefer manual control:

```bash
# Setup X11 forwarding
./setup-x11-forwarding.sh

# Build container manually
docker build --platform linux/x86_64 -t qiskit-metal-container .

# Run container manually
docker run --platform linux/x86_64 -it --rm \
  -e DISPLAY=host.docker.internal:0 \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v $(pwd):/workspace/host \
  qiskit-metal-container bash
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
├── Dockerfile                    # Ubuntu 20.04 container with pre-installed packages
├── requirements-docker.txt       # Python package requirements
├── build-container.sh           # Build optimized container (one-time setup)
├── run-qiskit-metal.sh          # Run container with GUI support (daily use)
├── setup-x11-forwarding.sh      # macOS X11 configuration script
├── install-qiskit-metal.sh      # Alternative installation script
├── test_installation.py         # Installation verification script
├── examples/
│   ├── Example full chip design.ipynb  # Original Jupyter notebook
│   └── full_chip_design.py            # Converted Python script
├── DevLog.md                    # Complete development journey
└── README.md                    # This file
```

## 🔧 Technical Details

### Why Docker + x86_64 Emulation?
- **Architecture Compatibility**: PySide2 wheels only available for x86_64
- **Consistent Environment**: Ubuntu 20.04 with compatible Qt5/LLVM versions
- **GUI Support**: X11 forwarding enables native Qt applications on macOS
- **Reproducible**: Same environment every time, no host system conflicts
- **Pre-installed Packages**: All dependencies built into container for instant startup

### Package Versions (Pre-installed)
- **Python**: 3.9
- **PySide2**: 5.15.2.1 (from official wheel)
- **qiskit-metal**: 0.1.5
- **scqubits**: 3.1.0
- **NumPy**: 1.20.3 (compatible with all packages)
- **Qt**: 5.12.8 (Ubuntu 20.04 default)
- **LLVM**: 10 (compatible with PySide2)

### Performance Optimizations
- **Fast Startup**: ~5 seconds (vs ~10 minutes with package installation)
- **Layer Caching**: Docker efficiently caches build layers
- **Pre-compiled**: All packages compiled during build time
- **Offline Capable**: Works without internet after initial build

## 🐛 Troubleshooting

### GUI Not Appearing?
1. Ensure XQuartz is running: `open -a XQuartz`
2. Allow localhost connections: `xhost +localhost`
3. Check DISPLAY variable: `echo $DISPLAY`
4. Try restarting: `./run-qiskit-metal.sh` (script handles most issues automatically)

### Container Build Issues?
1. Ensure Docker has sufficient resources (4GB+ RAM, 10GB+ disk space)
2. Check internet connection for package downloads
3. Try rebuilding: `./build-container.sh` (removes old container first)
4. For manual rebuild: `docker build --no-cache --platform linux/x86_64 -t qiskit-metal-container .`

### Slow Startup?
1. Use optimized workflow: `./build-container.sh` once, then `./run-qiskit-metal.sh` daily
2. Avoid manual package installation - everything is pre-installed
3. Check if you're using the right container: `docker images | grep qiskit-metal`

### Package Issues?
1. All packages are pre-installed - no need to install manually
2. If issues persist, rebuild container: `./build-container.sh`
3. Check versions: `python -c "import qiskit_metal, scqubits, PySide2; print('All OK')"`

## 📚 Usage Examples

### Running the Full Chip Design Example
```bash
# Inside the container
source venv/bin/activate
cd /workspace/host/examples

# Option 1: Run the converted Python script
python full_chip_design.py

# Option 2: Use Jupyter notebook
pip install jupyter nbconvert
jupyter nbconvert --to script "Example full chip design.ipynb"
python "Example full chip design.py"

# Option 3: Run Jupyter server for interactive use
jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root --NotebookApp.token=''
# Then open http://localhost:8888 in your browser
```

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
- ⚡ **Instant startup** - Optimized container with pre-installed packages
- 🔬 **Full qiskit-metal functionality** - Complete device design suite
- 📊 **Complete scqubits library** - Advanced qubit analysis tools
- 🖥️ **GUI support** - Interactive design with native macOS integration
- 🐳 **Reproducible environment** - Consistent Docker-based setup
- 📓 **Example notebooks** - Ready-to-run quantum chip designs
- 🚀 **One-command workflow** - `./run-qiskit-metal.sh` and you're ready!

## 🔄 Workflow Summary

### One-Time Setup:
```bash
./build-container.sh  # Build optimized container (5-10 minutes)
```

### Daily Usage:
```bash
./run-qiskit-metal.sh  # Start environment (5 seconds)
```

### Inside Container:
```bash
cd /workspace/host/examples
python full_chip_design.py  # Run quantum chip design
```

Happy quantum computing! 🚀⚛️

---

*For detailed development history and troubleshooting journey, see [DevLog.md](DevLog.md)*
