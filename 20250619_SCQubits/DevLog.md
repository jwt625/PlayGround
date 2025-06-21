# Development Log: Building PySide2 from Source for qiskit-metal

## Project Overview
Attempting to install qiskit-metal and scqubits in a Python virtual environment, which requires PySide2 5.15.2.1 that is not available as a pre-built wheel for the current platform (Apple Silicon macOS).

## Initial Setup
- **Environment**: macOS Apple Silicon, Python 3.9
- **Goal**: Install qiskit-metal 0.1.5 and scqubits 3.1.0
- **Challenge**: PySide2 5.15.2.1 dependency not available as pre-built wheel

## Installation Journey

### Phase 1: Basic Package Installation
1. Created virtual environment in `20250619_SCQubits/venv`
2. Successfully installed qiskit-metal without dependencies using `--no-deps`
3. Installed core scientific packages with correct versions:
   - numpy==1.24.2 → 1.19.5 (downgraded for PySide2 compatibility)
   - pandas==1.5.3
   - matplotlib==3.7.0
   - shapely==2.0.1
   - scipy==1.10.0
   - scqubits==3.1.0

### Phase 2: Dependency Resolution
Identified missing qiskit-metal dependencies:
- ✅ addict==2.4.0
- ✅ descartes==1.1.0
- ✅ gdspy==1.6.12
- ✅ pint==0.20.1
- ✅ pyyaml==6.0
- ✅ pygments==2.14.0
- ✅ ipython==8.10.0
- ❌ **pyside2==5.15.2.1** (main blocker)
- ❌ geopandas==0.12.2 (optional)
- ❌ gmsh==4.11.1 (optional)
- ❌ pyaedt==0.6.46 (optional)
- ❌ pyEPR-quantum==0.8.5.7 (optional)
- ❌ qdarkstyle==3.1 (optional)

### Phase 3: PySide2 Build Attempt
**Decision**: Build PySide2 5.15.8 from source since pre-built wheels unavailable.

#### Step 1: Source Download
```bash
wget https://download.qt.io/official_releases/QtForPython/pyside2/PySide2-5.15.8-src/pyside-setup-opensource-src-5.15.8.tar.xz
tar -xf pyside-setup-opensource-src-5.15.8.tar.xz
```

#### Step 2: Build Dependencies Resolution
**Challenge 1: CMake Version**
- Initial error: CMake version incompatibility
- **Solution**: Downgraded from CMake 3.31+ to CMake 3.19.8
```bash
wget https://github.com/Kitware/CMake/releases/download/v3.19.8/cmake-3.19.8-macos-universal.tar.gz
# Installed to /usr/local/bin/
```

**Challenge 2: LLVM/Clang**
- PySide2 requires libclang version ≥3.9
- **Solution**: Installed LLVM@12 via Homebrew
```bash
brew install llvm@12
export LLVM_INSTALL_DIR="/opt/homebrew/opt/llvm@12"
```

**Challenge 3: Qt5**
- **Solution**: Installed Qt5 via Homebrew
```bash
brew install qt@5
export PATH="/opt/homebrew/opt/qt@5/bin:$PATH"
```

#### Step 3: NumPy Compatibility Crisis
**Critical Issue**: PySide2 5.15.8 requires `NPY_ARRAY_UPDATEIFCOPY` flag which was removed in NumPy 1.20+

**Resolution Process**:
1. Attempted NumPy 1.24.2 → Failed (flag removed)
2. Attempted NumPy 1.20.3 → Failed (wheel compatibility issues)
3. **Success**: Downgraded to NumPy 1.19.5
   - Required compatible Cython version: 0.29.32 (downgraded from 3.1.2)
   - Used `--no-use-pep517` flag to avoid wheel building issues

#### Step 4: Build Process
**Environment Setup**:
```bash
export PATH="/opt/homebrew/opt/qt@5/bin:/opt/homebrew/opt/llvm@12/bin:$PATH"
export LDFLAGS="-L/opt/homebrew/opt/qt@5/lib -L/opt/homebrew/opt/llvm@12/lib"
export CPPFLAGS="-I/opt/homebrew/opt/qt@5/include -I/opt/homebrew/opt/llvm@12/include"
export PKG_CONFIG_PATH="/opt/homebrew/opt/qt@5/lib/pkgconfig"
export LLVM_INSTALL_DIR="/opt/homebrew/opt/llvm@12"
```

**Build Command**:
```bash
python setup.py build --qmake=/opt/homebrew/opt/qt@5/bin/qmake
```

### Phase 4: Build Results

#### ✅ Successful Components
1. **shiboken2**: Built successfully (100% completion)
   - API extractor compiled
   - libshiboken shared library created
   - shiboken2 generator built
   - All tests passed

#### ❌ Final Blocker: C++ Standard Library Compatibility
**Issue**: PySide2 5.15.8 incompatible with modern libc++ (Clang 16+)
- Error: `"Libc++ only supports Clang 16 and later"` but old PySide2 expects pre-Clang 16 behavior
- Multiple C++ namespace and type declaration conflicts
- Root cause: PySide2 5.15.8 predates modern C++ standard library changes

## Key Learnings

### Successful Compatibility Matrix
- ✅ CMake 3.19.8 (vs modern 3.31+)
- ✅ LLVM/Clang 12.0.1 (vs system Clang 16+)
- ✅ NumPy 1.19.5 with `NPY_ARRAY_UPDATEIFCOPY` (vs NumPy 1.20+)
- ✅ Cython 0.29.32 (vs modern 3.1.2)
- ✅ Qt5 5.15.16 via Homebrew

### Technical Insights
1. **NumPy API Evolution**: `NPY_ARRAY_UPDATEIFCOPY` was deprecated and removed, requiring specific old versions
2. **Build Tool Compatibility**: Modern build tools often break compatibility with older source code
3. **C++ Standard Library Evolution**: libc++ changes between Clang versions can break old code
4. **Dependency Cascade**: Each component requires specific versions of its dependencies

## Phase 5: Docker Solution with x86_64 Emulation

### Final Breakthrough: Architecture-Specific Wheels
After extensive source building attempts, discovered that PySide2 wheels are available but only for x86_64 architecture.

**Key Insight**: Apple Silicon (ARM64) was the blocker - not the build process itself.

#### Solution Implementation
1. **Docker Container**: Created Ubuntu 20.04 container with all dependencies
2. **x86_64 Emulation**: Used `--platform linux/x86_64` for architecture compatibility
3. **Pre-built Wheels**: Successfully used official PySide2 5.15.2.1 wheel
4. **X11 Forwarding**: Configured GUI tunneling to macOS via XQuartz

#### Final Installation Process
```bash
# Build x86_64 container
docker build --platform linux/x86_64 -t qiskit-metal-container .

# Run with GUI support
docker run --platform linux/x86_64 -it --rm \
  -e DISPLAY=host.docker.internal:0 \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v $(pwd):/workspace/host \
  qiskit-metal-container bash

# Install PySide2 from wheel
pip install https://files.pythonhosted.org/packages/c2/9a/78ca8bada6cf4d2798e0c823c025c590517d74445837f4eb50bfddce8737/PySide2-5.15.2.1-5.15.2-cp35.cp36.cp37.cp38.cp39.cp310-abi3-manylinux1_x86_64.whl

# Install remaining packages
pip install -r requirements-docker.txt
pip install qiskit-metal
```

## Final Success ✅

### Working Installation
- ✅ **qiskit-metal 0.1.5**: Fully functional with GUI support
- ✅ **scqubits 3.1.0**: Complete quantum circuit analysis
- ✅ **PySide2 5.15.2.1**: GUI framework working with X11 forwarding
- ✅ **Scientific stack**: NumPy, matplotlib, pandas, scipy all compatible
- ✅ **GUI Tunneling**: Qt applications display natively on macOS desktop

### Performance Metrics
- **Total Development Time**: ~6 hours
- **Source Build Attempts**: Multiple (ultimately unnecessary)
- **Final Solution Time**: ~15 minutes (Docker + wheel installation)
- **Container Build Time**: ~5 minutes
- **Package Installation**: ~2 minutes

## Key Learnings

### Technical Insights
1. **Architecture Matters**: ARM64 vs x86_64 compatibility is critical for Python wheels
2. **Docker Emulation**: Modern Docker can seamlessly emulate x86_64 on Apple Silicon
3. **Wheel Availability**: Always check wheel availability before attempting source builds
4. **GUI Forwarding**: X11 forwarding through Docker works excellently for Qt applications

### Strategic Insights
1. **Try Simple Solutions First**: Check for pre-built wheels before complex source builds
2. **Platform Awareness**: Consider architecture compatibility early in troubleshooting
3. **Docker Benefits**: Containerization provides consistent, reproducible environments
4. **Incremental Problem Solving**: Each failed attempt provided valuable debugging information

## Files Created
- `Dockerfile`: Ubuntu 20.04 container with Qt5 and development tools
- `requirements-docker.txt`: Compatible package versions for container environment
- `install-qiskit-metal.sh`: Automated installation script (ultimately unused)
- `setup-x11-forwarding.sh`: macOS X11 configuration for GUI support
- `test_installation.py`: Comprehensive installation test script
- `DevLog.md`: This comprehensive development log
- `README.md`: User-friendly setup instructions

## Current Status: COMPLETE ✅
- **qiskit-metal**: Fully functional with GUI support
- **scqubits**: Complete quantum circuit simulation capabilities
- **Development Environment**: Reproducible Docker-based setup
- **GUI Support**: Native Qt applications on macOS desktop
- **Documentation**: Complete setup and troubleshooting guide

---
*Total time invested: ~6 hours of systematic problem-solving leading to complete success*
