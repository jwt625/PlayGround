
# References

User Guide:
- https://doc.openfoam.com/2312/


Docker installation:
- https://develop.openfoam.com/Development/openfoam/-/wikis/precompiled/docker
- https://develop.openfoam.com/packaging/containers


# Setup

Had to get the default container that has the tutorials.
- `./openfoam-docker -default -update`

Trying out SimFlow's GUI
- https://sim-flow.com/download-simflow/

Following this example:
- https://help.sim-flow.com/tutorials/buildings

# SimFlow

Actually tried SimFlow and it is pretty good and easy to get started with.
- https://sim-flow.com/

# MOOSE Framework

MOOSE (Multiphysics Object-Oriented Simulation Environment) is a finite element framework for solving coupled multiphysics problems.
- https://mooseframework.inl.gov/

## Installation

### Step 1: Install Miniforge (conda package manager)

Miniforge is installed at `~/miniforge3` and includes both `conda` and `mamba` package managers.

```bash
curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
bash Miniforge3-$(uname)-$(uname -m).sh
```

**Note:** Use `~/miniforge3/bin/mamba` or `~/miniforge3/bin/conda` if the commands are not in your PATH.

### Step 2: Create and activate the moose environment

```bash
# Create environment with Python 3.10
~/miniforge3/bin/conda create -n moose python=3.10

# Activate the environment
conda activate moose
# OR if conda activate doesn't work:
source ~/miniforge3/bin/activate moose

# Install MOOSE development dependencies
conda install -c conda-forge moose-dev
```

### Step 3: Install Peacock GUI (Two Options)

MOOSE provides two GUI options for input file editing and simulation visualization:

#### Option A: Peacock (Original PyQt5-based Desktop GUI)

The traditional desktop application with full Qt5 interface.

```bash
# Install via conda (includes all dependencies: PyQt5, VTK, matplotlib, pandas)
mamba install -c https://conda.software.inl.gov/public moose-peacock
```

#### Option B: Peacock-Trame (Web-based GUI)

Modern web-based interface that runs in your browser.
- GitHub: https://github.com/Kitware/peacock
- PyPI: https://pypi.org/project/peacock-trame/

```bash
# Install via pip
pip install peacock-trame
```

**Note:** You can install both! They serve different use cases and don't conflict.

## Verification

### Check that the moose environment exists:
```bash
~/miniforge3/bin/mamba env list
# OR
~/miniforge3/bin/conda env list
```
Expected output should include:
```
moose    /Users/wentaojiang/miniforge3/envs/moose
```

### Verify Peacock installations:

**Check installed packages:**
```bash
~/miniforge3/bin/mamba list -n moose | grep peacock
```
Expected output:
```
moose-peacock      2025.04.17    openmpi_0    conda.software.inl.gov/public
peacock-trame      0.1.2         pypi_0       pypi
```

**Test Peacock-Trame (web-based):**
```bash
~/miniforge3/envs/moose/bin/python -c "import peacock_trame; print('Peacock-Trame installed successfully')"
```

**Test Original Peacock (PyQt5):**
```bash
~/miniforge3/envs/moose/bin/python -c "from peacock.CheckRequirements import has_requirements; print('Peacock requirements:', 'OK' if has_requirements() else 'MISSING')"
```

## Examples

- https://vimeo.com/838073269
- https://www.kitware.com/the-evolution-of-peacock-a-powerful-interface-for-moose-simulations/
- https://mooseframework.inl.gov/python/peacock.html

## Building MOOSE Examples

Before running peacock-trame with MOOSE examples, you must build the example executable. The `moose-dev` conda package provides pre-built libraries (libmesh, petsc, wasp), so you don't need to build MOOSE from source.

### Build Requirements

Set the following environment variables to use conda-provided libraries:

```bash
export PATH=~/miniforge3/envs/moose/bin:$PATH
export LIBMESH_DIR=~/miniforge3/envs/moose/libmesh
export WASP_DIR=~/miniforge3/envs/moose/wasp
export PETSC_DIR=~/miniforge3/envs/moose
export MOOSE_NO_CODESIGN=1  # Required on macOS to avoid code signing errors
```

### Build Example Application

```bash
cd ~/peacock-work/moose/examples/ex08_materials
make -j4
```

This creates the `ex08-opt` executable required by both Peacock GUIs.

## Running Peacock GUIs

### Option 1: Original Peacock (PyQt5 Desktop GUI)

The traditional desktop application with full Qt5 interface:

```bash
# Using the wrapper script (recommended)
./run_peacock_qt.sh ~/peacock-work/moose/examples/ex08_materials/ex08.i

# Or run directly
~/miniforge3/envs/moose/bin/python ~/peacock-work/moose/python/peacock/peacock -i ~/peacock-work/moose/examples/ex08_materials/ex08.i
```

**Features:**
- Native desktop application
- Full Qt5 interface with all widgets
- Integrated input file editor
- Mesh and results visualization
- Execution control

### Option 2: Peacock-Trame (Web-based GUI)

Modern web-based interface that runs in your browser:

```bash
# Using the wrapper script (recommended - includes hit.explode() fix)
./run_peacock.sh ~/peacock-work/moose/examples/ex08_materials ex08.i

# Or run directly (may have compatibility issues)
cd ~/peacock-work/moose/examples/ex08_materials
export PYTHONPATH=~/peacock-work/moose/python:$PYTHONPATH
peacock-trame -I ./ex08.i
```

The GUI will be available at: http://localhost:8080/

**Features:**
- Web-based interface (runs in browser)
- Modern Trame framework
- Input file editing
- Basic visualization
- Can be accessed remotely

**Note:** Both GUIs require the `PYTHONPATH` to include `~/peacock-work/moose/python` because they need the `mooseutils` module which is part of the MOOSE repository.

## Development Log

### 2025-12-24: Installation and Build Verification

**Initial Setup:**
- Installed Miniforge3 conda package manager
- Created `moose` environment with Python 3.10
- Installed `moose-dev` from INL conda channel (provides pre-built libmesh, petsc, wasp)
- Installed `peacock-trame` via pip
- Created `~/peacock-work` directory structure
- Cloned peacock repository: `git clone --recursive https://github.com/Kitware/peacock.git`
- Cloned MOOSE repository: `git clone https://github.com/idaholab/moose.git --depth 1`

**Build Issues Encountered:**

1. **Missing Dependencies for peacock-trame:**
   - The `moose-dev` conda package does not include the `mooseutils` Python module
   - Solution: Set `PYTHONPATH=~/peacock-work/moose/python:$PYTHONPATH`

2. **MOOSE Example Build Failures:**
   - Initial build attempts failed with missing libmesh-config, WASP errors
   - Root cause: Build system expected submodules to be built from source
   - Solution: Configure build to use conda-provided libraries via environment variables:
     - `LIBMESH_DIR=~/miniforge3/envs/moose/libmesh`
     - `WASP_DIR=~/miniforge3/envs/moose/wasp`
     - `PETSC_DIR=~/miniforge3/envs/moose`

3. **macOS Code Signing Error:**
   - Build completed but failed during code signing step
   - Error: "file is already signed. pass -f to sign regardless"
   - Solution: Set `MOOSE_NO_CODESIGN=1` environment variable

**Successful Build Configuration:**
With proper environment variables set, MOOSE examples build successfully using conda-provided libraries without requiring submodule initialization or building from source.

**Verification Results:**
- miniforge3 installed at `~/miniforge3`
- moose environment with moose-dev packages
- peacock-trame version 0.1.2 installed
- MOOSE repository cloned with examples
- mooseutils Python module accessible
- ex08-opt executable builds successfully (116KB)
- peacock-trame launches with built executable

**Key Findings:**
- The `moose-dev` conda package provides pre-built libraries sufficient for building MOOSE applications
- Git submodules (libmesh, petsc, wasp) do not need to be initialized when using conda packages
- MOOSE example applications must be compiled before use with peacock-trame
- peacock-trame requires both the MOOSE Python modules and compiled executables for full functionality

### 2025-12-24: Peacock-Trame GUI Fix and Usage

**Issue:** peacock-trame 0.1.2 expects `hit.explode()` function which was removed from current MOOSE version

**Error:**
```
WARNING: setInputFile exception: module 'hit' has no attribute 'explode'
```

**Root Cause:**
- peacock-trame calls `hit.explode(root)` in `InputFile.py` after parsing input files
- The `hit` module (compiled from `~/peacock-work/moose/framework/contrib/hit/hit.pyx`) no longer includes the `explode()` function
- This is a version compatibility issue between peacock-trame and the current MOOSE repository

**Solution:**
Created a wrapper script `run_peacock.sh` that monkey-patches the `hit` module to add a no-op `explode()` function before launching peacock-trame.

**Usage:**
```bash
# Run with default example (ex01_inputfile)
./run_peacock.sh

# Run with specific example
./run_peacock.sh ~/peacock-work/moose/examples/ex08_materials ex08.i
```

The GUI will be available at: http://localhost:8080/

**Environment Setup:**
The script automatically sets:
- `PYTHONPATH` to include MOOSE Python modules and compiled hit module
- `PATH` to include moose conda environment binaries
- Creates timestamped log files for each run

**Available Examples:**
All examples are in `~/peacock-work/moose/examples/`:
- ex01_inputfile - Basic input file and mesh loading
- ex02_kernel - Custom kernel implementation
- ex03_coupling - Coupled physics
- ex04_bcs - Boundary conditions
- ex05_amr - Adaptive mesh refinement
- ex06_transient - Transient simulations
- ex07_ics - Initial conditions
- ex08_materials - Material properties
- ex09_stateful_materials - Stateful materials
- ex10_aux - Auxiliary variables

**Troubleshooting:**
```bash
# Check if peacock-trame is running
ps aux | grep peacock-trame

# Stop the GUI
pkill -f peacock-trame

# View latest log
ls -lt peacock_*.log | head -1
```

**Note:** The warning "ParaView is not available" is expected. The GUI will still work for input file editing and basic visualization.

### 2025-12-25: Original Peacock (PyQt5) Installation

**Installation:**
- Installed `moose-peacock` conda package version 2025.04.17
- Package includes all required dependencies: PyQt5, VTK, matplotlib, pandas

**Known Issue: Qt5/Qt6 Library Conflict**

The environment has both Qt5 and Qt6 installed, causing segmentation faults:

**Root Cause:**
- `moose-peacock` → requires `vtk` → requires `vtk-base 9.3.1`
- `vtk-base 9.3.1` → requires `qt6-main` (VTK switched from Qt5 to Qt6 in version 9.3.x)
- `moose-peacock` → also requires `pyqt` → requires `qt-main` (Qt5)
- Result: Both `qt-main` (Qt5) and `qt6-main` (Qt6) are installed
- macOS runtime error: "Class QT_ROOT_LEVEL_POOL is implemented in both libQt5Core and libQt6Core"

**Why Both Versions Exist:**
- VTK 9.2.6 and earlier used Qt5
- VTK 9.3.0+ switched to Qt6
- The `moose-peacock` package depends on VTK 9.3.1, which requires Qt6
- But Peacock's GUI (PyQt5) requires Qt5
- Conda installs both to satisfy all dependencies, but they conflict at runtime

**Setup:**
- Created `run_peacock_qt.sh` wrapper script for easy launching
- Both Peacock GUIs (original and trame) can coexist in the same environment
- Original Peacock provides traditional desktop GUI experience
- Peacock-Trame provides modern web-based interface

**Key Differences:**
1. **Original Peacock (PyQt5)**:
   - Native desktop application
   - Full Qt5 widget interface
   - More mature and feature-complete
   - Better for local development
   - Requires X11/display server

2. **Peacock-Trame**:
   - Web-based (runs in browser)
   - Modern Trame framework
   - Can be accessed remotely
   - Lighter weight
   - Some compatibility issues with current MOOSE (requires hit.explode() monkey patch)

**Attempted Solution: Dedicated Qt5 Environment**

Attempted to create separate conda environment `peacock-qt5` with Qt5-compatible dependencies, but encountered macOS-specific conda packaging issues.

**Issue Summary:**
1. Qt5/Qt6 conflict successfully resolved by isolating Qt5 in separate environment
2. Secondary issue: libgfortran RPATH error on macOS ARM64
   - Error: "Library not loaded: @rpath/libgfortran.5.dylib (duplicate LC_RPATH)"
   - Root cause: Conda packages have malformed RPATH entries on macOS
   - Affects both pkgs/main and conda-forge libopenblas packages

**Workaround:**
Use Peacock-Trame (web-based version) which works correctly:

```bash
./run_peacock.sh ~/peacock-work/moose/examples/ex08_materials ex08.i
```

**Technical Details:**
- peacock-qt5 environment created with: Python 3.10.16, VTK 9.2.6, Qt5 only
- Qt5/Qt6 conflict resolved, but libgfortran linking prevents execution
- See DevLog-001-Peacock-Qt5-Qt6-Conflict.md for detailed analysis

**Current Status:**
- Original Peacock (PyQt5): ❌ **Not working** - macOS conda packaging issue
- Peacock-Trame (web-based): ✅ **Working** - Recommended


