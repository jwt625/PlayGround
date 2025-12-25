
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

### Step 3: Install Peacock-Trame for GUI visualization

Peacock-Trame provides a web-based interface for MOOSE input file editing and simulation visualization.
- GitHub: https://github.com/Kitware/peacock
- PyPI: https://pypi.org/project/peacock-trame/

```bash
# Make sure moose environment is activated
pip install peacock-trame
```

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

### Verify Peacock-Trame installation:
```bash
# Using the moose environment's Python directly
~/miniforge3/envs/moose/bin/python -c "import peacock_trame; print('Peacock-Trame installed successfully')"

# OR activate the environment first
conda activate moose
python -c "import peacock_trame; print('Peacock-Trame installed successfully')"
```

### Check installed version:
```bash
~/miniforge3/envs/moose/bin/pip list | grep peacock
```
Expected output: `peacock-trame      0.1.2`

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

This creates the `ex08-opt` executable required by peacock-trame.

## Running peacock-trame

After building the example executable:

```bash
# Set PYTHONPATH to include MOOSE Python modules
export PYTHONPATH=~/peacock-work/moose/python:$PYTHONPATH

# Run peacock-trame with the example
cd ~/peacock-work/moose/examples/ex08_materials
peacock-trame -I ./ex08.i
```

**Note:** The `PYTHONPATH` must be set to include `~/peacock-work/moose/python` because peacock-trame requires the `mooseutils` module which is not installed via pip but is part of the MOOSE repository.

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


