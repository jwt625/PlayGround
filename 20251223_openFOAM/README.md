
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

