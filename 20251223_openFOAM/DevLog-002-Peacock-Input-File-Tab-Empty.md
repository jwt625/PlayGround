# DevLog-002: Peacock Input File Tab Empty - Missing libwaspplot Library

**Date:** 2025-12-25  
**Issue:** Peacock Qt5 GUI launches but Input File tab is empty with all menu options greyed out  
**Status:** In Progress - Root cause identified

## Problem

After successfully resolving the Qt5/Qt6 conflict and RPATH issues (see DevLog-001), Peacock Qt5 GUI launches without crashing. However, the Input File tab is non-functional:

- Input File tab shows no content (no tree view, no file browser)
- All Input File dropdown menu options are greyed out
- No working directory or files displayed
- Tree view on left side is empty even when launching with `-i` flag and input file path

## Expected Behavior

According to Peacock documentation, when launched with an input file using the `-i` flag:

1. The Input File tab should display a tree view on the left showing the input file structure
2. Tree view should be populated with blocks and parameters from the loaded input file (Mesh, Variables, Kernels, BCs, Materials, Executioner, Outputs, etc.)
3. Blue items in tree can be right-clicked to add new elements
4. Black items can be double-clicked to edit parameters
5. If mesh is defined, a 3D mesh view should appear on the right side

## Investigation

### Initial Hypothesis: File Access Permissions

Initially suspected macOS file access permissions issue, as:
- NSOpenPanel warning appeared in logs
- Python/Terminal might lack "Full Disk Access" or "Files and Folders" permissions

However, no explicit permission denial errors were found in logs.

### Root Cause Discovery

The actual issue is that **Peacock cannot execute the MOOSE application binary** to query the input file syntax.

#### How Peacock Works

Peacock requires the MOOSE executable to:
1. Run with `--yaml` or `--dump` flag to get input file syntax/schema
2. Parse the YAML output to build the tree view structure
3. Validate and display the input file against the schema

If the executable cannot run, Peacock cannot build the tree view, resulting in an empty Input File tab.

#### Testing the Executable

Attempted to run the MOOSE executable manually:

```bash
export DYLD_LIBRARY_PATH=$HOME/miniforge3/envs/peacock-qt5/lib:$HOME/miniforge3/envs/moose/lib:$HOME/miniforge3/envs/moose/wasp/lib:$HOME/miniforge3/envs/moose/libmesh/lib:$HOME/miniforge3/envs/moose/petsc/lib
cd ~/peacock-work/moose/examples/ex08_materials
./ex08-opt --yaml
```

**Result:** Executable fails with missing library error:

```
dyld[67491]: Library not loaded: @rpath/libwaspplot.04.dylib
  Referenced from: /Users/wentaojiang/peacock-work/moose/examples/ex08_materials/ex08-opt
  Reason: tried: [multiple paths] (no such file)
```

### Library Dependency Analysis

Checked executable dependencies:

```bash
otool -L ~/peacock-work/moose/examples/ex08_materials/ex08-opt | grep wasp
```

**Output:**
```
@rpath/libwasplsp.04.dylib (compatibility version 4.0.0, current version 4.4.0)
@rpath/libwasphalite.04.dylib (compatibility version 4.0.0, current version 4.4.0)
@rpath/libwaspson.04.dylib (compatibility version 4.0.0, current version 4.4.0)
@rpath/libwaspcore.04.dylib (compatibility version 4.0.0, current version 4.4.0)
@rpath/libwaspplot.04.dylib (compatibility version 4.0.0, current version 4.4.0)
@rpath/libwaspexpr.04.dylib (compatibility version 4.0.0, current version 4.4.0)
@rpath/libwaspjson.04.dylib (compatibility version 4.0.0, current version 4.4.0)
@rpath/libwaspddi.04.dylib (compatibility version 4.0.0, current version 4.4.0)
@rpath/libwasphive.04.dylib (compatibility version 4.0.0, current version 4.4.0)
@rpath/libwasphit.04.dylib (compatibility version 4.0.0, current version 4.4.0)
@rpath/libwaspsiren.04.dylib (compatibility version 4.0.0, current version 4.4.0)
```

Searched for the missing library:

```bash
find ~/miniforge3/envs/moose -name "libwaspplot*.dylib"
# No results

find ~/miniforge3/envs/moose/wasp/lib -name "libwasp*.dylib"
# Found all other wasp libraries EXCEPT libwaspplot.04.dylib
```

**Confirmed:** `libwaspplot.04.dylib` is completely missing from the wasp installation.

### Environment Analysis

Checked installed wasp packages:

```bash
mamba list -n moose | grep wasp
```

**Output:**
```
moose-wasp                2025.05.13    build_1    https://conda.software.inl.gov/public
moose-wasp-base           2025.05.13    build_1    https://conda.software.inl.gov/public
```

The wasp packages are installed, but the `libwaspplot` library is not included in the package distribution.

## Root Cause

The MOOSE executable `ex08-opt` was compiled against a version of wasp that included `libwaspplot.04.dylib`, but the currently installed `moose-wasp` package (version 2025.05.13) does not provide this library. This creates a broken dependency chain:

1. Peacock launches successfully (Qt5 environment works)
2. Peacock attempts to execute `ex08-opt --yaml` to get input file syntax
3. Executable fails to load due to missing `libwaspplot.04.dylib`
4. Peacock cannot retrieve syntax information
5. Input File tab remains empty and non-functional

## Attempted Solutions

### 1. Library Path Configuration

Updated `run_peacock_qt5.sh` to include moose environment libraries:

```bash
export DYLD_LIBRARY_PATH=$HOME/miniforge3/envs/peacock-qt5/lib:$HOME/miniforge3/envs/moose/lib:$HOME/miniforge3/envs/moose/wasp/lib:$HOME/miniforge3/envs/moose/libmesh/lib:$HOME/miniforge3/envs/moose/petsc/lib
```

**Result:** Failed - library still not found because it does not exist in any environment.

**Note:** Initial attempt used `~` in paths, which dyld does not expand. Corrected to use `$HOME` for proper path expansion.

## Proposed Solution

Rebuild the MOOSE example executable against the currently installed libraries:

```bash
cd ~/peacock-work/moose/examples/ex08_materials
make clean
make -j4
```

This will recompile `ex08-opt` to link against the libraries actually present in the current moose environment, eliminating the dependency on the missing `libwaspplot.04.dylib`.

## Next Steps

1. Rebuild the ex08 example executable
2. Verify the executable runs with `--yaml` flag
3. Test Peacock Input File tab functionality
4. Document the complete working solution
5. Consider if other MOOSE examples need rebuilding

## Technical Notes

- The missing library issue indicates a version mismatch between when the executable was built and the current environment
- This is a common issue when MOOSE packages are updated but pre-compiled executables are not rebuilt
- The wasp library suite appears to have removed or renamed the `libwaspplot` component in recent versions
- Peacock's dependency on executable introspection means any executable loading failure will result in non-functional Input File tab


## Debug process:

The MOOSE source code is incompatible with the conda-provided libmesh version. The MOOSE repo you cloned is too new for the conda packages. You need to either:

1. Use a pre-built MOOSE executable from conda
2. Use an older MOOSE version that matches the conda packages

?

The problem is clear: MOOSE code uses `DynamicArrayWrapper` and `SemiDynamicSparseNumberArrayGeneric` from `MetaPhysicL` library, but the conda-provided libmesh has an older version that doesn't have these classes.

## Solution: Use Compatible MOOSE Version

**Date:** 2025-12-25
**Status:** RESOLVED

### Root Cause

The issue was a version mismatch between the MOOSE source code and the conda-provided libraries:

- Latest MOOSE repository (main branch, December 2025) uses newer MetaPhysicL API
- Conda packages are from May 2025:
  - moose-libmesh: 2025.05.23
  - moose-wasp: 2025.05.13
  - moose-dev: 2025.06.09

The latest MOOSE code requires `DynamicArrayWrapper` and `SemiDynamicSparseNumberArrayGeneric` classes that don't exist in the May 2025 libmesh version.

### Solution Implementation

Clone MOOSE at a release tag that matches the conda package timeframe:

```bash
cd ~/peacock-work
rm -rf moose
git clone --branch 2025-09-05-release --depth 1 https://github.com/idaholab/moose.git
```

The `2025-09-05-release` tag (released May 9, 2025) is compatible with the conda-provided libraries.

### Build Process

Build the example executable with conda-provided libraries:

```bash
cd ~/peacock-work/moose/examples/ex08_materials

export PATH=~/miniforge3/envs/moose/bin:$PATH
export LIBMESH_DIR=~/miniforge3/envs/moose/libmesh
export WASP_DIR=~/miniforge3/envs/moose/wasp
export PETSC_DIR=~/miniforge3/envs/moose
export MOOSE_NO_CODESIGN=1

make -j4
```

**Result:** Executable `ex08-opt` builds successfully (116KB).

### Peacock-Trame Configuration

Updated `run_peacock.sh` to use ex08_materials as the default example:

```bash
EXAMPLE_DIR=$(eval echo "${1:-~/peacock-work/moose/examples/ex08_materials}")
INPUT_FILE="${2:-ex08.i}"
```

### Verification

Peacock-trame now launches successfully with full functionality:

```bash
./run_peacock.sh
```

GUI available at: http://localhost:8080/

**Confirmed working features:**
- Input file loading and parsing
- Tree view with MOOSE input file structure
- 3D mesh viewer (previously non-functional)
- Input file editing capabilities

### Technical Notes

- The MOOSE repository must be kept at the 2025-09-05-release tag to maintain compatibility
- Updating to newer MOOSE versions will require updating conda packages to matching versions
- The conda packages provide stable, pre-built libraries that eliminate the need to build MOOSE framework from source
- This approach is more reliable than attempting to build the latest MOOSE against older libraries

### Lessons Learned

1. When using conda-provided development libraries, the application source code version must match the library versions
2. MOOSE release tags correspond to specific library versions and should be used for reproducible builds
3. The shallow clone approach (`--depth 1`) is sufficient and saves disk space
4. Version compatibility issues manifest as missing symbols or template definitions during compilation

