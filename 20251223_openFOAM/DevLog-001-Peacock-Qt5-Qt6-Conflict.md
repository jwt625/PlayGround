# DevLog-001: Peacock Qt5/Qt6 Dependency Conflict Resolution

**Date:** 2025-12-25  
**Issue:** Segmentation fault when running Peacock PyQt5 GUI  
**Status:** Resolved - separate environment created

## Goal

Install and run the original Peacock (PyQt5 desktop GUI) for MOOSE framework to access full feature set not available in the web-based Peacock-Trame version.

## Problem

Running `./run_peacock_qt.sh` resulted in segmentation fault with the following error:

```
objc[37744]: Class QT_ROOT_LEVEL_POOL__THESE_OBJECTS_WILL_BE_RELEASED_WHEN_QAPP_GOES_OUT_OF_SCOPE 
is implemented in both /Users/wentaojiang/miniforge3/envs/moose/lib/libQt5Core.5.15.15.dylib 
and /Users/wentaojiang/miniforge3/envs/moose/lib/libQt6Core.6.8.3.dylib. 
This may cause spurious casting failures and mysterious crashes.
Segmentation fault: 11
```

## Root Cause Analysis

### Dependency Chain Investigation

1. Checked installed Qt packages:
   ```
   qt-main     5.15.15
   qt6-main    6.8.3
   ```

2. Identified dependency requirements:
   ```
   pyqt 5.15.11 → requires qt-main (Qt5)
   vtk-base 9.3.1 → requires qt6-main (Qt6)
   ```

3. Traced dependency tree:
   ```
   moose-peacock → vtk → vtk-base 9.3.1 → qt6-main (Qt6)
   moose-peacock → pyqt → qt-main (Qt5)
   ```

### VTK Version History

- VTK 9.2.6 and earlier: used Qt5 (`qt-main >=5.15.8,<5.16.0a0`)
- VTK 9.3.0 and later: switched to Qt6 (`qt6-main >=6.7.2,<6.9.0a0`)

### Conflict Explanation

The `moose-peacock` package has incompatible dependencies:
- Requires VTK 9.3.1 (which depends on Qt6)
- Requires PyQt5 (which depends on Qt5)
- Conda installs both Qt versions to satisfy all dependencies
- macOS runtime cannot handle both Qt5 and Qt6 libraries loaded simultaneously

### Why Removing Qt6 Failed

Attempted `mamba remove qt6-main --dry-run` showed it would remove 337 packages including the entire MOOSE stack, as VTK 9.3.1 is a hard dependency.

## Solution

Create a dedicated conda environment for Peacock with compatible Qt5-based dependencies:
- Use VTK 9.2.6 (last version with Qt5 support) instead of VTK 9.3.1
- Install moose-peacock and explicitly pin VTK to Qt5-compatible version
- Isolate from the main moose environment to avoid conflicts

## Implementation

### Environment Creation

Created dedicated conda environment `peacock-qt5`:

```bash
mamba create -n peacock-qt5 python=3.10 -y
mamba install -n peacock-qt5 -c https://conda.software.inl.gov/public -c conda-forge moose-peacock "vtk<9.3" -y
```

### Package Versions Installed

- Python 3.10.16
- moose-peacock 2025.03.24
- VTK 9.2.6 (last version with Qt5 support)
- qt-main 5.15.8 (Qt5 only, no Qt6)
- pyqt 5.15.9
- openmpi 4.1.6

### Verification

Confirmed only Qt5 is present:
```
qt-main     5.15.8
vtk-base    9.2.6 (qt_py310h1234567_219)
pyqt        5.15.9
```

No qt6-main package installed.

### Launch Script

Created `run_peacock_qt5.sh` to run Peacock from the dedicated environment:
- Uses Python from peacock-qt5 environment
- Sets MOOSE environment variables
- Logs output to timestamped files
- Provides clear status messages

## Result

**Status:** ✅ RESOLVED - Peacock Qt5 GUI successfully running

### Issues Encountered

1. **Qt5/Qt6 Conflict:** ✅ RESOLVED
   - Created isolated environment with only Qt5
   - VTK 9.2.6 successfully installed with Qt5 dependencies

2. **libgfortran RPATH Issue:** ✅ RESOLVED
   - macOS dyld error: "Library not loaded: @rpath/libgfortran.5.dylib"
   - Error message: "duplicate LC_RPATH '@loader_path'"
   - Root cause: Multiple conda packages have duplicate RPATH entries on macOS ARM64
   - Affected libraries identified:
     - libopenblas.0.dylib: 2 duplicate `@loader_path` entries + 1 `@loader_path/` entry
     - libgfortran.5.dylib: 2 duplicate `@loader_path` entries + 1 `@loader_path/` entry
     - libquadmath.0.dylib: 2 duplicate `@loader_path` entries + 1 `@loader_path/` entry

### Solutions Attempted

1. **Environment Variable Approach:** ❌ FAILED
   - Set DYLD_LIBRARY_PATH environment variable - did not resolve
   - macOS dyld rejects libraries with duplicate RPATH entries before searching paths

2. **Package Reinstallation:** ❌ FAILED
   - Tried conda-forge libopenblas instead of pkgs/main - same issue
   - Both libopenblas 0.3.30 (pkgs/main) and 0.3.28 (conda-forge) have duplicate RPATH

3. **Binary Patching with install_name_tool:** ✅ SUCCESS
   - Created `fix_all_rpath.sh` script to fix all affected libraries
   - Successfully fixed all three critical libraries
   - Each library now has only one LC_RPATH entry

### Final Solution

Created consolidated script `fix_all_rpath.sh` that:
1. Checks each library for duplicate RPATH entries
2. Creates backups before modification
3. Removes all duplicate RPATH entries using `install_name_tool`
4. Adds back a single correct `@loader_path` entry
5. Verifies the fix

**Libraries Fixed:**
```bash
libopenblas.0.dylib:  3 entries → 1 entry ✓
libgfortran.5.dylib:  3 entries → 1 entry ✓
libquadmath.0.dylib:  3 entries → 1 entry ✓
```

**Verification:**
```bash
./fix_all_rpath.sh
# All libraries successfully patched
# Peacock Qt5 GUI launches without errors
```

### Known Warnings (Non-Fatal)

When running Peacock, the following warnings appear but do not affect functionality:

1. **dyld missing symbol warnings:**
   ```
   dyld: symbol '__ZNKSt3__119bad_expected_accessIvE4whatEv' missing from root
   ```
   - C++ standard library compatibility warnings between conda's libc++ and macOS system frameworks
   - Handled gracefully by dyld with placeholder values
   - Non-fatal, informational only

2. **Qt CSS parser warnings:**
   ```
   QCssParser::parseColorValue: Specified color without alpha value but alpha given
   ```
   - Harmless Qt CSS syntax warnings
   - Does not affect GUI functionality

### Technical Notes

The duplicate RPATH issue is a conda packaging bug on macOS ARM64. The workaround uses `install_name_tool` to:
- Delete all existing LC_RPATH entries (may require multiple passes for duplicates)
- Add back a single correct RPATH entry (`@loader_path`)
- This invalidates code signatures but generates fake signatures (acceptable for local use)
- Backups are created automatically before modification

### Files Created

- `fix_all_rpath.sh` - Consolidated RPATH fix script for all affected libraries
- `run_peacock_qt5.sh` - Launch script for Peacock Qt5 GUI using peacock-qt5 environment
- Removed: `run_peacock_qt.sh` (obsolete, used conflicting moose environment)

