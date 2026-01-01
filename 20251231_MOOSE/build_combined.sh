#!/bin/bash
# Build combined module
# set the required environment variables and build the combined module executable

export PATH=~/miniforge3/envs/moose/bin:$PATH
export LIBMESH_DIR=~/miniforge3/envs/moose/libmesh
export WASP_DIR=~/miniforge3/envs/moose/wasp
export PETSC_DIR=~/miniforge3/envs/moose
export MOOSE_NO_CODESIGN=1

# Use conda-provided compilers to avoid macOS linker issues with -lm
export CC=clang
export CXX=clang++
export FC=gfortran

# macOS workaround: Set SDK path for conda gfortran to find system libraries
export SDKROOT=$(xcrun --show-sdk-path)
export CONDA_BUILD_SYSROOT=$SDKROOT

# macOS workaround: Create empty libm.a since math functions are in libSystem
# This prevents "ld: library not found for -lm" errors with conda gfortran
if [ ! -f ~/miniforge3/envs/moose/lib/libm.a ]; then
    echo "Creating empty libm.a for macOS compatibility..."
    echo "" | clang -c -x c - -o /tmp/empty_libm.o
    ar cr ~/miniforge3/envs/moose/lib/libm.a /tmp/empty_libm.o
    ranlib ~/miniforge3/envs/moose/lib/libm.a
    rm /tmp/empty_libm.o
fi

cd ~/peacock-work/moose/modules/combined
make -j4