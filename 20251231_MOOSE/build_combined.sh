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

cd ~/peacock-work/moose/modules/combined
make -j4