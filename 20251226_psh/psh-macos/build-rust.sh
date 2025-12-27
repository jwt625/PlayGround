#!/bin/bash
set -e

echo "Building Rust FFI library..."

# Navigate to psh-ffi directory
cd "$(dirname "$0")/../psh-ffi"

# Build in release mode
cargo build --release

echo "Rust library built successfully at: target/release/libpsh_ffi.a"

