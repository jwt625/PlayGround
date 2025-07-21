# ParaView Setup for Palace Electromagnetic Simulations

## Overview
This directory contains Palace electromagnetic simulation results that can be visualized using ParaView. Palace is a finite element electromagnetic simulation software, and this appears to be a coaxial waveguide simulation with transient analysis.

## Installation Status
✅ ParaView 6.0.0-RC2 has been successfully installed via Homebrew

## Available Data

### 1. Transient Volume Data
- **Location**: `example_coaxial_postpro_open/paraview/transient/`
- **Main file**: `transient.pvd`
- **Description**: 3D volume electromagnetic field data over time
- **Time range**: 0 to 1.0 (21 time steps, Δt = 0.05)

### 2. Boundary Surface Data
- **Location**: `example_coaxial_postpro_open/paraview/transient_boundary/`
- **Main file**: `transient_boundary.pvd`
- **Description**: Surface electromagnetic field data on boundaries
- **Time range**: 0 to 1.0 (21 time steps, Δt = 0.05)

### 3. Additional Data Files
- `domain-E.csv`: Electric field domain data
- `port-I.csv`: Port current data
- `port-V.csv`: Port voltage data
- `error-indicators.csv`: Mesh refinement indicators

## Quick Start

### Method 1: Using the provided script
```bash
# Open transient volume data (default)
./open_paraview.sh

# Open boundary surface data
./open_paraview.sh boundary

# Open both datasets
./open_paraview.sh both
```

### Method 2: Direct ParaView commands
```bash
# Volume data
paraview example_coaxial_postpro_open/paraview/transient/transient.pvd

# Boundary data
paraview example_coaxial_postpro_open/paraview/transient_boundary/transient_boundary.pvd
```

## Visualization Tips

### Basic Navigation
1. **Time Animation**: Use the play button in the toolbar to animate through time steps
2. **Camera Controls**: 
   - Left click + drag: Rotate
   - Middle click + drag: Pan
   - Scroll wheel: Zoom
3. **Reset View**: Click the "Reset Camera" button

### Common Electromagnetic Field Visualizations

#### 1. Electric Field (E)
- **Type**: Vector field
- **Units**: V/m
- **Visualization**: Use "Glyph" filter for arrows, or color by magnitude

#### 2. Magnetic Field (H)
- **Type**: Vector field  
- **Units**: A/m
- **Visualization**: Use "Streamline" filter for field lines

#### 3. Current Density (J)
- **Type**: Vector field
- **Units**: A/m²
- **Visualization**: Color by magnitude to show current flow

#### 4. Power Density
- **Type**: Scalar field
- **Units**: W/m³
- **Visualization**: Volume rendering or isosurfaces

### Useful Filters

#### Cross-sections
1. **Filters** → **Common** → **Slice**
2. Choose plane orientation (XY, XZ, YZ)
3. Adjust position with slider

#### Field Lines
1. **Filters** → **Common** → **Streamline**
2. Select vector field (E or H)
3. Adjust seed points and integration parameters

#### Isosurfaces
1. **Filters** → **Common** → **Contour**
2. Select scalar field
3. Set contour values

### Color Mapping
1. In **Properties** panel, change "Coloring" dropdown
2. Adjust color scale range in **Color Map Editor**
3. Use "Rescale to Data Range" for automatic scaling

## File Structure
```
example_coaxial_postpro_open/
├── palace.json                    # Simulation metadata
├── domain-E.csv                   # Electric field data
├── port-I.csv                     # Port current data  
├── port-V.csv                     # Port voltage data
├── error-indicators.csv           # Mesh quality data
└── paraview/
    ├── transient/
    │   ├── transient.pvd          # Main volume data file
    │   └── Cycle000000-000020/    # Individual time steps
    └── transient_boundary/
        ├── transient_boundary.pvd # Main boundary data file
        └── Cycle000000-000020/    # Individual time steps
```

## Simulation Details
- **Software**: Palace v0.12.0
- **Problem Type**: Electromagnetic transient analysis
- **Geometry**: Coaxial waveguide
- **DOF**: 12,840 degrees of freedom
- **Elements**: 128 mesh elements
- **Time Steps**: 21 (0 to 1.0, Δt = 0.05)

## Troubleshooting

### ParaView won't open
```bash
# Check if ParaView is properly installed
paraview --version

# If not found, reinstall
brew reinstall --cask paraview
```

### Data files not loading
- Ensure you're in the correct directory (`20250712_palace`)
- Check that file paths are correct
- Verify .pvd files exist and are readable

### Performance issues
- For large datasets, try loading only boundary data first
- Use "Decimation" filter to reduce mesh density
- Close unused data objects in Pipeline Browser
