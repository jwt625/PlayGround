# DevLog-000: Thin Film Optical Model for BTO Thickness Estimation

## Date
2026-02-03 to 2026-02-04

## Objective
Develop an optical model to estimate BaTiO3 (BTO) thin film thickness from color images of a 12-inch wafer. The goal is to:
1. Estimate absolute or relative film thickness from observed color
2. Estimate thickness uniformity from color variation across the wafer

## Background

### Physical System
- Substrate: Silicon (Si)
- Thin film: BaTiO3 (BTO), grown by MBE
- Stack structure: Air / BTO / Si (no oxide layer for MBE-grown films)
- Observation: Color image taken at approximately 56 degrees from normal

### Physics
Thin-film interference causes wavelength-dependent reflectance. When white light illuminates the film, constructive/destructive interference at different wavelengths produces characteristic colors that depend on:
- Film thickness
- Refractive indices of all layers
- Incident angle
- Polarization state

## Implementation

### Transfer Matrix Method
Implemented the standard transfer matrix formalism for multilayer thin films:

1. **Fresnel coefficients** at each interface for s and p polarizations
2. **Interface matrices** relating fields across boundaries
3. **Propagation matrices** for phase accumulation through layers
4. **Total transfer matrix** M = product of all interface and propagation matrices
5. **Reflection coefficient** r = M[1,0] / M[0,0]

Key equations:
- Snell's law: n1 * sin(theta1) = n2 * sin(theta2)
- Phase: delta = (2 * pi * n * d * cos(theta)) / lambda
- Reflectance: R = |r|^2

### Material Optical Constants

| Material | Model | Notes |
|----------|-------|-------|
| Air | n = 1.0 | Constant |
| BTO | Sellmeier dispersion | n^2 = 4.064 + 1.216*lambda^2/(lambda^2 - 0.177^2), gives n = 2.3-2.5 in visible |
| SiO2 | Malitson Sellmeier | For reference, not used in MBE stack |
| Si | Tabulated n,k | Aspnes & Studna (1983), 300-1000 nm, interpolated |

### Color Conversion Pipeline
Spectrum to RGB conversion using standard colorimetry:

1. Calculate reflectance spectrum R(lambda)
2. Multiply by illuminant spectrum I(lambda) - using D65 (daylight)
3. Integrate with CIE 1931 color matching functions to get XYZ
4. Convert XYZ to linear sRGB using standard matrix
5. Apply gamma correction (sRGB gamma ~ 2.4)
6. Clip to [0, 1] range

### Ellipsometry Calculations
Added ellipsometry parameter calculations:
- Psi: tan(Psi) = |rp| / |rs| (amplitude ratio)
- Delta: Delta = arg(rp) - arg(rs) (phase difference)

Configured for near-Brewster angle measurements:
- Brewster angle for Air/BTO: arctan(2.4) = 67.4 degrees
- Three angles: 62, 67, 72 degrees
- Wavelength range: 250-900 nm

## Files Created

### Core Modules
| File | Description |
|------|-------------|
| transfer_matrix.py | Transfer matrix calculations, reflectance, ellipsometry |
| materials.py | Wavelength-dependent refractive indices for Air, BTO, SiO2, Si |
| color_conversion.py | CIE 1931 colorimetry, sRGB conversion |

### Demo Scripts
| File | Description |
|------|-------------|
| demo_reflectance.py | Reflectance spectra visualization at 56 deg |
| demo_color.py | Thickness-to-color mapping at 56 deg |
| demo_ellipsometry.py | Ellipsometry Psi/Delta calculations near Brewster |

### Generated Outputs
| File | Description |
|------|-------------|
| reflectance_spectra_56deg.png | R vs wavelength for different thicknesses |
| reflectance_vs_thickness_56deg.png | R vs thickness, polarization comparison |
| color_vs_thickness_56deg.png | Color chart and RGB values vs thickness |
| ellipsometry_vs_wavelength_62_67_72deg.png | Psi, Delta spectra at three angles |
| ellipsometry_vs_thickness_67deg.png | Psi, Delta vs thickness at Brewster |
| psi_delta_trajectory_62_67_72deg.png | Psi-Delta trajectories |

## Key Decisions

1. **No SiO2 layer**: MBE-grown BTO on Si typically removes native oxide. SrTiO3 buffer layer is optically similar to BTO (n ~ 2.3-2.4) and can be neglected or treated as part of BTO thickness.

2. **56 degree incident angle**: Determined from image analysis showing specular reflection at this angle.

3. **Unpolarized light**: Assumed for camera observation, calculated as (Rs + Rp) / 2.

4. **D65 illuminant**: Standard daylight assumption for the light source.

5. **sRGB color space**: Standard for consumer cameras and displays.

## Sample Results

### Color Table at 56 degrees (selected values)
| Thickness (nm) | Hex Color |
|----------------|-----------|
| 0 | #9eaa8d |
| 50 | #624b3a |
| 100 | #83a58f |
| 150 | #a58d2e |
| 200 | #1b7b92 |
| 250 | #93a93f |
| 300 | #9e5d8e |

### Ellipsometry at 300 nm BTO, 67 degrees
| Wavelength | Psi | Delta |
|------------|-----|-------|
| 400 nm | 29.7 | -83.3 |
| 500 nm | 27.5 | 69.9 |
| 600 nm | 17.8 | -118.2 |
| 700 nm | 14.5 | 138.8 |

## Next Steps

1. Load and analyze actual wafer image (image.png)
2. Extract RGB values from different regions (center vs edge)
3. Create inverse lookup: RGB to thickness
4. Generate thickness map and uniformity metrics
5. Compare with ellipsometry measurements if available

## Dependencies
- numpy
- scipy
- matplotlib

Environment managed with uv virtual environment.

