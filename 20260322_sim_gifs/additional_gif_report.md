# Additional GIF Report

## Scope
Scanned:
- `SOURCE_COMSOL_EXTRA_A`
- `SOURCE_COMSOL_EXTRA_B`

Compared against the existing montage sources:
- `SOURCE_LNSOI_SIM`
- `SOURCE_COMSOL_PRIMARY`

## Counts
- GIFs found in `SOURCE_COMSOL_EXTRA_A`: `34`
- GIFs found in `SOURCE_COMSOL_EXTRA_B`: `32`
- Combined raw total in the two new trees: `66`
- Unique new GIF hashes not already present in the existing montage library: `60`

## Duplicate Notes
- Several files are mirrored between the two new trees, especially:
  - `20241210_CTE_bending`
  - `20241218_dual_pol_horn_antenna`
- One file is already present in the existing montage library:
  - `20240709_mmw_PC/fullGeom_offResonant_20240709.gif`

## Best-Guess Topic Labels For Ambiguous Folders

### `SOURCE_COMSOL_EXTRA_A/20240501_BLDC`
- Nearby files are only `Untitled3.gif` and `Untitled_2D.gif`.
- Best guess: BLDC motor or rotating electromechanical toy study.
- Confidence: low to medium.

### `SOURCE_COMSOL_EXTRA_A/20240709_gear_box`
- Context file: `gearbox_vibration_noise_bearing.mph`
- Best guess: gearbox vibration / noise / bearing mechanics.
- Confidence: high.

### `SOURCE_COMSOL_EXTRA_A/20240709_katana`
- Context file: `Animation_Different Quantities Animation.mp4`
- Folder name suggests a katana-shaped or blade-like mechanics study.
- Best guess: mechanical vibration or stress animation on a katana / blade geometry.
- Confidence: medium.

### `SOURCE_COMSOL_EXTRA_A/20240709_yagi_antenna`
- Context file: `yagi_uda_antenna.mph`
- Best guess: Yagi-Uda antenna field or radiation behavior.
- Confidence: high.

### `SOURCE_COMSOL_EXTRA_A/20240810_thermal_cryo`
- Context files:
  - `model60_heat_transfer_3D_copper_plate_20240810.mph`
  - `model60_heat_transfer_3D_copper_plate_20240810_700s.mph`
- Best guess: cryogenic or transient thermal diffusion in a 3D copper plate.
- Confidence: high.

### `SOURCE_COMSOL_EXTRA_A/20240813_microwave_oven`
- Context files:
  - `model60_emw_3D_microwave_oven_20240813.mph`
  - `model60_emw_3D_microwave_oven_20240813_fundamental.mph`
- Best guess: electromagnetic cavity field in a microwave oven.
- Confidence: high.

### `SOURCE_COMSOL_EXTRA_B/20250815_plasma`
- Context file: `Untitled.mph.mph`
- Best guess: plasma-related electromagnetic toy or field visualization.
- Confidence: low.

## Strong New Topic Buckets
- Rotating / macro-mechanical systems:
  - BLDC
  - gearbox
  - katana
- Thermal:
  - thermal cryo copper plate
  - egg cooking
  - CTE bending
- RF / microwave / antennas:
  - Yagi antenna
  - slot antenna far field
  - mmWave star coupler
  - waveguide to coax
  - differential microstrip to WR10
  - dual-pol horn / Vivaldi
  - radar cross section of cars
  - microwave oven
- Photonics / optics:
  - X_HW_zine MMI / grating / coupler studies
  - anti-reflection surface microstructure
  - superlens
  - waveguide beam splitter
- Acoustics / mechanics toys:
  - acoustic taper toy
  - SAW toy
  - isolation bob

## Most Useful Additions For Diversity
These appear to broaden the montage the most relative to the current cut:
- Gearbox vibration / noise
- Yagi antenna
- Thermal cryo copper plate
- Microwave oven cavity field
- Slot antenna far field
- mmWave star coupler
- Radar cross section car study
- AR surface microstructure
- WG to coax transition
- X_HW_zine photonics clips
- Egg cooking thermal clip
- Acoustic taper toy
- SAW toy
- Differential microstrip to WR10
- Superlens
- Isolation bob

## Files Created For Follow-Up
- `montage_manifest_diversity_additions.tsv`

This file is a curated add-on manifest of new, deduped candidates chosen to increase topic diversity without touching the current montage files.
