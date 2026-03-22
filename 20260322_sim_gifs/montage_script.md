# PAS Simulation Montage Script

## Working Title
**From Phononic Bands to Integrated Waves**

## Format
- Short research montage
- Organize by topic first, then by increasing system complexity
- Use as many GIFs as possible
- Speed up clips when needed to preserve momentum
- Omit only the weakest near-duplicates or clips that truly break flow
- Favor clips with strong motion, clear field patterns, or obvious geometric progression

## Library Stats
- Total GIFs available: `60`
- Source split: `7` from `SOURCE_LNSOI_SIM` and `53` from `SOURCE_COMSOL_PRIMARY`
- Major buckets:
  - `24` mechanics / electromechanics from the older Si + LN / LNSOI branches
  - `19` GIFs in `LN/mechanics`
  - `9` GIFs in `mmwave`
  - `7` GIFs in `acoustic`
  - `4` GIFs in `LN/optics`
  - `8` GIFs in later `Si` optics / taper / demo folders

## Runtime and Cut Estimates
- If all `60` GIFs are used, the montage has `60` shots and at least `59` direct cuts.
- If `55` GIFs are used, the montage has `55` shots and at least `54` direct cuts.
- If `50` GIFs are used, the montage has `50` shots and at least `49` direct cuts.

## Current Build
- Build status: implemented and rendered
- Ordering file: `montage_manifest.tsv`
- Render pipeline: `render_montage.sh`
- Frame format: fixed `4:5` mobile-friendly frame at `1080x1350`
- Aspect-ratio handling: preserve original geometry and pad black space instead of stretching
- Frame rate: `30 fps`
- Long clips: sped up when longer than `2.4` seconds, targeting about `2.1` seconds
- Very short clips: held to at least `1.35` seconds if needed

### Actual render stats
- GIFs used in current cut: `60 / 60`
- Raw total GIF duration: `128.54` seconds
- Rendered montage duration: `108.8` seconds
- Shot count: `60`
- Direct cuts: `59`
- Average rendered shot length: about `1.81` seconds

### Current outputs
- MP4: `build/pas_montage_4x5.mp4` (`33,080,969` bytes)
- WebM: `build/pas_montage_4x5.webm` (`23,529,529` bytes)
- Per-clip timing report: `build/report.tsv`

### Decent runtime targets
- At `1.5` seconds per GIF on average:
  - `60` GIFs -> about `90` seconds
  - `55` GIFs -> about `83` seconds
  - `50` GIFs -> about `75` seconds
- At `2.0` seconds per GIF on average:
  - `60` GIFs -> about `120` seconds
  - `55` GIFs -> about `110` seconds
  - `50` GIFs -> about `100` seconds
- At `2.5` seconds per GIF on average:
  - `60` GIFs -> about `150` seconds
  - `55` GIFs -> about `138` seconds
  - `50` GIFs -> about `125` seconds

### Recommended target
- Best balance for “use most of the library” is roughly `50` to `58` GIFs.
- A good final runtime is about `1:45` to `2:30`.
- That implies an average on-screen time of roughly `1.8` to `2.6` seconds per GIF, with speed-up applied where the original animations are slow.

## Editorial Principles
- Keep similar topics adjacent in time.
- Start with simpler, more abstract visuals.
- Move toward driven devices, full systems, and multi-physics complexity.
- End on the cleanest and most legible optical / EM visuals.
- Default to inclusion.
- Only remove a GIF if it is a true near-duplicate, visually weak, or interrupts the chapter flow.

## Chapter 1: Foundations
**Theme:** Mechanical modes, dispersion, and the earliest phononic intuition.

### Visual role
Introduce the language of the work: band diagrams, guided modes, localized displacement, and periodic structures.

### Suggested clip types
- Si mechanics band plots
- Si floating / half-anchored waveguide displacement fields
- Early LN / LNSOI mechanical mode shapes

### Candidate folders
- `SOURCE_COMSOL_PRIMARY/Si/mechanics/20190321_floatingSMWG`
- `SOURCE_COMSOL_PRIMARY/Si/mechanics/20190327_halfAnchoredSMWG`
- `SOURCE_LNSOI_SIM/electromechanics/20190530_SiBlock`

### Notes
- Open with a band diagram or dispersion sweep.
- Cut into several displacement-field animations to turn abstraction into structure.
- Keep pacing moderate and clean.

## Chapter 2: Electromechanics
**Theme:** IDTs and piezo-driven wave transport in LN / LNSOI structures.

### Visual role
Show the jump from passive mechanical structures to electrically driven devices and guided propagation.

### Suggested clip types
- LNSOI IDT-driven waveguides
- PGT taper transitions
- Guided electromechanical transport into structured waveguides

### Candidate folders
- `SOURCE_LNSOI_SIM/electromechanics/20190531_LNSOIIDT`
- `SOURCE_LNSOI_SIM/electromechanics/20190821_PGT2SMWG2OMC`

### Notes
- Sequence should feel like a system becoming purposeful.
- Favor clips where energy is visibly launched, tapered, or funneled.
- If both `PGT2OMC` and `PGT2SMWG2OMC` are used, place the simpler geometry first.
- This chapter can carry several related variants because the story benefits from visible progression.

## Chapter 3: Coupling and Cavities
**Theme:** Localization, coupling efficiency, and device-level mechanical design.

### Visual role
Shift from transport to confinement and coupling between elements.

### Suggested clip types
- PNC to waveguide coupling
- Nanobeam coupling structures
- Localized mechanical cavity behavior

### Candidate folders
- `SOURCE_COMSOL_PRIMARY/LN/mechanics/20190319_PNC`
- `SOURCE_COMSOL_PRIMARY/LN/mechanics/20200925_LiSa_mech_NB_coupling`

### Notes
- This chapter can be short and precise.
- Use clips that make localization or coupling visually obvious.
- Treat this as a bridge into more complete transducer systems.
- Keep most clips here unless two files read as visually identical.

## Chapter 4: Transducers and Acousto-Optic Systems
**Theme:** Full devices, multi-phase excitation, flip-chip stacks, and more integrated architectures.

### Visual role
Raise the perceived engineering complexity.

### Suggested clip types
- Flip-chip LN / Si mechanics
- Four-phase and two-phase IDT comparisons
- Full-IDT displacement or stress fields
- FBAR / chirped-IDT concepts

### Candidate folders
- `SOURCE_COMSOL_PRIMARY/LNSOI/flipchip/20210124_AOM`
- `SOURCE_COMSOL_PRIMARY/LN/mechanics/20210112_LNOS_flipchip`
- `SOURCE_COMSOL_PRIMARY/LN/mechanics/20210211_four_phase_IDT`
- `SOURCE_COMSOL_PRIMARY/LN/mechanics/20220712_FBAR`

### Notes
- This is the central high-complexity chapter.
- Progress from compact structures to visibly larger, fuller system views.
- If comparing two-phase vs four-phase IDTs, keep the contrast tight and immediate.
- End this chapter on one of the strongest full-device stress or displacement visuals.
- This is one of the best places to keep multiple related variants because the increasing complexity reads well on screen.

## Chapter 5: Radiation Into Surroundings
**Theme:** Acoustic waves leaving the structure and interacting with a medium.

### Visual role
Provide a visual reset while keeping the physics thread intact.

### Suggested clip types
- LN in water radiation
- Directional or vertical radiation variants

### Candidate folders
- `SOURCE_COMSOL_PRIMARY/acoustic/20201218_LN_in_water`

### Notes
- Use this as a short interlude, not a long section.
- The circular radiation patterns are visually distinct and should read immediately.
- Include several radiation variants if they feel distinct enough in propagation pattern or geometry.

## Chapter 6: High-Frequency Electromagnetics
**Theme:** Microwave and mmWave waveguide transitions, probes, and photonic-crystal-like EM structures.

### Visual role
Expand from mechanics and acoustics into electromagnetic field engineering.

### Suggested clip types
- WR10 transitions
- Radial probe / loop antenna structures
- mmWave photonic crystal or defect-waveguide field patterns

### Candidate folders
- `SOURCE_COMSOL_PRIMARY/mmwave/20210913_WR10_TWM`
- `SOURCE_COMSOL_PRIMARY/mmwave/20211204_photonic_crystal`

### Notes
- Let the montage feel broader here, not disconnected.
- Choose clips with strong field propagation and visible confinement.
- Keep most WR10 variants unless they are frame-for-frame repetitive.

## Chapter 7: Optical Coupling and Clean Endings
**Theme:** Optical transport, tapers, and polished field propagation visuals.

### Visual role
End with the clearest and most elegant imagery.

### Suggested clip types
- Fiber to LN waveguide coupling
- Thickness ramp and cone taper propagation
- Simple Si optical waveguide visuals

### Candidate folders
- `SOURCE_COMSOL_PRIMARY/LN/optics/20220901_thickness_ramp`
- `SOURCE_COMSOL_PRIMARY/Si/20220901_ewfd_test`
- `SOURCE_COMSOL_PRIMARY/Si/20220916_UHNA3_taper`
- `SOURCE_COMSOL_PRIMARY/Si/20221026_defense_toys`

### Notes
- This chapter should feel lighter and cleaner than the mechanical sections.
- Favor red-blue optical field propagation clips with clear longitudinal flow.
- A simple Si wire optics clip can work as the closing image.
- These clips are especially useful near the end because they remain readable even at higher playback speed.

## Recommended Overall Arc
1. Band plots and mechanical waveguides
2. LN / LNSOI electromechanical transport
3. Coupling and cavity devices
4. Integrated transducers and acousto-optic systems
5. Acoustic radiation in fluid
6. mmWave / EM structures
7. Optical taper and waveguide visuals

## Selection Strategy
- Pull most or all clips from each chapter.
- Treat `50` GIFs as the practical floor for a “broad survey” version.
- Treat `55` to `60` GIFs as the target for a “maximal montage” version.

### Suggested chapter allocation for a near-complete cut
- Chapter 1: `8` to `10` GIFs
- Chapter 2: `5` to `7` GIFs
- Chapter 3: `3` to `5` GIFs
- Chapter 4: `10` to `14` GIFs
- Chapter 5: `3` to `5` GIFs
- Chapter 6: `7` to `9` GIFs
- Chapter 7: `7` to `10` GIFs

This yields a montage of roughly `43` to `60` clips, with the intended target closer to the top of that range.

## What To Omit First
- Near-duplicate parameter sweeps that do not materially change the visual story
- Variants that are harder to read than a cleaner clip from the same folder
- Clips that require too much technical context to parse quickly
- Any true outlier that interrupts the mechanics -> electromechanics -> EM / optics progression

## Rough Tone Script
Use this as the editorial voice for pacing and sequencing:

> Start with structure, symmetry, and dispersion.  
> Move into guided mechanical motion.  
> Introduce electrical drive and directed transport.  
> Tighten into coupling, confinement, and engineered devices.  
> Expand outward into radiation and surrounding media.  
> Shift scale into microwave and electromagnetic transport.  
> Finish with the clearest optical propagation shots, where the geometry and the wave both read instantly.

## Next Pass
- Build a file-by-file shortlist inside each chapter
- Mark a single preferred GIF for each folder
- Estimate per-clip duration and transitions
- Decide whether the cut targets `50+` GIFs or the full `60`
