# PAS Simulation Montage

This directory contains the notes and render script used to assemble a short montage from a large personal simulation library spanning mechanics, electromechanics, acoustics, mmWave, photonics, and a small curated Lumerical subset.

## Final State
- Final cut length: about `90 s`
- Fixed output frame: `1080x1350` (`4:5`)
- Subtitle style: two-line white text with black stroke
- Current speed policy:
  - default non-Lumerical clips: `3x`
  - very short non-Lumerical clips: relaxed to `2x` or `1.5x`
  - curated Lumerical clips: `20x`
- A soundtracked export was also produced using a trimmed intro segment from `The Son of Flynn`

## What Was Done
- inventoried GIFs from the main simulation trees
- sampled representative media and nearby project files to infer topic labels
- wrote a chaptered montage treatment
- built progressively more diverse manifests
- added a subtitle-capable FFmpeg/ImageMagick render pipeline
- added a curated Lumerical subset with manual dedup
- tuned clip speed to keep short COMSOL shots readable
- generated per-shot timing reports for the final cut

## Useful Files
- `render_montage_final_mp4.sh`
  - current render pipeline for the curated final cut
- `montage_script.md`
  - high-level editorial treatment and chapter structure
- `additional_gif_report.md`
  - notes from scanning extra COMSOL trees and identifying new topic buckets

## Local-Only Artifacts
The following were intentionally left as local working artifacts and should not be tracked:
- build outputs under `build/`
- local source manifests that contain machine-specific absolute paths
- sampled frames and review copies
- downloaded soundtrack source audio

## Source Labels Used In Notes
- `SOURCE_LNSOI_SIM`
  - the original LN SOI simulation tree
- `SOURCE_COMSOL_PRIMARY`
  - the main COMSOL archive used for the montage
- `SOURCE_COMSOL_EXTRA_A`
  - an additional COMSOL tree scanned for diversity
- `SOURCE_COMSOL_EXTRA_B`
  - another additional COMSOL tree scanned for diversity

## Rebuilding Locally
Run:

```bash
./render_montage_final_mp4.sh montage_manifest_final_mp4_lumerical_manual_dedup.tsv
```

Environment variables can override the defaults, for example:

```bash
LUMERICAL_SPEED=20 BASE_SPEED=3 FPS=30 ./render_montage_final_mp4.sh montage_manifest_final_mp4_lumerical_manual_dedup.tsv
```

## Notes
- The Lumerical `.mpg` files have unreliable metadata durations. The final visual result is usable, but any raw metadata-derived timings for those files should be treated cautiously.
- The current renderer names outputs from the chosen speed settings, so future rebuilds will reflect the active speed policy in the filename.
