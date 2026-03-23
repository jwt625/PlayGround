# PAS Simulation Montage

This directory contains the notes and render scripts used to assemble a vertical montage from a large personal simulation library spanning mechanics, electromechanics, acoustics, mmWave, photonics, and a small curated Lumerical subset.

## Final State
- Main GIF-only cut length: about `90 s`
- Extended final cut length with appended slow-wave GIFs and still-image tail: `161.133333 s`
- Fixed output frame: `1080x1350` (`4:5`)
- Subtitle style: two-line white text with black stroke
- Current speed policy:
  - default non-Lumerical clips: `3x`
  - very short non-Lumerical clips: relaxed to `2x` or `1.5x`
  - curated Lumerical clips: `20x`
- Slow-wave toy additions appended later:
  - `Slow Wave Toy / Linear Chain`
  - `Slow Wave Toy / Ring Chain`
  - both rendered at `2x`
- A soundtracked export was produced using a trimmed intro segment from `The Son of Flynn`
- A still-image scout / append workflow was also built for extending the montage with rapid-fire static frames

## What Was Done
- inventoried GIFs from the main simulation trees
- sampled representative media and nearby project files to infer topic labels
- wrote a chaptered montage treatment
- built progressively more diverse manifests
- added a subtitle-capable FFmpeg/ImageMagick render pipeline
- added a curated Lumerical subset with manual dedup
- tuned clip speed to keep short COMSOL shots readable
- generated per-shot timing reports for the final cut
- appended two late-added slow-wave toy GIFs with matching subtitle cards
- built a separate still-image preview pipeline with a two-stage duration ramp for scouting static inserts

## Useful Files
- `render_montage_final_mp4.sh`
  - current render pipeline for the curated final cut
- `render_still_shortlist_preview.sh`
  - preview renderer for the still-image scout using a duration ramp
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
- still-image shortlist manifests and notes that embed local source paths
- older exploratory render scripts superseded by the final renderer

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

For the still-image scout preview:

```bash
SECOND_STAGE_COUNT=58 MID_SECONDS=0.1 END_SECONDS=0.02 ./render_still_shortlist_preview.sh
```

This expects a local `still_image_shortlist.tsv` manifest, which is intentionally not tracked because it contains machine-specific source paths.

## Current Outputs
- Current soundtracked extended output:
  - `build/montage_manifest_final_with_son_of_flynn_plus_gifs_and_stills_fixed_v2.mp4`
- Current still scout preview:
  - `build/still_image_shortlist_preview.mp4`

## Notes
- The Lumerical `.mpg` files have unreliable metadata durations. The final visual result is usable, but any raw metadata-derived timings for those files should be treated cautiously.
- The render script derives output names from the active speed settings. Older local files may still carry legacy names from earlier speed policies.
