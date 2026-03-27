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
- `scrape_comsol_release_media.sh`
  - wrapper for caching COMSOL release pages and extracting embedded Wistia media
- `scripts/comsol_release_media.py`
  - staged CLI for scraping release subpages, media metadata, and downloaded MP4s
- `build_comsol_release_browser.sh`
  - generates a compact local review page for all scraped COMSOL release videos
- `scripts/build_comsol_release_browser.py`
  - builds the static HTML browser grouped by release version with inline video playback
- `render_comsol_selected_montage.sh`
  - renders a selected COMSOL release shortlist into a single fixed-FPS H.264 montage with bottom subtitle cards
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

## COMSOL Release Media Workflow
This repo now includes a scraper workflow for COMSOL release highlight pages using the fixed release set:
- `5.4`
- `5.5`
- `5.6`
- `6.0`
- `6.1`
- `6.2`
- `6.3`
- `6.4`

It does the following:
- caches each release root page and its side-menu subpages under `build/comsol_release_scrape/cache/`
- scans each module page for embedded Wistia media IDs
- stores Wistia metadata JSON under `build/comsol_release_scrape/metadata/wistia/`
- writes per-page and per-release summaries under `build/comsol_release_scrape/metadata/` and `build/comsol_release_scrape/indexes/`
- optionally downloads the preferred MP4 asset for each embed under `build/comsol_release_scrape/media/`

Examples:

```bash
./scrape_comsol_release_media.sh scrape-release 5.4 --no-download-media
```

```bash
./scrape_comsol_release_media.sh scrape-all --no-download-media
```

```bash
./scrape_comsol_release_media.sh scrape-all
```

To limit a full run to selected releases:

```bash
./scrape_comsol_release_media.sh scrape-all --release 5.4 --release 6.4
```

Use `--force` to refetch pages or media even if a cached copy already exists.

## COMSOL Review Browser
To build a compact browser for reviewing the scraped release videos:

```bash
./build_comsol_release_browser.sh
```

This writes:
- `build/comsol_release_browser/index.html`

The browser groups videos by release version, shows the title and module slug, and plays clips in place. It defaults to a `sub 10 s` filter to make montage candidates easier to review quickly.
It also supports per-clip checkboxes, `Select visible` / `Unselect visible` actions, persistent selections in the browser, and export of the current shortlist as TSV or JSON.

## COMSOL Montage Workflow
The selected COMSOL release videos were manually shortlisted from the browser export and then deduplicated by exact `media_id`.

Important local artifacts:
- `build/comsol_release_browser/export_20260327.json`
  - original manual shortlist exported from the review browser
- `build/comsol_release_browser/export_20260327_dedup.json`
  - deduplicated shortlist used for the current final cut
- `build/comsol_release_selected_montage_1280x720_h264_dedup_subtitle_log.tsv`
  - per-clip subtitle log showing the exact generated caption, line 1, and line 2
- `build/comsol_release_selected_montage_1280x720_h264_dedup_with_audio.mp4`
  - current final COMSOL montage with audio

Render behavior:
- frame size: `1280x720`
- frame rate: fixed `30 fps`
- video codec: H.264 / `yuv420p` / `+faststart`
- audio source: first 5 minutes of `The Son of Flynn`
- audio fadeout: last `2 s`
- speed policy for selected clips:
  - `1x` below `1 s`
  - `2x` for `1 s` to `<2 s`
  - `3x` for `2 s` to `<4 s`
  - `4x` for `4 s` to `<10 s`
  - `5x` for `10 s` to `30 s`
- subtitle source:
  - derived from the local media filename
  - module slug removed
  - filler like `animation`, `RH`, and `RH61`-style markers removed
  - release version appended as `(vX.X)`

To rerender the deduplicated video-only montage:

```bash
OUTPUT_NAME=comsol_release_selected_montage_1280x720_h264_dedup.mp4 ./render_comsol_selected_montage.sh build/comsol_release_browser/export_20260327_dedup.json
```

To add audio afterward:

```bash
ffmpeg -nostdin -y -v warning \
  -i build/comsol_release_selected_montage_1280x720_h264_dedup.mp4 \
  -i build/son_of_flynn_first5min.m4a \
  -filter:a "atrim=0:185.256293,afade=t=out:st=183.256293:d=2" \
  -map 0:v:0 -map 1:a:0 \
  -c:v copy -c:a aac -b:a 192k -movflags +faststart -shortest \
  build/comsol_release_selected_montage_1280x720_h264_dedup_with_audio.mp4
```

Current deduplicated cut stats:
- selected clips before dedup: `110`
- selected clips after dedup: `103`
- accelerated video-only runtime: about `185.256 s`

## Notes
- The Lumerical `.mpg` files have unreliable metadata durations. The final visual result is usable, but any raw metadata-derived timings for those files should be treated cautiously.
- The render script derives output names from the active speed settings. Older local files may still carry legacy names from earlier speed policies.
