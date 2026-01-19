# Hero Video Assembly Plan

**Status**: Complete
**Output**: `assets/video-output/realfab-hero-final.mp4`
**Date**: 2026-01-18

## Overview

- **Audio**: `assets/audio-narrator/realfab-hero-final.m4a` (70.98s)
- **Final Video**: 71.00s (time-stretched clips + 3s outro)
- **Resolution**: 1920x1080 @ 30fps
- **Structure**:
  - 0:00 - 0:30 → Old Big Fab (crisis, centralized manufacturing)
  - 0:30 - 1:08 → New Additive/Distributed Fab (positive vision)
  - 1:08 - 1:11 → Outro: "Build Real Fabs"

---

## Section 1: OLD BIG FAB (0:00 - 0:30)

| # | Clip | Trim | Speed | Output |
|---|------|------|-------|--------|
| 1 | Inside Micron Taiwan.mov | full 12.9s | 6x | 2.15s |
| 2 | Unveiling High NA EUV ASML.mov | 0-15s | 5x | 3s |
| 3 | RF GaN Fab Tour.mov | 0-15s | 5x | 3s |
| 4 | Processing Bell Labs 1979.mp4 | 1:50-2:50 (60s) | 20x | 3s |
| 5 | Siltronic insights.mov | 0-12s | 4x | 3s |
| 6 | ficonTEC Wafer-level.mp4 | 0:53-1:08 | 5x | 3s |
| 7 | SMD Bestückung.mp4 | 0:15-0:23 | 4x | 2s |
| 8 | AWG PLC Auto Align.mp4 | 0-30s | 10x | 3s |
| 9 | Nordson ASYMTEK NexJet.mov | 0-12s | 4x | 3s |
| 10 | Turbo pump explode.mov | full 4.7s | 1x | 4.7s |

---

## Section 2: NEW ADDITIVE/DISTRIBUTED FAB (0:30 - 1:08)

| # | Clip | Trim | Speed | Output |
|---|------|------|-------|--------|
| 11 | Microscale 3D printing spaceship.mov | full 40s | 8x | 5s |
| 12 | EHLA Extreme High-speed Laser.mov | 0-25s | 5x | 5s |
| 13 | Eplus3D Metal 3D Printers.mov | full 20s | 8x | 2.5s |
| 14 | TruLaser Cell 7040.mov | full 15s | 4x | 3.75s |
| 15 | 5-Axis Cutting Silicon Carbide.mov | full 20s | 5x | 4s |
| 16 | SparkNano Omega.mov | 0-25s | 5x | 5s |
| 17 | xTool P3.mp4 | 0:02-0:14 | 3x | 4s |
| 18 | xTool F2.mp4 | 0:16-0:28 | 3x | 4s |
| 19 | Makera Z1 Unicorn.mov | full 10s | 2.5x | 4s |

---

## Section 3: OUTRO (1:08 - 1:11)

- **Clip**: `assets/video-output/outro_3s.mp4`
- Animation: Word-by-word reveal ("Build" → "Real" → "Fabs"), fade out

---

## Build Process

1. `build_hero_video.sh` - Processes 19 source clips into normalized segments
2. `concatenate_hero.sh` - Concatenates clips, time-stretches to match audio, appends outro, adds audio track

Time-stretch factor applied: 1.012x (clips sped up 1.2% to match audio duration minus outro)

---

## Technical Notes

### Scaling Strategy
- 1920x1080 clips: use as-is
- 1324x740 clips: scale up with padding (pillarbox) or crop-to-fit
- Other resolutions: scale + pad black bars to maintain aspect ratio

### Speed-up Method
Use `setpts` filter with frame dropping (no interpolation):
```
ffmpeg -i input.mp4 -filter:v "setpts=PTS/2" -r 30 output.mp4
```

### Encoding
- Codec: H.264 (libx264)
- Preset: medium
- CRF: 18-20
- Pixel format: yuv420p
- Faststart: +movflags faststart

---

## Outro

```bash
ffmpeg -y -f lavfi -i color=c=black:s=1920x1080:r=30:d=3 \
  -vf "
    drawtext=text='Build':fontfile='/System/Library/Fonts/Supplemental/Arial Bold.ttf':fontsize=120:fontcolor=white:x=490:y=(h/2)-60+60*(1-min(t/0.6\,1)):alpha='if(lt(t,0.6),t/0.6,if(lt(t,2.5),1,(3-t)/0.5))',
    drawtext=text='Real':fontfile='/System/Library/Fonts/Supplemental/Arial Bold.ttf':fontsize=120:fontcolor=white:x=860:y=(h/2)-60+60*(1-min((t-0.3)/0.6\,1)):alpha='if(lt(t,0.3),0,if(lt(t,0.9),(t-0.3)/0.6,if(lt(t,2.5),1,(3-t)/0.5)))',
    drawtext=text='Fabs':fontfile='/System/Library/Fonts/Supplemental/Arial Bold.ttf':fontsize=120:fontcolor=white:x=1170:y=(h/2)-60+60*(1-min((t-0.6)/0.6\,1)):alpha='if(lt(t,0.6),0,if(lt(t,1.2),(t-0.6)/0.6,if(lt(t,2.5),1,(3-t)/0.5)))'
  " \
  -c:v libx264 -preset medium -crf 18 -pix_fmt yuv420p \
  -movflags +faststart \
  assets/video-output/outro_3s.mp4
```



