# Supplementary Video Player

Chrome extension that replaces supplementary video download links on journal pages with inline players.

Supports MP4, WebM, MOV, AVI, MKV (non-native formats transcoded via FFmpeg.wasm). Includes playback speed control (0.5x-2x) and fullscreen.

## Setup

After cloning, download `ffmpeg-core.wasm` and place it in `ffmpeg/`:

```bash
curl -L https://unpkg.com/@ffmpeg/core@0.12.6/dist/esm/ffmpeg-core.wasm -o ffmpeg/ffmpeg-core.wasm
```

## Install

1. Go to `chrome://extensions/`, enable Developer mode
2. Click "Load unpacked", select this folder


## Test examples

mp4:
- https://www.nature.com/articles/s41565-025-02109-6#Sec33

avi:
- https://www.nature.com/articles/s41565-024-01779-y#Sec25

Five videos, zipped:
- https://www.science.org/doi/10.1126/science.aeb3637

Single, zipped:
- https://www.science.org/doi/10.1126/science.aec5660

TODO:
- Phys Rev: https://journals.aps.org/prl/abstract/10.1103/6cdq-1nvv