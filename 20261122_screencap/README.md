# ScreenCap

Minimal macOS menu bar screen recorder: drag a region like Cmd+Shift+5, record it with **system output audio** (ScreenCaptureKit, no BlackHole/OBS), then trim the start/end with a QuickTime-style trim bar.

## Build / install
```
./build.sh            # -> build/ScreenCap.app (ad-hoc signed)
./build.sh install    # also copies to ~/Applications/ScreenCap.app
```
Grant **System Settings > Privacy & Security > Screen & System Audio Recording > ScreenCap**. Ad-hoc signing means each rebuild needs the permission re-granted.

## Use
- **Cmd+Shift+7** (or menu bar icon > Record Region): drag to select, drag inside to move, drag edges/corners to resize. **Enter**, **Record**, or double-click starts; **Esc** cancels. Last region is preselected.
- Stop: **Cmd+Shift+7** again or click `REC m:ss` in the menu bar.
- Trim window opens on stop: drag the yellow handles, **Trim** keeps the selection (lossless passthrough; original goes to Trash), **Cancel** keeps the file as recorded.
- Output: `~/Movies/ScreenCap/ScreenCap YYYY-MM-DD HH.MM.SS.mp4`, H.264 + AAC 48 kHz stereo, faststart, native pixel resolution (capped at 4096 px max side), up to 60 fps, variable frame rate.

## Test modes
```
ScreenCap.app/Contents/MacOS/ScreenCap --test 5 --region x,y,w,h   # record (global points), print path
ScreenCap.app/Contents/MacOS/ScreenCap --trim FILE START_S END_S    # headless trim
ScreenCap.app/Contents/MacOS/ScreenCap --edit FILE                  # open trim window on a file
```

## Current state
See `DevLog/DevLog-000-region-capture-with-system-audio.md`.
