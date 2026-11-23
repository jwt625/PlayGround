# DevLog-000: Region screen capture with system audio

| Field | Value |
|---|---|
| Date | 2026-11-22 |
| Status | Working (manually verified 2026-11-22) |
| Scope | Small native macOS menu bar tool: drag-select a region (like Cmd+Shift+5), record it to MP4 with system output audio |

## Requirements
- As easy as the built-in capture tool: global hotkey, drag a region, adjust it, press Enter to record.
- Captures system output audio (what you hear), not just the microphone.
- No OBS, no BlackHole dependency.

## Design
- ScreenCaptureKit (macOS 13+): `SCStreamConfiguration.capturesAudio = true` returns the system audio mix directly; `sourceRect` crops the display (points, display-local, top-left origin).
- AppKit menu bar app (`LSUIElement`), single `main.swift`, built with `swiftc` into `ScreenCap.app` (ad-hoc signed) by `build.sh`. No Xcode project.
- Region selector: borderless overlay window per screen; drag to draw, drag inside to move, drag edges/corners to resize; Enter or the Record button starts; Esc cancels. Last region is remembered and preselected.
- Hotkey Cmd+Shift+7 (Carbon `RegisterEventHotKey`, no Accessibility permission): opens the selector, or stops if recording. Clicking the menu bar item while recording also stops.
- Writer: AVAssetWriter, H.264 (VideoToolbox) + AAC 48 kHz stereo 192 kb/s, MP4, `shouldOptimizeForNetworkUse` (moov at front). Native pixel resolution (backing scale), even dimensions, up to 60 fps.
- SCK only delivers frames when the screen changes; the last frame is re-appended at stop so video and audio durations match. The file is variable frame rate.
- Own app windows (overlay, red recording border) are excluded from capture via `SCContentFilter(excludingApplications:)`.
- Output: `~/Movies/ScreenCap/ScreenCap YYYY-MM-DD HH.MM.SS.mp4`, revealed in Finder at stop.

- Trim (added 2026-11-22 on request): after stop, a window opens on the recording in AVPlayerView trimming mode (QuickTime-style trim bar). Trim -> AVAssetExportSession passthrough with timeRange to `<name>.trimming.mp4`, original moved to Trash, trimmed file takes the original name. Cancel or closing the window keeps the file untouched.

## TODO
- [x] main.swift: app, menu, hotkey, overlay selector, recorder
- [x] build.sh: compile, bundle, ad-hoc sign
- [x] Install to ~/Applications
- [x] Trim window after recording + headless `--trim FILE START END` test mode, `--edit FILE` to open the trim window on any file
- [x] Trim accuracy verified on a synthetic 60 fps clip (GOP 2 s): cut 1.3-7.7 s -> 6.400 s, first/last displayed frames = source frames 78/461 (frame MD5 match). Passthrough keeps preroll samples and hides them with an MP4 edit list.
- [ ] Recording self-test (`--test 5 --region ...` with a system sound) verified with ffprobe + volumedetect: blocked on Screen Recording permission for the rebuilt binary
- [x] Manual UI check by Wentao (2026-11-22): recording with audio and trim confirmed working
- [x] README

## Progress log
- 2026-11-22: Plan written. Machine already has BlackHole + Multi-Output Device (headphones + BlackHole); not needed by this tool.
- 2026-11-22: First run crashed: AVAssetWriterInputPixelBufferAdaptor was created after startWriting() (NSInvalidArgumentException). Fixed by creating it before startWriting.
- 2026-11-22: Ad-hoc signing pins TCC Screen Recording permission to the cdhash, so every rebuild needs a re-grant. No code-signing identity on the machine (`security find-identity` empty). A self-signed cert would make it stable; not done.
- 2026-11-22: Trim window added and export verified (see TODO).
