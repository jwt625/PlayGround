# Nature Video Player Chrome Extension

A Chrome extension that replaces Nature supplementary video download links with inline video players.

## Features

- Plays supplementary videos directly on Nature paper pages
- No need to download videos
- Full video controls (play, pause, volume, fullscreen)
- Playback speed control (0.5x to 2x)
- Responsive player that adapts to screen size
- Automatically detects and replaces all video links

## Installation

### Method 1: Load Unpacked Extension (Development)

1. Open Chrome and navigate to `chrome://extensions/`
2. Enable "Developer mode" (toggle in top right)
3. Click "Load unpacked"
4. Select the extension folder (`20260207_chrome_extension_nature_supplementary_video`)
5. The extension is now installed!

### Method 2: Create Icons First (Optional)

The extension needs icon files. You can either:
- Use the placeholder icons (will be created automatically)
- Replace them with custom icons (16x16, 48x48, 128x128 PNG files)

## Usage

1. Navigate to any Nature paper with supplementary videos
2. The extension automatically replaces video download links with inline players
3. Click play to watch videos directly on the page
4. Use the controls for playback speed, volume, and fullscreen

## Supported Sites

- nature.com
- springer.com (Nature's publisher)

## Technical Details

- Uses Video.js for robust video playback
- Supports multiple video formats (MP4, MOV, AVI, WebM, etc.)
- Monitors page for dynamic content changes
- Lightweight and fast

## Troubleshooting

**Videos not showing?**
- Check that the page has supplementary videos
- Ensure the extension is enabled in `chrome://extensions/`
- Try refreshing the page

**Player not loading?**
- Check browser console for errors
- Ensure internet connection (Video.js loads from CDN)

## Privacy

This extension:
- Does NOT collect any data
- Does NOT track your browsing
- Only runs on Nature/Springer domains
- All processing happens locally in your browser

