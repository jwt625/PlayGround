# YouTube Liked Videos Backup Chrome Extension

A Chrome extension to backup metadata from YouTube liked videos with automatic removal capability to circumvent the 5000 video limit.

## Features

- **Comprehensive Backup**: Scrapes complete metadata including titles, channels, durations, thumbnails, and more
- **Smart Verification**: Multi-source data validation ensures backup integrity
- **Safe Removal**: Automatically removes videos from liked list after successful backup
- **Multiple Export Formats**: Export data as JSON or CSV
- **Rate Limiting**: Respects YouTube's limits to avoid detection
- **Progress Tracking**: Real-time progress indicators and error reporting

## Installation & Usage

### Installation
1. Open Chrome and navigate to `chrome://extensions/`
2. Enable "Developer mode" (top-right toggle)
3. Click "Load unpacked" and select the `youtube-liked-backup` folder
4. Extension should load without errors

### Usage
1. Navigate to YouTube liked videos: `https://www.youtube.com/playlist?list=LL`
2. Click the extension icon in toolbar
3. Click "Start Backup" to begin backup process
4. Monitor progress and export data as needed

### Troubleshooting
- **Extension won't load**: Check for errors at `chrome://extensions/`
- **Background script errors**: Click "Inspect views: service worker" to debug
- **Popup shows loading**: Reload extension and try again
- **Content script issues**: Check browser console (F12) on YouTube page

## Architecture

### File Structure

```
youtube-liked-backup/
├── manifest.json                 # Extension configuration
├── background/
│   ├── background.js            # Service worker
│   ├── storage-manager.js       # Data storage management
│   └── export-manager.js        # Data export functionality
├── content/
│   ├── content.js              # Main content script
│   ├── video-scraper.js        # Video metadata extraction
│   ├── pagination-handler.js   # Infinite scroll handling
│   └── removal-handler.js      # Video removal functionality
├── popup/
│   ├── popup.html              # Extension popup
│   ├── popup.js                # Popup logic
│   └── popup.css               # Popup styling
├── options/
│   ├── options.html            # Settings page
│   ├── options.js              # Settings logic
│   └── options.css             # Settings styling
└── utils/
    ├── constants.js            # Configuration constants
    ├── data-schemas.js         # Data structures
    ├── data-validator.js       # Data validation
    └── youtube-api.js          # YouTube utilities
```

### Data Schema

Each video record includes:
- Video ID, title, channel information
- Duration, view count, upload date
- Thumbnails in multiple resolutions
- Backup status and verification score
- Timestamps for tracking

## Configuration

Access settings through the extension options page:

- **Backup Settings**: Auto-removal, verification thresholds
- **Export Settings**: Default formats, included data
- **Performance Settings**: Rate limits, delays, retries
- **Data Management**: View stats, export data, clear storage

## Technical Details

### Storage
- Primary: Chrome Storage API (~5MB limit)
- Fallback: IndexedDB for larger datasets
- Automatic quota management

### Rate Limiting
- Configurable scroll delays
- Removal rate limiting (default: 20/minute)
- Background tab navigation limits

### Data Validation
- Multi-level verification (Basic, Standard, Complete)
- Verification scoring (0-100)
- Missing field detection

## Development & Technical Notes

### Current Status
- ✅ Phase 1 (Core Functionality) - Complete
- 🔄 Phase 2 (Verification & Removal) - Planned
- 📋 Phase 3 (Advanced Features) - Planned

### Known Issues
- Icons not implemented (uses default Chrome icon)
- Background script initialization may need reload on first install
- Network request interception not yet implemented

### Architecture Notes
- Manifest V3 service worker architecture
- Dual storage system (Chrome Storage + IndexedDB fallback)
- Modular design with clear separation of concerns
- Rate limiting and error handling throughout

## Limitations & Privacy

**Current Limitations:**
- Manual navigation to YouTube liked videos required
- Subject to YouTube's anti-bot measures
- Cannot access private/deleted videos
- Chrome extension storage quotas apply

**Privacy:**
- All processing happens locally
- No external servers or data transmission
- User controls all data and exports

**License:** Educational use only. Respect YouTube's Terms of Service.
