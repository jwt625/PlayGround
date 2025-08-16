# YouTube Liked Videos Backup Chrome Extension

A Chrome extension to backup metadata from YouTube liked videos with automatic removal capability to circumvent the 5000 video limit.

## Features

- **Comprehensive Backup**: Scrapes complete metadata including titles, channels, durations, thumbnails, and more
- **Smart Verification**: Multi-source data validation ensures backup integrity
- **Safe Removal**: Automatically removes videos from liked list after successful backup
- **Multiple Export Formats**: Export data as JSON or CSV
- **Rate Limiting**: Respects YouTube's limits to avoid detection
- **Progress Tracking**: Real-time progress indicators and error reporting

## Installation

### Development Installation

1. Clone or download this repository
2. Open Chrome and navigate to `chrome://extensions/`
3. Enable "Developer mode" in the top right
4. Click "Load unpacked" and select the `youtube-liked-backup` folder
5. The extension should now appear in your extensions list

### Usage

1. Navigate to your YouTube liked videos page (`https://www.youtube.com/playlist?list=LL`)
2. Click the extension icon in the toolbar
3. Click "Start Backup" to begin the backup process
4. Monitor progress in the popup
5. Export your data using the export buttons

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

## Development

### Prerequisites
- Chrome browser
- Basic knowledge of JavaScript and Chrome Extensions

### Building
No build process required - this is a vanilla JavaScript extension.

### Testing
1. Load the extension in developer mode
2. Navigate to YouTube liked videos
3. Test backup functionality
4. Check browser console for errors

## Limitations

- Requires manual navigation to YouTube liked videos page
- Subject to YouTube's rate limiting and anti-bot measures
- Cannot access private or deleted videos
- Limited by Chrome extension storage quotas

## Privacy

- All data processing happens locally
- No external servers or data transmission
- User controls all data export and deletion

## License

This project is for educational purposes. Please respect YouTube's Terms of Service.

## Contributing

This is part of a larger playground repository. Feel free to suggest improvements or report issues.

## Support

For issues or questions, please refer to the main repository documentation or create an issue in the GitHub repository.
