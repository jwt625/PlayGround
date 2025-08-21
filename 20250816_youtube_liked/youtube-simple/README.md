# YouTube Simple Extension

A lightweight Chrome extension for managing YouTube videos with basic scraping and removal functionality.

## Installation

1. Open Chrome and navigate to `chrome://extensions/`
2. Enable "Developer mode" (top-right toggle)
3. Click "Load unpacked" and select the `youtube-simple` folder
4. Refresh any open YouTube pages after installation

## Features

### Grab Video Titles
- Extracts video titles and channel names from YouTube pages
- Works on playlists, search results, and channel pages
- Exports data as JSON with metadata

### Scroll to Bottom
- Automatically scrolls to the bottom of the page
- Useful for loading more videos on infinite scroll pages

### Remove Top Video
- Removes the first video from YouTube liked videos list
- Clicks the action menu and selects "Remove from Liked videos"
- Requires being on the liked videos page (`youtube.com/playlist?list=LL`)

### Scrape Video Info
- Extracts comprehensive metadata from individual video pages
- Must be on a video page (`youtube.com/watch?v=...`)
- Captures detailed information from the `ytd-watch-metadata` element

## Button Functions

### Grab Video Titles
- **Purpose**: Extract video titles from list pages
- **Usage**: Navigate to any YouTube page with video lists, click button
- **Output**: JSON export with video titles, channels, and IDs

### Scroll to Bottom
- **Purpose**: Load more content on infinite scroll pages
- **Usage**: Click to scroll to bottom and trigger more video loading
- **Note**: May need multiple clicks on long pages

### Remove Top Video
- **Purpose**: Remove the first video from liked videos list
- **Usage**: Go to liked videos page, click to remove top video
- **Requirements**: Must be on `youtube.com/playlist?list=LL`

### Scrape Video Info
- **Purpose**: Extract detailed metadata from current video
- **Usage**: Navigate to any video page, click to scrape information
- **Requirements**: Must be on `youtube.com/watch?v=...`
- **Data Extracted**:
  - Video title, channel, subscriber count
  - View count, upload date, precise timestamps
  - Like/dislike counts with accessibility labels
  - Video description (multiple extraction methods)
  - Video duration, comment count
  - Scrape timestamp and click timestamp

## Important Notes

- **Page Refresh Required**: After installing or updating the extension, refresh YouTube pages for the extension to work properly
- **Page Requirements**: Different buttons work on different types of YouTube pages
- **Rate Limiting**: No built-in rate limiting - use responsibly to avoid detection
- **Data Storage**: All data is processed locally, no external transmission

## File Structure

```
youtube-simple/
├── manifest.json    # Extension configuration
├── content.js       # Main content script with all functionality
├── popup.html       # Extension popup interface
├── popup.js         # Popup logic and button handlers
└── example-element.html  # Sample YouTube metadata structure
```

## Technical Details

### Content Script Functions
- `getVideoTitles()`: Extracts video data from list pages
- `removeTopVideo()`: Handles video removal from liked list
- `scrapeVideoInfo()`: Extracts comprehensive video metadata
- `extractVideoData()`: Helper for parsing individual video elements

### Data Extraction
- Uses YouTube's DOM structure and element selectors
- Focuses on `ytd-watch-metadata` for video page scraping
- Multiple fallback selectors for reliability
- Tracks data source for debugging

### Popup Interface
- Simple button-based interface
- Real-time status updates
- Click-to-copy JSON output
- Export functionality for grabbed videos

## Troubleshooting

- **Buttons not working**: Refresh the YouTube page after extension installation
- **No videos found**: Ensure you're on the correct type of YouTube page
- **Scrape fails**: Make sure you're on an individual video page
- **Remove fails**: Verify you're on the liked videos playlist page
- **Extension not loading**: Check for errors at `chrome://extensions/`

## Limitations

- Manual navigation to appropriate YouTube pages required
- Subject to YouTube's DOM structure changes
- No automatic retry or error recovery
- Limited to publicly accessible video information
- Chrome extension storage quotas apply

## Privacy

- All processing happens locally in the browser
- No external servers or data transmission
- User controls all data and exports
- Respects YouTube's client-side data only

**License**: Educational use only. Respect YouTube's Terms of Service.
