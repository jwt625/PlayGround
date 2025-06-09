# Chrome Extension for Keystroke Tracker

## Installation

### 1. Load Extension in Chrome
1. Open Chrome and go to `chrome://extensions/`
2. Enable "Developer mode" (top right toggle)
3. Click "Load unpacked"
4. Select this `chrome-extension` folder
5. Note the extension ID (needed for native messaging)

### 2. Set up Native Messaging (Optional)
For direct file system access, set up native messaging:

1. Update `native-host-manifest.json` with your actual extension ID
2. Copy to Chrome's native messaging directory:
   ```bash
   # macOS
   mkdir -p ~/Library/Application\ Support/Google/Chrome/NativeMessagingHosts/
   cp native-host-manifest.json ~/Library/Application\ Support/Google/Chrome/NativeMessagingHosts/com.keystroketracker.chromehelper.json
   ```
3. Make sure the Go native host binary exists (will be created later)

### 3. Fallback Mode
If native messaging fails, the extension falls back to Chrome storage API. The Go server can poll this storage.

## Features

- **Tab Switch Tracking**: Detects switching between different domains
- **URL Change Monitoring**: Tracks SPA navigation within tabs
- **Domain Extraction**: Clean domain names for metrics
- **Fallback Storage**: Works even without native messaging
- **Privacy Focused**: Only tracks domains, not full URLs

## Data Flow

```
Tab Switch → Background Script → Native Host → Go Server → Prometheus
                ↓ (fallback)
             Chrome Storage → Go Polling → Prometheus
```

## Testing

1. Install extension
2. Switch between tabs with different domains
3. Check browser console for logs
4. Monitor Chrome storage at `chrome://extensions/` → Extension details → Storage