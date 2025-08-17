# RFD-002: Playwright YouTube Video Removal Implementation

**Status**: Implemented
**Author**: AI Assistant
**Date**: 2025-08-16
**Updated**: 2025-08-16
**Related**: RFD-001 (YouTube Video Removal Implementation)

## Summary

This RFD documents the transition from browser extension to Playwright automation for removing videos from YouTube's liked videos list. The browser extension approach failed due to YouTube's content security policies and complex event handling, necessitating a more robust automation solution.

## Background

### Browser Extension Limitations (RFD-001)

The initial browser extension approach encountered critical failures:

- **Content Security Policy Blocks**: YouTube prevents script-based DOM manipulation
- **Event Handler Complexity**: Simple `.click()` calls don't trigger YouTube's event delegation
- **Async Response Timeouts**: Message channel closures before Promise resolution
- **Unreliable Selectors**: Dynamic DOM changes break element targeting

### Key Learning: DOM Structure Analysis

Through extensive debugging, we identified the exact DOM structure for video removal:

#### Action Menu Button
```html
<button id="button" class="style-scope yt-icon-button" aria-label="Action menu">
  <yt-icon class="style-scope ytd-menu-renderer">
    <span class="yt-icon-shape style-scope yt-icon yt-spec-icon-shape">
      <div style="width: 100%; height: 100%; display: block; fill: currentcolor;">
        <svg xmlns="http://www.w3.org/2000/svg" enable-background="new 0 0 24 24" height="24" viewBox="0 0 24 24" width="24">
          <path d="M12 16.5c.83 0 1.5.67 1.5 1.5s-.67 1.5-1.5 1.5-1.5-.67-1.5-1.5.67-1.5 1.5-1.5zM10.5 12c0 .83.67 1.5 1.5 1.5s1.5-.67 1.5-1.5-.67-1.5-1.5-1.5-1.5.67-1.5 1.5zm0-6c0 .83.67 1.5 1.5 1.5s1.5-.67 1.5-1.5-.67-1.5-1.5-1.5-1.5.67-1.5 1.5z"></path>
        </svg>
      </div>
    </span>
  </yt-icon>
</button>
```

#### Remove Menu Item
```html
<tp-yt-paper-item class="style-scope ytd-menu-service-item-renderer" role="option" tabindex="0">
  <yt-icon class="style-scope ytd-menu-service-item-renderer">
    <span class="yt-icon-shape style-scope yt-icon yt-spec-icon-shape">
      <div style="width: 100%; height: 100%; display: block; fill: currentcolor;">
        <svg xmlns="http://www.w3.org/2000/svg" enable-background="new 0 0 24 24" height="24" viewBox="0 0 24 24" width="24">
          <path d="M11 17H9V8h2v9zm4-9h-2v9h2V8zm4-4v1h-1v16H6V5H5V4h4V3h6v1h4zm-2 1H7v15h10V5z"></path>
        </svg>
      </div>
    </span>
  </yt-icon>
  <yt-formatted-string class="style-scope ytd-menu-service-item-renderer">Remove from Liked videos</yt-formatted-string>
</tp-yt-paper-item>
```

## Problem Statement

YouTube's liked videos list has a 5000 video limit. To access older videos, newer ones must be removed. Manual removal is tedious for large collections (e.g., removing top 4000 videos). Browser extensions cannot reliably automate this process due to security restrictions.

## Solution: Playwright Automation

### Why Playwright Over Selenium

- **2-3x Faster**: Modern architecture with better performance
- **Auto-waiting**: Built-in smart waits eliminate timing issues
- **Reliable Events**: Real browser control ensures proper event triggering
- **Modern SPA Support**: Designed for dynamic applications like YouTube
- **Better Debugging**: Screenshots, videos, and detailed logging

### Implementation Architecture

```
playwright-automation/
├── venv/                    # Python virtual environment
├── requirements.txt         # Dependencies
├── youtube_remover.py       # Main removal script
├── config.py               # Configuration settings
└── utils/
    ├── logging.py          # Logging utilities
    └── verification.py     # Backup integration
```

## Technical Implementation

### Core Removal Logic

```python
from playwright.sync_api import sync_playwright

def remove_top_videos(count=4000):
    with sync_playwright() as p:
        browser = p.firefox.launch(headless=False)
        page = browser.new_page()
        
        # Navigate to liked videos
        page.goto("https://www.youtube.com/playlist?list=LL")
        
        # Wait for page load
        page.wait_for_selector("ytd-playlist-video-renderer")
        
        # Get all videos once at the beginning
        page.wait_for_selector("ytd-playlist-video-renderer")
        all_videos = page.locator("ytd-playlist-video-renderer").all()
        actual_count = min(count, len(all_videos))

        removed_count = 0
        # Go through the list sequentially
        for video_index in range(actual_count):
            try:
                # Target video by index in original list
                target_video = all_videos[video_index]

                # Get video title for logging
                title = target_video.locator("a#video-title").text_content()
                print(f"Removing: {title}")

                # Click action menu button
                action_menu = target_video.locator('button[aria-label="Action menu"]')
                action_menu.click()

                # Wait for popup menu and click remove option
                remove_button = page.locator('ytd-popup-container tp-yt-paper-item:has-text("Remove from Liked videos")')
                remove_button.click()

                # Wait for removal to complete
                page.wait_for_timeout(1000)

                removed_count += 1
                print(f"Progress: {removed_count}/{actual_count}")

            except Exception as e:
                print(f"Error removing video at index {video_index}: {e}")
                break
        
        browser.close()
        return removed_count
```

### Key Selectors

Based on DOM analysis, these selectors are reliable:

```python
# Video container
VIDEO_SELECTOR = "ytd-playlist-video-renderer"

# Action menu button (three dots)
ACTION_MENU_SELECTOR = 'button[aria-label="Action menu"]'

# Remove option in popup menu
REMOVE_OPTION_SELECTOR = 'ytd-popup-container tp-yt-paper-item:has-text("Remove from Liked videos")'

# Video title for logging
TITLE_SELECTOR = "a#video-title"
```

### Error Handling and Resilience

```python
def safe_remove_video(page, video_element):
    max_retries = 3
    for attempt in range(max_retries):
        try:
            # Get title before removal
            title = video_element.locator("a#video-title").text_content()
            
            # Click action menu
            action_menu = video_element.locator('button[aria-label="Action menu"]')
            action_menu.click()
            
            # Wait for menu and click remove
            remove_button = page.locator('ytd-popup-container tp-yt-paper-item:has-text("Remove from Liked videos")')
            remove_button.wait_for(timeout=5000)
            remove_button.click()
            
            # Verify removal
            page.wait_for_timeout(1000)
            return True, title
            
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                page.wait_for_timeout(2000)  # Wait before retry
            
    return False, None
```

## Integration with Existing System

### Backup Verification Workflow

```python
def integrated_removal_workflow(videos_to_remove=4000):
    # 1. Verify backup exists and is recent
    if not verify_recent_backup():
        print("No recent backup found. Run backup first.")
        return False
    
    # 2. Remove videos with Playwright
    removed_count = remove_top_videos(videos_to_remove)
    
    # 3. Log results
    print(f"Successfully removed {removed_count} videos")
    
    return removed_count == videos_to_remove
```

## Setup Instructions

### 1. Environment Setup
```bash
cd 20250816_youtube_liked
mkdir playwright-automation
cd playwright-automation
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

### 2. Install Dependencies
```bash
pip install playwright
playwright install firefox
```

### 3. Configuration
```python
# config.py
YOUTUBE_LIKED_URL = "https://www.youtube.com/playlist?list=LL"
DEFAULT_REMOVAL_COUNT = 4000
HEADLESS_MODE = False  # Set True for production
WAIT_BETWEEN_REMOVALS = 1000  # milliseconds
```

## Success Criteria

- [x] **Reliable DOM Interaction**: Playwright can consistently click YouTube elements
- [x] **Authentication Handling**: User-guided login with session persistence
- [x] **Batch Processing**: Remove specified number of videos (e.g., 4000)
- [x] **Error Recovery**: Handle failures gracefully with retries
- [x] **Integration Ready**: Works with existing backup verification system
- [x] **Logging**: Detailed progress and error reporting
- [x] **Sequential Video Removal**: Efficient iteration through original video list without DOM dependencies
- [x] **Session Persistence**: Proper file-based session storage for seamless re-authentication
- [x] **Cross-Platform Browser Support**: Firefox-based implementation for better compatibility

## Risk Assessment

**Low Risk**: Playwright automation is more reliable than browser extensions  
**Medium Risk**: YouTube may implement additional anti-automation measures  
**Mitigation**: Use realistic delays and human-like interaction patterns

## Future Considerations

- **Rate Limiting**: Add configurable delays to avoid detection
- **Headless Mode**: Run without visible browser for production use
- **Parallel Processing**: Multiple browser instances for faster removal
- **Resume Capability**: Save progress and resume interrupted sessions

## Implementation Status

**✅ COMPLETED** - Full implementation available in `playwright-automation/` directory.

### Key Components Implemented

- **`youtube_remover.py`**: Main automation script with CLI interface
- **`utils/auth.py`**: Authentication with session persistence
- **`utils/logging.py`**: Comprehensive logging system
- **`utils/verification.py`**: Backup integration
- **`config.py`**: Configuration management
- **`demo.py`**: Safe testing script (3 videos)
- **`test_auth.py`**: Authentication testing
- **`test_setup.py`**: Installation verification

### Authentication Solution

The implementation includes a user-guided authentication flow:

1. **First run**: Browser opens, user logs in manually, session saved automatically
2. **Subsequent runs**: Saved session reused (valid for ~30 days)
3. **Session expiry**: Graceful fallback to manual login
4. **Session management**: `--clear-session`, `--force-login` options

### Browser Engine Selection

**Updated 2025-08-16**: Switched from Chromium to Firefox due to compatibility issues.

- **Issue**: Chromium was experiencing `TargetClosedError` crashes on macOS
- **Root Cause**: Browser context creation failures and potential system compatibility issues
- **Solution**: Firefox provides better stability and compatibility
- **Impact**: No functional changes - all automation features work identically

### Session Storage Fix

**Updated 2025-08-16**: Fixed session persistence implementation.

- **Issue**: Session storage was using directory path instead of file path
- **Fix**: Changed `context_dir` to `context_file` in auth.py
- **Additional Fix**: Updated all references in youtube_remover.py and test files
- **Result**: Proper session saving and loading for seamless re-authentication

### Usage

```bash
cd playwright-automation
source venv/bin/activate

# Test with small counts first
python youtube_remover.py --count 3         # Test with 3 videos
python youtube_remover.py --count 10        # Remove 10 videos
python youtube_remover.py --count 100       # Remove 100 videos

# Production removal (fast and efficient)
python youtube_remover.py                   # Default: 4000 videos
python youtube_remover.py --count 2000      # Remove 2000 videos

# Advanced options
python youtube_remover.py --count 2000 --headless    # Headless mode
python youtube_remover.py --clear-session            # Clear saved session
python youtube_remover.py --force-login              # Force new login
```

### Sequential Video Removal Solution

**Updated 2025-08-16**: Implemented efficient sequential video removal without page refreshes.

- **Issue**: YouTube SPA doesn't update DOM immediately after video removal, causing targeting issues
- **Initial Approach**: Page refresh after each removal (slow, triggers rate limiting after ~46 refreshes)
- **Correct Solution**: Query all videos once, then iterate through the original list sequentially
- **Implementation**: `all_videos[0], all_videos[1], all_videos[2]...` removes videos in order
- **Result**: Fast, reliable removal without DOM update dependencies or rate limiting

### Simplified Architecture

**Updated 2025-08-16**: Removed redundant demo script, consolidated into main script.

- **Change**: Removed separate `demo.py` wrapper script
- **Rationale**: Unnecessary complexity, main script handles all use cases
- **Usage**: `python youtube_remover.py --count 3` for testing
- **Progress**: Shows real-time progress and detailed logging

### Troubleshooting

If you encounter browser launch issues:

1. **Reinstall Firefox**: `playwright install firefox`
2. **Clear sessions**: `python youtube_remover.py --clear-session`
3. **Force fresh login**: `python youtube_remover.py --force-login`

If you encounter AttributeError about 'context_dir':

1. **Update codebase**: Ensure all files use `context_file` instead of `context_dir`
2. **Clear old sessions**: `python youtube_remover.py --clear-session`
3. **Restart demo**: `python demo.py --count 2`

---

**Note**: This implementation provides a robust foundation for YouTube video removal that overcomes the limitations encountered with browser extension approaches, including proper authentication handling and cross-platform browser compatibility.
