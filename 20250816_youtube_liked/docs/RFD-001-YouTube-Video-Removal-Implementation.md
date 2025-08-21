# RFD-001: YouTube Video Removal Implementation

**Status**: In Development  
**Author**: AI Assistant  
**Date**: 2025-08-16  
**Related**: RFD-000 (YouTube Liked Videos Backup Extension)

## Summary

This RFD documents the implementation approach for programmatically removing videos from YouTube's liked videos list through DOM manipulation and UI automation.

## Background

The original YouTube Liked Videos Backup Extension (RFD-000) was overly complex and failed due to:
- Async message channel closure errors
- Complex scrolling logic with timeouts
- Over-engineered data validation and storage systems
- Multiple interdependent components causing cascading failures

### Minimal Extension Success

A simplified extension (`youtube-simple/`) was successfully implemented with core functionality:

#### ✅ **Working Features**
1. **Video Title Extraction**: Successfully scrapes video titles, channels, and IDs from current page
2. **Manual Scrolling**: Simple scroll-to-bottom functionality
3. **JSON Export**: Downloads video data as structured JSON files
4. **Multi-Layout Support**: Works across different YouTube layouts (playlist, grid, search)

#### 🔧 **Technical Foundation**
- **Manifest V3**: Clean, minimal permissions (`activeTab`, `scripting`)
- **Synchronous Operations**: No async message handling issues
- **Robust Selectors**: Multiple fallback selectors for different YouTube layouts
- **User Control**: Manual process allowing user oversight at each step

This working foundation provides the platform for adding video removal functionality.

## Problem Statement

YouTube's liked videos list has a 5000 video limit. To access older videos, newer ones must be removed. Manual removal is tedious for large collections. We need a programmatic way to remove videos while maintaining data integrity.

## Implementation Approaches Considered

### Approach 1: Direct DOM Manipulation ⭐ **CHOSEN**
- **Method**: Find and click YouTube's existing "Remove" buttons
- **Pros**: Uses YouTube's intended UI flow, no API reverse engineering needed
- **Cons**: Fragile to UI changes, requires finding correct selectors
- **Implementation**: Simulate user clicks on remove buttons

### Approach 2: YouTube Internal API Calls
- **Method**: Intercept/mimic network requests when unliking videos
- **Pros**: More reliable than DOM clicking
- **Cons**: Complex, requires reverse engineering, may violate ToS
- **Status**: Rejected for complexity

### Approach 3: Simulate User Actions
- **Method**: Multi-step UI automation (menu → remove option)
- **Pros**: Follows exact user workflow
- **Cons**: Multiple steps, timing dependent, complex state management
- **Status**: Fallback approach

## Technical Implementation

### Minimal Extension Architecture

```
youtube-simple/
├── manifest.json          # Minimal Manifest V3 config
├── popup.html             # 4 buttons: Grab, Scroll, Remove, Export
├── popup.js               # Button handlers and messaging
└── content.js             # DOM manipulation and video removal
```

#### Proven Working Components

**Video Extraction (`getVideoTitles()`)**:
```javascript
// Multi-selector approach for different YouTube layouts
const selectors = [
  'ytd-playlist-video-renderer',  // Playlist view ✅
  'ytd-grid-video-renderer',      // Grid view ✅
  'ytd-video-renderer',           // Search results ✅
  'ytd-compact-video-renderer'    // Sidebar ✅
];

// Robust data extraction with fallbacks
const titleSelectors = [
  'a#video-title',                // Primary ✅
  'h3 a',                        // Fallback ✅
  'a[href*="/watch?v="]'         // Generic ✅
];
```

**Export Functionality**:
- ✅ Structured JSON with metadata (export date, total count, source URL)
- ✅ Automatic filename generation (`youtube-videos-2025-01-16.json`)
- ✅ Browser download API integration
- ✅ Error handling and user feedback

**User Interface**:
- ✅ Clean popup with status messages
- ✅ Disabled states and loading indicators
- ✅ Color-coded buttons (red for destructive actions)
- ✅ Real-time video count display

### Video Removal Strategy

#### Primary Method: Direct Remove Button
```javascript
const removeSelectors = [
  'button[aria-label*="Remove from"]',
  'button[aria-label*="remove"]', 
  'button[aria-label*="Remove"]'
];
```

#### Fallback Method: Menu Navigation
```javascript
// 1. Find menu button (three dots)
const menuSelectors = [
  'button[aria-label*="Action menu"]',
  'button[aria-label*="More actions"]',
  'button[aria-haspopup="true"]'
];

// 2. Click menu to open options
menuButton.click();

// 3. Find remove option in opened menu
const menuRemoveSelectors = [
  'tp-yt-paper-listbox button[aria-label*="Remove"]',
  '[role="menuitem"][aria-label*="Remove"]'
];
```

### Current Challenges

#### 1. YouTube's Complex DOM Structure
- Content wrapped in `tp-yt-app-drawer` elements
- Dynamic loading and shadow DOM usage
- Selectors may not penetrate all container levels

#### 2. Hidden UI Elements
- Remove buttons may be hidden until hover
- Menu buttons may not be immediately visible
- Requires interaction triggers to reveal options

#### 3. UI Pattern Variations
- Different layouts (grid vs list view)
- Mobile vs desktop patterns
- A/B testing variations

## Debugging Implementation

### Page Structure Analysis
```javascript
// Debug container hierarchy
const mainContainers = [
  'ytd-app', 'tp-yt-app-drawer', '#content', '#primary'
];

// Count video elements in each container
const videoSelectors = [
  'ytd-playlist-video-renderer',
  'ytd-grid-video-renderer', 
  '[data-video-id]'
];
```

### Button Discovery
```javascript
// Log all buttons in first video element
const allButtons = firstVideo.querySelectorAll('button');
allButtons.forEach(btn => {
  console.log({
    'aria-label': btn.getAttribute('aria-label'),
    'title': btn.getAttribute('title'),
    'class': btn.className,
    'text': btn.textContent?.trim()
  });
});
```

## Alternative Solutions

### Hover-Triggered UI Revelation
```javascript
// Trigger hover to reveal hidden buttons
firstVideo.dispatchEvent(new MouseEvent('mouseenter', {bubbles: true}));
setTimeout(() => {
  // Look for newly revealed buttons
}, 500);
```

### SVG Icon Detection
```javascript
// Find three dots by SVG path or icon classes
'button svg[d*="M12"]',  // Common three dots path
'button .yt-icon-shape'   // YouTube icon wrapper
```

### Visual Pattern Matching
```javascript
// Find buttons containing three dots characters
const text = btn.textContent || btn.innerHTML;
if (text.includes('⋮') || text.includes('⋯') || text.includes('•••')) {
  // Potential menu button
}
```

## Next Steps

1. **Run Debug Implementation**: Execute comprehensive page structure analysis
2. **Analyze Results**: Identify actual selectors and button patterns
3. **Refine Selectors**: Update removal logic based on findings
4. **Test Edge Cases**: Handle different layouts and UI states
5. **Error Handling**: Implement robust fallbacks and user feedback

## Success Criteria

### ✅ **Already Achieved (Minimal Extension)**
- [x] Successfully extract video metadata from YouTube pages
- [x] Handle multiple YouTube layouts (playlist, grid, search)
- [x] Export structured JSON data with metadata
- [x] Provide user feedback and error handling
- [x] Simple scroll functionality for loading more content

### 🎯 **Video Removal Goals**
- [ ] Successfully identify first video element
- [ ] Find and click remove button or menu
- [ ] Confirm video removal from liked list
- [ ] Handle errors gracefully with user feedback
- [ ] Work across different YouTube layouts

## Risk Assessment

**High Risk**: YouTube UI changes breaking selectors  
**Medium Risk**: Rate limiting or spam detection  
**Low Risk**: User accidentally removing wrong videos

## Future Considerations

- Batch removal capabilities
- Undo functionality
- Rate limiting to avoid detection
- Integration with backup verification system
- Support for different YouTube layouts (mobile, TV, etc.)

---

**Note**: This implementation relies on YouTube's current DOM structure and may require updates as YouTube evolves their interface.
