# Google Drive Export Project Log

## Project Overview
**Goal**: Export files from organization-managed Google Drive account without service account access

## Session History

### Session 1 - June 15, 2025

#### Environment Setup
- **Issue**: Python venv creation and activation problems
- **Problem**: Python executable not found in activated venv, PATH conflicts with pyenv
- **Resolution**: Updated ~/.zshrc PATH to prioritize Homebrew Python over Framework Python
- **Status**: Python 3.11.9 working in virtual environment

#### Initial Approach: Service Account
- **Attempted**: Creating service account JSON for Google Drive API
- **Blocker**: Cannot create Google Cloud project due to organization restrictions
- **File Created**: `test.py` (service account approach - abandoned)

#### Second Approach: OAuth 2.0
- **Attempted**: OAuth flow for personal authentication
- **Blocker**: Still requires Google Cloud project creation for OAuth credentials
- **Files Created**: 
  - `OAuth_test.py` (OAuth approach - requires project)
  - `instructions.md` (OAuth setup guide)

#### Third Approach: Web Automation
- **Decision**: Use browser automation to simulate manual downloads
- **Options Evaluated**:
  1. **Selenium** - Most common, easily detectable
  2. **Playwright** - Modern, less detectable, better performance
  3. **Mouse clickers** - Fragile, most detectable
  4. **Google Takeout** - Manual but reliable alternative

- **Final Choice**: Playwright for learning opportunity and better stealth

#### Files Created
1. `test.py` - Service account approach (abandoned)
2. `OAuth_test.py` - OAuth approach (requires GCP project)
3. `instructions.md` - OAuth setup instructions
4. `selenium_drive.py` - Selenium automation script
5. `playwright_drive.py` - **Selected solution** - Playwright automation

## Technical Decisions

### Why Playwright Over Selenium?
- **Stealth**: Better at avoiding bot detection
- **Performance**: Faster execution, built-in waiting
- **Modern**: More reliable selectors and async support
- **Learning**: New technology for user to explore

### Key Features Implemented
- **Authentication**: Manual login step to avoid bot detection
- **Randomization**: Random delays between actions
- **Stealth Mode**: Removed automation markers from browser
- **Download Handling**: Automatic download monitoring
- **Folder Navigation**: Support for specific folders or full drive
- **Error Handling**: Graceful fallbacks for different UI elements

## Major Script Updates

### Update 1: Recursive Download with Rate Limiting
**Date**: June 15, 2025 (continued)
**Changes**: Complete rewrite of download logic

#### New Features Added:
1. **Manual Folder Navigation**: User navigates to target folder before script starts
2. **Recursive Directory Traversal**: Automatically processes subfolders
3. **Individual File Downloads**: Downloads files one-by-one instead of bulk selection
4. **Progressive Rate Limiting**: Smart delays that increase with download count
5. **Folder Structure Preservation**: Maintains hierarchy without ZIP files

#### Rate Limiting Strategy:
- **Base delay**: 2 seconds between downloads
- **Progressive backoff**: +0.5s every 10 downloads  
- **Random jitter**: Adds unpredictability to timing
- **Download counter**: Tracks progress and adjusts behavior

#### Technical Implementation:
- `get_current_folder_items()`: Identifies files vs folders in current directory
- `download_individual_file()`: Downloads single files with rate limiting
- `download_folder_recursively()`: Traverses directory tree depth-first
- `apply_rate_limit()`: Implements progressive delay strategy

#### User Experience Improvements:
- **Interactive start**: User confirms when ready to begin
- **Progress tracking**: Shows files/folders being processed
- **Clear navigation**: Logs folder entry/exit for visibility
- **Error handling**: Graceful fallbacks for UI element detection

### Update 2: Firefox Implementation with Complete Working Solution
**Date**: June 15, 2025 (continued)
**Changes**: Switched from Chrome to Firefox and completed full implementation

#### Major Changes:
1. **Browser Switch**: Moved from Chrome to Firefox due to persistent context issues
2. **Working Login Flow**: Successfully implemented manual login workflow
3. **File Detection Fixed**: Resolved issues with detecting actual files vs UI elements
4. **Download Implementation**: Complete working download functionality
5. **Folder Structure Preservation**: Downloads maintain exact folder hierarchy
6. **Continuous Operation Mode**: Script continues running for multiple folder downloads

#### Firefox-Specific Implementation:
- `firefox_drive.py`: Complete working Firefox-based solution
- Profile detection and selection for existing Firefox profiles
- Proper stealth injection with Firefox-specific user agents
- Fixed navigation issues (Alt+ArrowLeft for Firefox back button)
- Element refresh mechanism to handle DOM detachment

#### Download Features Completed:
- **Folder Structure Preservation**: 
  - `current_local_path` tracks hierarchy
  - Creates subdirectories automatically
  - Preserves exact folder structure from Drive
- **Continuous Operation**: 
  - Script runs in loop allowing multiple folder downloads
  - User navigates to new folders between sessions
  - Only exits when explicitly requested
- **Robust File Detection**:
  - Filters out UI elements (Settings, Support, etc.)
  - Uses data-id and aria-label for accurate identification
  - Comprehensive debug output for troubleshooting

#### User Workflow:
1. Script opens Firefox with existing profile
2. User logs in to Google Drive manually  
3. User navigates to desired folder
4. Press Enter to start download
5. Script recursively downloads preserving structure
6. User can navigate to next folder and repeat
7. Type 'quit' to exit

## Current Status
- ✅ Playwright script created with stealth features
- ✅ Download monitoring implemented
- ✅ Recursive folder traversal implemented
- ✅ Individual file download with rate limiting
- ✅ Manual folder navigation workflow
- ✅ **Firefox-based solution working completely**
- ✅ **Folder structure preservation implemented**
- ✅ **Continuous operation mode implemented**
- ✅ **Chrome profile selection for Chrome version**
- ✅ **Complete working solution ready for production use**

## Next Steps
1. ✅ Install Playwright: `pip install playwright`
2. ✅ Install browser: `playwright install firefox`
3. ✅ Test script with Google Drive folder
4. ✅ Refine UI selectors based on actual Drive interface
5. 🔄 Add file type filtering and exclusion options (future enhancement)
6. 🔄 Implement resume capability for interrupted downloads (future enhancement)
7. 🔄 Optimize download speed (user noted it's slow)
8. 🔄 Add Google Docs HTML export preference (currently downloads as DOCX)

## Technical Notes
- **User Agent**: Using realistic Chrome user agent
- **Viewport**: Standard desktop resolution (1920x1080)
- **Automation Markers**: Removed webdriver detection properties
- **Download Directory**: Configurable, defaults to `./playwright_downloads`
- **Rate Limiting**: Progressive delays from 2s to 5s+ based on download count
- **File Detection**: Uses `data-tooltip` attributes to identify Drive items
- **Folder Detection**: Looks for folder icons to distinguish from files

## Lessons Learned
- Organization-managed Google accounts have API restrictions
- Browser automation is viable alternative when API access is blocked
- Playwright offers better stealth capabilities than Selenium
- Manual authentication step reduces bot detection risk
- Individual file downloads are less suspicious than bulk operations
- Progressive rate limiting helps avoid Google's bot detection
- UI selectors may need adjustment based on Drive interface variations
- **Firefox more reliable than Chrome for persistent contexts**
- **Element detachment common in SPA - need refresh mechanisms**
- **Folder structure preservation requires careful path tracking**
- **Right-click context menus more reliable than keyboard shortcuts for downloads**
- **Continuous operation mode greatly improves user experience**