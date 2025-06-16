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

## Current Status
- ✅ Playwright script created with stealth features
- ✅ Download monitoring implemented
- ✅ Folder-specific and full-drive download options
- 🔄 Ready for testing and refinement

## Next Steps
1. Install Playwright: `pip install playwright`
2. Install browser: `playwright install chromium`
3. Test script with Google Drive
4. Refine selectors based on actual Drive UI
5. Add progress tracking and resume capabilities

## Technical Notes
- **User Agent**: Using realistic Chrome user agent
- **Viewport**: Standard desktop resolution (1920x1080)
- **Automation Markers**: Removed webdriver detection properties
- **Download Directory**: Configurable, defaults to `./playwright_downloads`

## Lessons Learned
- Organization-managed Google accounts have API restrictions
- Browser automation is viable alternative when API access is blocked
- Playwright offers better stealth capabilities than Selenium
- Manual authentication step reduces bot detection risk