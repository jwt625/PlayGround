# Google Drive Export Tool

A Playwright-based automation tool to recursively download files from Google Drive, especially useful when API access is restricted (e.g., organization-managed accounts).

## Features

- 🔐 **No API Keys Required** - Uses browser automation with manual login
- 📁 **Recursive Download** - Automatically traverses subfolders
- ⚡ **Individual File Downloads** - Downloads files one-by-one to preserve folder structure
- 🛡️ **Rate Limiting** - Progressive delays to avoid detection
- 🎯 **Manual Navigation** - You choose which folder to download
- 📊 **Progress Tracking** - Shows download progress and statistics

## Installation

### Prerequisites
- Python 3.7 or higher
- Chrome/Chromium browser

### Step 1: Create Virtual Environment
```bash
# Create virtual environment
python3 -m venv .venv

# Activate virtual environment
# On macOS/Linux:
source .venv/bin/activate
# On Windows:
# .venv\Scripts\activate
```

### Step 2: Install Dependencies
```bash
# Install dependencies
pip install -r requirements.txt

# Install browser (required for Playwright)
playwright install chromium
```

## Quick Start

### 1. Configure Profiles (Optional)
The script will auto-detect your Chrome profiles. To customize:
```bash
# Edit .env file to update email addresses or default profile
cp .env.example .env  # if needed
```

### 2. Run the Script
```bash
python playwright_drive.py
```

### 2. Follow the Interactive Process
1. **Browser Opens** - A Chrome window will open automatically
2. **Login** - Sign in to your Google Drive account
3. **Navigate** - Browse to the folder you want to download
4. **Confirm** - Press Enter in the terminal when ready
5. **Wait** - The script will recursively download all files

### 3. Monitor Progress
The script will show:
- Current folder being processed
- Files being downloaded
- Rate limiting delays
- Total download count

### 4. Find Your Files
Downloaded files will be saved to `./playwright_downloads/`

## How It Works

### Authentication
- Opens real Chrome browser
- You manually log in (no credentials stored)
- Works with organization-managed accounts
- Bypasses API restrictions

### Download Process
1. **Scan** - Identifies files and folders in current directory
2. **Download Files** - Downloads all files in current folder
3. **Enter Subfolders** - Recursively processes each subfolder
4. **Navigate Back** - Returns to parent folder when done
5. **Rate Limit** - Applies progressive delays between downloads

### File Formats
- **Google Docs** → HTML format
- **Google Sheets** → HTML format  
- **Google Slides** → HTML format
- **Other Files** → Original format
- **Folders** → Preserved as folder structure (not ZIP)

## Configuration

### Change Download Directory
```python
downloader = GoogleDrivePlaywright(download_dir="./my_custom_folder")
```

### Adjust Rate Limiting
```python
self.rate_limit_delay = 3  # Change base delay (default: 2 seconds)
```

### Run in Background (Headless)
```python
await downloader.setup_browser(headless=True)
```

## Rate Limiting Strategy

The script uses progressive rate limiting to avoid detection:

- **Base delay**: 2 seconds between downloads
- **Progressive increase**: +0.5 seconds every 10 downloads
- **Random jitter**: Adds unpredictability
- **Example timeline**:
  - Files 1-10: 2-3 second delays
  - Files 11-20: 2.5-3.5 second delays
  - Files 21-30: 3-4 second delays
  - And so on...

## Troubleshooting

### Common Issues

**"Browser not found"**
```bash
playwright install chromium
```

**"No items found in folder"**
- Check if you're in the correct folder
- UI selectors may need adjustment for your Drive interface
- Try refreshing the page before starting

**"Download failed"**
- Some files may have download restrictions
- Check if you have access to the file
- Large files may take longer to process

**"Rate limit exceeded"**
- Script automatically handles this with progressive delays
- If issues persist, increase base delay in the code

### UI Selector Issues

Google Drive's interface may change. If the script can't find files/folders, you may need to update these selectors in the code:

```python
# In get_current_folder_items()
file_elements = await self.page.query_selector_all('[data-tooltip]:not([data-tooltip=""])')
```

## Advanced Usage

### Custom File Filtering (Future Enhancement)
Currently downloads all files. Future versions will support:
- File type exclusions
- Name pattern filtering
- Size limits

### Resume Downloads (Future Enhancement)
Planned feature for resuming interrupted downloads.

## Security Notes

- ✅ No credentials stored locally
- ✅ Uses your existing browser session
- ✅ Read-only access to your Drive
- ✅ Progressive delays minimize detection risk
- ⚠️ Downloads are visible in your Google account activity

## Limitations

- Requires manual login and navigation
- UI-dependent (may break if Google changes interface)
- Not suitable for very large datasets (use Google Takeout instead)
- Some organization policies may block automation

## Alternative Solutions

If this tool doesn't work for your use case:

1. **Google Takeout** - Official export tool (manual)
2. **API Access** - Requires Google Cloud project
3. **Third-party tools** - Various commercial options available

## Contributing

Feel free to submit issues or improvements! Common areas for enhancement:
- UI selector robustness
- File filtering options
- Resume capabilities
- Error handling improvements

## License

This project is for educational and personal use. Respect Google's Terms of Service and your organization's policies.