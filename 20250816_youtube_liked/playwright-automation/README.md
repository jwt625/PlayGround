# YouTube Video Removal Automation

This Playwright-based automation tool removes videos from your YouTube liked videos list, implementing the solution described in RFD-002.

## Features

- **Reliable Automation**: Uses Playwright for robust browser automation
- **Smart Authentication**: User-guided login with session persistence
- **Session Management**: Login once, reuse for weeks (automatic session saving)
- **Backup Integration**: Verifies recent backup exists before removal
- **Error Handling**: Comprehensive retry logic and error recovery
- **Progress Tracking**: Detailed logging and progress reporting
- **Configurable**: Customizable removal count and behavior

## Setup

The environment is already set up with:
- Python virtual environment (`venv/`)
- Playwright installed with Chromium browser
- All dependencies configured

## Usage

### First Time Setup

1. **Test Authentication**:
```bash
source venv/bin/activate
python test_auth.py
```

2. **Run Demo** (removes 3 videos for testing):
```bash
python demo.py
```

### Basic Usage

Remove the default number of videos (4000) with backup verification:

```bash
source venv/bin/activate
python youtube_remover.py
```

**First run**: You'll be prompted to log in to YouTube manually. The session will be saved.
**Subsequent runs**: Uses saved session automatically (no login required).

### Custom Options

```bash
# Remove specific number of videos
python youtube_remover.py --count 2000

# Run in headless mode (no visible browser)
python youtube_remover.py --headless

# Force new login (ignore saved session)
python youtube_remover.py --force-login

# Clear saved session and exit
python youtube_remover.py --clear-session

# Skip backup verification (not recommended)
python youtube_remover.py --skip-backup-check
```

### Test Setup

Verify the installation is working:

```bash
source venv/bin/activate
python test_setup.py
```

## How It Works

1. **Authentication**:
   - First run: Prompts user to log in manually, saves session
   - Subsequent runs: Automatically uses saved session
   - Session expires after 30 days (configurable)

2. **Backup Verification**: Checks for recent backup in `../backup/` directory

3. **Browser Launch**: Opens Chromium browser and navigates to YouTube liked videos

4. **Video Removal**: For each video:
   - Locates the action menu button (three dots)
   - Clicks to open the menu
   - Selects "Remove from Liked videos"
   - Waits for removal to complete

5. **Progress Logging**: Tracks progress and logs detailed information

## Configuration

Edit `config.py` to customize:

- `DEFAULT_REMOVAL_COUNT`: Number of videos to remove (default: 4000)
- `HEADLESS_MODE`: Run without visible browser (default: False)
- `WAIT_BETWEEN_REMOVALS`: Delay between removals in milliseconds
- `MAX_RETRIES`: Number of retry attempts for failed removals

## Logging

Logs are saved to `logs/` directory with timestamps. Both console and file logging are enabled for comprehensive tracking.

## Safety Features

- **Backup Verification**: Ensures recent backup exists before proceeding
- **Retry Logic**: Handles temporary failures with automatic retries
- **Graceful Degradation**: Stops safely if too many consecutive failures occur
- **Detailed Logging**: Complete audit trail of all actions

## Troubleshooting

### Common Issues

1. **"No recent backup found"**: Run the backup script first
2. **Browser fails to start**: Ensure Chromium is properly installed with `playwright install chromium`
3. **Selectors not working**: YouTube may have changed their DOM structure

### Debug Mode

For debugging, run with visible browser:

```bash
python youtube_remover.py --count 5  # Test with small number first
```

## Architecture

```
playwright-automation/
├── venv/                    # Python virtual environment
├── requirements.txt         # Dependencies
├── youtube_remover.py       # Main removal script
├── config.py               # Configuration settings
├── test_setup.py           # Setup verification
├── logs/                   # Log files (created automatically)
└── utils/
    ├── logging.py          # Logging utilities
    └── verification.py     # Backup integration
```

## Related Documents

- **RFD-002**: Complete technical specification
- **RFD-001**: Previous browser extension approach (deprecated)

---

**⚠️ Important**: Always ensure you have a recent backup before running video removal. This tool permanently removes videos from your YouTube liked list.
