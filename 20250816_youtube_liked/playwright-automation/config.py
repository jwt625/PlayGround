"""
Configuration settings for YouTube video removal automation.
"""

# YouTube URLs
YOUTUBE_LIKED_URL = "https://www.youtube.com/playlist?list=LL"

# Removal settings
DEFAULT_REMOVAL_COUNT = 4000
HEADLESS_MODE = False  # Set True for production
WAIT_BETWEEN_REMOVALS = 1000  # milliseconds
MAX_RETRIES = 3

# Selectors based on DOM analysis from RFD-002
VIDEO_SELECTOR = "ytd-playlist-video-renderer"
ACTION_MENU_SELECTOR = 'button[aria-label="Action menu"]'
REMOVE_OPTION_SELECTOR = 'ytd-popup-container tp-yt-paper-item:has-text("Remove from Liked videos")'
TITLE_SELECTOR = "a#video-title"

# Timeouts
PAGE_LOAD_TIMEOUT = 30000  # 30 seconds
ELEMENT_WAIT_TIMEOUT = 5000  # 5 seconds
REMOVAL_WAIT_TIMEOUT = 1000  # 1 second

# Logging
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
