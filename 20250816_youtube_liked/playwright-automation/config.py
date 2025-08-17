"""
Configuration settings for YouTube automation (removal and metadata scraping).
"""

# YouTube URLs
YOUTUBE_LIKED_URL = "https://www.youtube.com/playlist?list=LL"

# Removal settings
DEFAULT_REMOVAL_COUNT = 4000
HEADLESS_MODE = False  # Set True for production
WAIT_BETWEEN_REMOVALS = 500  # milliseconds
MAX_RETRIES = 1000

# Selectors based on DOM analysis from RFD-002
VIDEO_SELECTOR = "ytd-playlist-video-renderer"
ACTION_MENU_SELECTOR = 'button[aria-label="Action menu"]'
REMOVE_OPTION_SELECTOR = 'ytd-popup-container tp-yt-paper-item:has-text("Remove from Liked videos")'
TITLE_SELECTOR = "a#video-title"

# Metadata scraping settings (RFD-003)
DEFAULT_SCRAPE_COUNT = 10000
DEFAULT_AVERAGE_PAUSE = 2.5  # seconds
METADATA_TIMEOUT = 10000  # 10 seconds

# Metadata selectors (from RFD-003 proven implementation)
METADATA_CONTAINER = "ytd-watch-metadata"
VIDEO_TITLE_SELECTOR = "ytd-watch-metadata h1 yt-formatted-string"
CHANNEL_NAME_SELECTOR = "ytd-watch-metadata ytd-channel-name a"
SUBSCRIBER_COUNT_SELECTOR = "ytd-watch-metadata #owner-sub-count"
VIEW_DATE_INFO_SELECTOR = "ytd-watch-info-text #info yt-formatted-string, ytd-watch-info-text #info"
PRECISE_DATE_SELECTOR = "ytd-watch-info-text tp-yt-paper-tooltip #tooltip"
LIKE_BUTTON_SELECTOR = "like-button-view-model button, segmented-like-dislike-button-view-model like-button-view-model button"
DISLIKE_BUTTON_SELECTOR = "dislike-button-view-model button, segmented-like-dislike-button-view-model dislike-button-view-model button"
BUTTON_TEXT_SELECTOR = ".yt-spec-button-shape-next__button-text-content"
DURATION_SELECTOR = ".ytp-time-duration"
COMMENT_COUNT_SELECTOR = "ytd-comments-header-renderer #count yt-formatted-string"

# Description selectors (4-tier fallback strategy)
DESCRIPTION_SELECTORS = [
    "ytd-text-inline-expander #expanded yt-attributed-string",  # Method 1: Expanded
    "ytd-text-inline-expander #attributed-snippet-text",       # Method 2: Snippet
    "#description-text-container #attributed-description-text", # Method 3: Container
    "#description yt-attributed-string, ytd-text-inline-expander yt-attributed-string"  # Method 4: Fallback
]

# Timeouts
PAGE_LOAD_TIMEOUT = 5000  # 5 seconds
ELEMENT_WAIT_TIMEOUT = 500  # 0.5 seconds
REMOVAL_WAIT_TIMEOUT = 500  # 0.5 second

# Logging
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
