/**
 * Constants for YouTube Liked Videos Backup Extension
 */

// YouTube URL patterns
const YOUTUBE_PATTERNS = {
  LIKED_VIDEOS_URL: '/playlist?list=LL',
  VIDEO_URL_PATTERN: /^https:\/\/www\.youtube\.com\/watch\?v=([a-zA-Z0-9_-]{11})/,
  CHANNEL_URL_PATTERN: /^https:\/\/www\.youtube\.com\/(channel\/|c\/|user\/|@)/,
  PLAYLIST_URL_PATTERN: /^https:\/\/www\.youtube\.com\/playlist\?list=([a-zA-Z0-9_-]+)/
};

// DOM selectors for YouTube elements
const YOUTUBE_SELECTORS = {
  // Video containers
  VIDEO_CONTAINER: 'ytd-playlist-video-renderer',
  VIDEO_CONTAINER_ALT: 'ytd-grid-video-renderer',
  
  // Video metadata
  VIDEO_TITLE: 'a#video-title',
  VIDEO_LINK: 'a#video-title',
  VIDEO_DURATION: '.ytd-thumbnail-overlay-time-status-renderer',
  VIDEO_THUMBNAIL: 'img',
  
  // Channel information
  CHANNEL_NAME: 'ytd-channel-name a',
  CHANNEL_LINK: 'ytd-channel-name a',
  
  // Video stats
  VIEW_COUNT: '#metadata-line span:first-child',
  UPLOAD_DATE: '#metadata-line span:last-child',
  
  // Playlist controls
  UNLIKE_BUTTON: 'button[aria-label*="Remove from"]',
  MENU_BUTTON: 'button[aria-label="Action menu"]',
  
  // Pagination
  LOAD_MORE_BUTTON: '#continuations button',
  SPINNER: 'tp-yt-paper-spinner',
  
  // Page indicators
  PLAYLIST_HEADER: 'ytd-playlist-header-renderer',
  VIDEO_COUNT: '#stats .style-scope.ytd-playlist-sidebar-primary-info-renderer'
};

// Storage keys
const STORAGE_KEYS = {
  VIDEOS: 'videos',
  SESSIONS: 'sessions', 
  SETTINGS: 'settings',
  STATE: 'state',
  CURRENT_SESSION: 'currentSession'
};

// Extension settings defaults
const DEFAULT_SETTINGS = {
  autoRemoveAfterBackup: false,
  verificationThreshold: 80,
  removalRateLimit: 20, // per minute
  exportFormat: 'json',
  includeDescriptions: true,
  downloadThumbnails: false,
  maxRetries: 3,
  batchSize: 50,
  scrollDelay: 2000, // ms between scroll actions
  navigationDelay: 5000 // ms between background tab navigations
};

// Rate limiting and timing
const TIMING = {
  SCROLL_DELAY: 2000,           // Delay between scroll actions
  SCRAPE_DELAY: 500,            // Delay between scraping individual videos
  REMOVAL_DELAY: 3000,          // Delay between video removals
  NAVIGATION_DELAY: 5000,       // Delay between background tab navigations
  RETRY_DELAY: 1000,            // Base delay for retries
  MAX_RETRY_DELAY: 10000,       // Maximum retry delay
  VERIFICATION_TIMEOUT: 30000,   // Timeout for verification operations
  PAGE_LOAD_TIMEOUT: 15000      // Timeout for page loads
};

// Batch processing limits
const LIMITS = {
  BATCH_SIZE: 50,               // Videos processed per batch
  MAX_BACKGROUND_TABS: 3,       // Maximum concurrent background tabs
  CHROME_STORAGE_QUOTA: 5242880, // ~5MB in bytes
  MAX_VIDEOS_CHROME_STORAGE: 10000, // Estimated max videos in Chrome storage
  EXPORT_CHUNK_SIZE: 1000,      // Videos per export chunk
  MAX_RETRIES: 3,               // Maximum retry attempts
  MIN_VERIFICATION_SCORE: 60    // Minimum score to consider verified
};

// Message types for communication between scripts
const MESSAGE_TYPES = {
  // Content script to background
  START_BACKUP: 'startBackup',
  PAUSE_BACKUP: 'pauseBackup',
  STOP_BACKUP: 'stopBackup',
  VIDEO_SCRAPED: 'videoScraped',
  BATCH_COMPLETE: 'batchComplete',
  BACKUP_COMPLETE: 'backupComplete',
  ERROR_OCCURRED: 'errorOccurred',
  
  // Background to content script
  BACKUP_STATUS: 'backupStatus',
  SETTINGS_UPDATED: 'settingsUpdated',
  START_REMOVAL: 'startRemoval',
  
  // Popup communication
  GET_STATUS: 'getStatus',
  GET_STATS: 'getStats',
  EXPORT_DATA: 'exportData',
  
  // Options page
  SAVE_SETTINGS: 'saveSettings',
  LOAD_SETTINGS: 'loadSettings'
};

// Error types
const ERROR_TYPES = {
  SCRAPING_ERROR: 'scrapingError',
  STORAGE_ERROR: 'storageError',
  NETWORK_ERROR: 'networkError',
  VALIDATION_ERROR: 'validationError',
  REMOVAL_ERROR: 'removalError',
  QUOTA_EXCEEDED: 'quotaExceeded',
  PERMISSION_DENIED: 'permissionDenied',
  TIMEOUT_ERROR: 'timeoutError'
};

// Metadata completeness levels
const METADATA_LEVELS = {
  BASIC: ['videoId', 'title', 'channelName', 'duration'],
  STANDARD: ['videoId', 'title', 'channelName', 'duration', 'uploadDate', 'viewCount', 'thumbnails'],
  COMPLETE: ['videoId', 'title', 'channelName', 'duration', 'uploadDate', 'viewCount', 'thumbnails', 'description', 'tags', 'category']
};

// Export formats
const EXPORT_FORMATS = {
  JSON: 'json',
  CSV: 'csv',
  BOTH: 'both'
};

// Extension states
const EXTENSION_STATES = {
  IDLE: 'idle',
  SCRAPING: 'scraping',
  VERIFYING: 'verifying',
  REMOVING: 'removing',
  EXPORTING: 'exporting',
  ERROR: 'error'
};

// Utility functions for constants
const UTILS = {
  /**
   * Check if current page is YouTube liked videos
   */
  isLikedVideosPage() {
    return window.location.pathname.includes('/playlist') && 
           window.location.search.includes('list=LL');
  },
  
  /**
   * Extract video ID from YouTube URL
   */
  extractVideoId(url) {
    const match = url.match(YOUTUBE_PATTERNS.VIDEO_URL_PATTERN);
    return match ? match[1] : null;
  },
  
  /**
   * Generate storage key for video
   */
  getVideoStorageKey(videoId) {
    return `video_${videoId}`;
  },
  
  /**
   * Generate storage key for session
   */
  getSessionStorageKey(sessionId) {
    return `session_${sessionId}`;
  }
};

// Export for different environments
if (typeof module !== 'undefined' && module.exports) {
  // Node.js environment
  module.exports = {
    YOUTUBE_PATTERNS,
    YOUTUBE_SELECTORS,
    STORAGE_KEYS,
    DEFAULT_SETTINGS,
    TIMING,
    LIMITS,
    MESSAGE_TYPES,
    ERROR_TYPES,
    METADATA_LEVELS,
    EXPORT_FORMATS,
    EXTENSION_STATES,
    UTILS
  };
} else {
  // Browser environment
  window.Constants = {
    YOUTUBE_PATTERNS,
    YOUTUBE_SELECTORS,
    STORAGE_KEYS,
    DEFAULT_SETTINGS,
    TIMING,
    LIMITS,
    MESSAGE_TYPES,
    ERROR_TYPES,
    METADATA_LEVELS,
    EXPORT_FORMATS,
    EXTENSION_STATES,
    UTILS
  };
}
