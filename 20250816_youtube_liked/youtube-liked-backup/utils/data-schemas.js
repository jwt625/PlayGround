/**
 * Data schemas and interfaces for YouTube Liked Videos Backup Extension
 * Based on RFD-000 specifications
 */

// Enums
const BackupStatus = {
  PENDING: 'pending',
  SCRAPED: 'scraped', 
  VERIFIED: 'verified',
  EXPORTED: 'exported',
  FAILED: 'failed'
};

const RemovalStatus = {
  KEPT: 'kept',
  QUEUED_FOR_REMOVAL: 'queued',
  REMOVED: 'removed', 
  REMOVAL_FAILED: 'failed',
  RESTORED: 'restored'
};

const DataSource = {
  DOM_SCRAPING: 'dom',
  NETWORK_INTERCEPT: 'network',
  YOUTUBE_API: 'api',
  THUMBNAIL_DOWNLOAD: 'thumbnail'
};

/**
 * Creates a new VideoRecord with default values
 * @param {Object} data - Initial video data
 * @returns {Object} VideoRecord object
 */
function createVideoRecord(data = {}) {
  const now = new Date();
  
  return {
    // Core Identifiers
    videoId: data.videoId || '',
    url: data.url || '',
    
    // Metadata
    title: data.title || '',
    description: data.description || null,
    duration: data.duration || 0,
    durationFormatted: data.durationFormatted || '',
    
    // Channel Information
    channelId: data.channelId || '',
    channelName: data.channelName || '',
    channelUrl: data.channelUrl || '',
    channelHandle: data.channelHandle || null,
    
    // Timestamps
    uploadDate: data.uploadDate || null,
    likedDate: data.likedDate || now,
    scrapedDate: data.scrapedDate || now,
    lastVerified: data.lastVerified || now,
    
    // Media Assets
    thumbnails: data.thumbnails || createThumbnailSet(),
    
    // Engagement Data
    viewCount: data.viewCount || null,
    likeCount: data.likeCount || null,
    
    // Backup Metadata
    backupStatus: data.backupStatus || BackupStatus.PENDING,
    removalStatus: data.removalStatus || RemovalStatus.KEPT,
    dataSource: data.dataSource || [],
    
    // Quality Assurance
    verificationScore: data.verificationScore || 0,
    missingFields: data.missingFields || [],
    
    // Additional Context
    playlistPosition: data.playlistPosition || null,
    tags: data.tags || null,
    category: data.category || null,
    language: data.language || null
  };
}

/**
 * Creates a new ThumbnailSet with default values
 * @param {Object} data - Initial thumbnail data
 * @returns {Object} ThumbnailSet object
 */
function createThumbnailSet(data = {}) {
  return {
    default: data.default || '',    // 120x90
    medium: data.medium || '',      // 320x180
    high: data.high || '',          // 480x360
    standard: data.standard || null, // 640x480
    maxres: data.maxres || null     // 1280x720
  };
}

/**
 * Creates a new BackupSession with default values
 * @param {Object} data - Initial session data
 * @returns {Object} BackupSession object
 */
function createBackupSession(data = {}) {
  return {
    sessionId: data.sessionId || generateSessionId(),
    startTime: data.startTime || new Date(),
    endTime: data.endTime || null,
    videosProcessed: data.videosProcessed || 0,
    videosRemoved: data.videosRemoved || 0,
    errors: data.errors || [],
    settings: data.settings || createBackupSettings()
  };
}

/**
 * Creates a new BackupSettings with default values
 * @param {Object} data - Initial settings data
 * @returns {Object} BackupSettings object
 */
function createBackupSettings(data = {}) {
  return {
    autoRemoveAfterBackup: data.autoRemoveAfterBackup || false,
    verificationThreshold: data.verificationThreshold || 80,
    removalRateLimit: data.removalRateLimit || 20, // Max removals per minute
    exportFormat: data.exportFormat || 'json',
    includeDescriptions: data.includeDescriptions || true,
    downloadThumbnails: data.downloadThumbnails || false,
    maxRetries: data.maxRetries || 3
  };
}

/**
 * Creates an error log entry
 * @param {string} type - Error type
 * @param {string} message - Error message
 * @param {Object} context - Additional context
 * @returns {Object} ErrorLog object
 */
function createErrorLog(type, message, context = {}) {
  return {
    timestamp: new Date(),
    type: type,
    message: message,
    context: context
  };
}

/**
 * Generates a unique session ID
 * @returns {string} Session ID
 */
function generateSessionId() {
  return `session_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
}

/**
 * Validates a VideoRecord object
 * @param {Object} video - VideoRecord to validate
 * @returns {Object} Validation result with isValid and errors
 */
function validateVideoRecord(video) {
  const errors = [];
  const requiredFields = ['videoId', 'title', 'channelName', 'duration'];
  
  requiredFields.forEach(field => {
    if (!video[field]) {
      errors.push(`Missing required field: ${field}`);
    }
  });
  
  if (video.videoId && video.videoId.length !== 11) {
    errors.push('Invalid videoId length (must be 11 characters)');
  }
  
  if (video.duration && typeof video.duration !== 'number') {
    errors.push('Duration must be a number');
  }
  
  return {
    isValid: errors.length === 0,
    errors: errors
  };
}

// Export all schemas and utilities
if (typeof module !== 'undefined' && module.exports) {
  // Node.js environment
  module.exports = {
    BackupStatus,
    RemovalStatus,
    DataSource,
    createVideoRecord,
    createThumbnailSet,
    createBackupSession,
    createBackupSettings,
    createErrorLog,
    generateSessionId,
    validateVideoRecord
  };
} else {
  // Browser environment
  window.DataSchemas = {
    BackupStatus,
    RemovalStatus,
    DataSource,
    createVideoRecord,
    createThumbnailSet,
    createBackupSession,
    createBackupSettings,
    createErrorLog,
    generateSessionId,
    validateVideoRecord
  };
}
