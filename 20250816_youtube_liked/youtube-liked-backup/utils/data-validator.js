/**
 * Data Validator for YouTube Liked Videos Backup Extension
 * Ensures metadata completeness and integrity
 */

class DataValidator {
  constructor() {
    this.requiredFields = {
      BASIC: ['videoId', 'title', 'channelName', 'duration'],
      STANDARD: ['videoId', 'title', 'channelName', 'duration', 'thumbnails', 'url'],
      COMPLETE: ['videoId', 'title', 'channelName', 'duration', 'thumbnails', 'url', 'uploadDate', 'viewCount']
    };
    
    this.fieldValidators = {
      videoId: this.validateVideoId.bind(this),
      title: this.validateTitle.bind(this),
      channelName: this.validateChannelName.bind(this),
      duration: this.validateDuration.bind(this),
      url: this.validateUrl.bind(this),
      thumbnails: this.validateThumbnails.bind(this),
      uploadDate: this.validateDate.bind(this),
      likedDate: this.validateDate.bind(this),
      scrapedDate: this.validateDate.bind(this),
      viewCount: this.validateNumber.bind(this),
      likeCount: this.validateNumber.bind(this),
      verificationScore: this.validateScore.bind(this)
    };
  }

  /**
   * Validate a video record
   * @param {Object} video - VideoRecord to validate
   * @param {string} level - Validation level ('BASIC', 'STANDARD', 'COMPLETE')
   * @returns {Object} Validation result
   */
  validateVideo(video, level = 'STANDARD') {
    const result = {
      isValid: true,
      errors: [],
      warnings: [],
      missingFields: [],
      score: 0,
      level: level
    };

    if (!video || typeof video !== 'object') {
      result.isValid = false;
      result.errors.push('Video record is not a valid object');
      return result;
    }

    // Check required fields for the specified level
    const requiredFields = this.requiredFields[level] || this.requiredFields.STANDARD;
    
    for (const field of requiredFields) {
      if (!this.hasValidValue(video[field])) {
        result.missingFields.push(field);
        result.errors.push(`Missing required field: ${field}`);
      }
    }

    // Validate individual fields
    for (const [field, value] of Object.entries(video)) {
      if (this.fieldValidators[field]) {
        const fieldResult = this.fieldValidators[field](value, field);
        if (!fieldResult.isValid) {
          result.errors.push(...fieldResult.errors);
          result.warnings.push(...fieldResult.warnings);
        }
      }
    }

    // Calculate verification score
    result.score = this.calculateVerificationScore(video);
    
    // Set overall validity
    result.isValid = result.errors.length === 0 && result.missingFields.length === 0;

    return result;
  }

  /**
   * Validate video ID
   * @param {string} videoId - Video ID to validate
   * @returns {Object} Validation result
   */
  validateVideoId(videoId) {
    const result = { isValid: true, errors: [], warnings: [] };

    if (!videoId || typeof videoId !== 'string') {
      result.isValid = false;
      result.errors.push('Video ID must be a non-empty string');
      return result;
    }

    if (videoId.length !== 11) {
      result.isValid = false;
      result.errors.push('Video ID must be exactly 11 characters long');
    }

    if (!/^[a-zA-Z0-9_-]+$/.test(videoId)) {
      result.isValid = false;
      result.errors.push('Video ID contains invalid characters');
    }

    return result;
  }

  /**
   * Validate video title
   * @param {string} title - Title to validate
   * @returns {Object} Validation result
   */
  validateTitle(title) {
    const result = { isValid: true, errors: [], warnings: [] };

    if (!title || typeof title !== 'string') {
      result.isValid = false;
      result.errors.push('Title must be a non-empty string');
      return result;
    }

    if (title.trim().length === 0) {
      result.isValid = false;
      result.errors.push('Title cannot be empty or only whitespace');
    }

    if (title.length > 1000) {
      result.warnings.push('Title is unusually long (>1000 characters)');
    }

    return result;
  }

  /**
   * Validate channel name
   * @param {string} channelName - Channel name to validate
   * @returns {Object} Validation result
   */
  validateChannelName(channelName) {
    const result = { isValid: true, errors: [], warnings: [] };

    if (!channelName || typeof channelName !== 'string') {
      result.isValid = false;
      result.errors.push('Channel name must be a non-empty string');
      return result;
    }

    if (channelName.trim().length === 0) {
      result.isValid = false;
      result.errors.push('Channel name cannot be empty or only whitespace');
    }

    return result;
  }

  /**
   * Validate duration
   * @param {number} duration - Duration in seconds
   * @returns {Object} Validation result
   */
  validateDuration(duration) {
    const result = { isValid: true, errors: [], warnings: [] };

    if (typeof duration !== 'number') {
      result.isValid = false;
      result.errors.push('Duration must be a number');
      return result;
    }

    if (duration < 0) {
      result.isValid = false;
      result.errors.push('Duration cannot be negative');
    }

    if (duration > 86400) { // 24 hours
      result.warnings.push('Duration is unusually long (>24 hours)');
    }

    return result;
  }

  /**
   * Validate URL
   * @param {string} url - URL to validate
   * @returns {Object} Validation result
   */
  validateUrl(url) {
    const result = { isValid: true, errors: [], warnings: [] };

    if (!url || typeof url !== 'string') {
      result.isValid = false;
      result.errors.push('URL must be a non-empty string');
      return result;
    }

    try {
      const urlObj = new URL(url);
      
      if (!urlObj.hostname.includes('youtube.com')) {
        result.warnings.push('URL is not from YouTube domain');
      }
      
    } catch (error) {
      result.isValid = false;
      result.errors.push('URL is not a valid URL format');
    }

    return result;
  }

  /**
   * Validate thumbnails object
   * @param {Object} thumbnails - Thumbnails object to validate
   * @returns {Object} Validation result
   */
  validateThumbnails(thumbnails) {
    const result = { isValid: true, errors: [], warnings: [] };

    if (!thumbnails || typeof thumbnails !== 'object') {
      result.isValid = false;
      result.errors.push('Thumbnails must be an object');
      return result;
    }

    const requiredThumbnails = ['default', 'medium', 'high'];
    const missingThumbnails = [];

    for (const thumb of requiredThumbnails) {
      if (!thumbnails[thumb] || typeof thumbnails[thumb] !== 'string') {
        missingThumbnails.push(thumb);
      }
    }

    if (missingThumbnails.length > 0) {
      result.warnings.push(`Missing thumbnail sizes: ${missingThumbnails.join(', ')}`);
    }

    // Validate thumbnail URLs
    for (const [size, url] of Object.entries(thumbnails)) {
      if (url && typeof url === 'string') {
        try {
          new URL(url);
        } catch (error) {
          result.warnings.push(`Invalid thumbnail URL for ${size}: ${url}`);
        }
      }
    }

    return result;
  }

  /**
   * Validate date
   * @param {Date|string} date - Date to validate
   * @returns {Object} Validation result
   */
  validateDate(date) {
    const result = { isValid: true, errors: [], warnings: [] };

    if (!date) {
      result.warnings.push('Date is missing');
      return result;
    }

    let dateObj;
    if (typeof date === 'string') {
      dateObj = new Date(date);
    } else if (date instanceof Date) {
      dateObj = date;
    } else {
      result.isValid = false;
      result.errors.push('Date must be a Date object or valid date string');
      return result;
    }

    if (isNaN(dateObj.getTime())) {
      result.isValid = false;
      result.errors.push('Date is not a valid date');
    }

    // Check if date is in the future
    if (dateObj > new Date()) {
      result.warnings.push('Date is in the future');
    }

    // Check if date is too old (before YouTube existed)
    const youtubeStart = new Date('2005-02-14');
    if (dateObj < youtubeStart) {
      result.warnings.push('Date is before YouTube was founded');
    }

    return result;
  }

  /**
   * Validate number
   * @param {number} value - Number to validate
   * @returns {Object} Validation result
   */
  validateNumber(value) {
    const result = { isValid: true, errors: [], warnings: [] };

    if (value !== null && value !== undefined) {
      if (typeof value !== 'number') {
        result.isValid = false;
        result.errors.push('Value must be a number');
      } else if (value < 0) {
        result.warnings.push('Number is negative');
      }
    }

    return result;
  }

  /**
   * Validate verification score
   * @param {number} score - Score to validate (0-100)
   * @returns {Object} Validation result
   */
  validateScore(score) {
    const result = { isValid: true, errors: [], warnings: [] };

    if (typeof score !== 'number') {
      result.isValid = false;
      result.errors.push('Verification score must be a number');
      return result;
    }

    if (score < 0 || score > 100) {
      result.isValid = false;
      result.errors.push('Verification score must be between 0 and 100');
    }

    return result;
  }

  /**
   * Calculate verification score based on available data
   * @param {Object} video - VideoRecord object
   * @returns {number} Verification score (0-100)
   */
  calculateVerificationScore(video) {
    const weights = {
      videoId: 20,
      title: 15,
      channelName: 15,
      duration: 10,
      thumbnails: 10,
      url: 10,
      viewCount: 5,
      uploadDate: 5,
      channelId: 5,
      playlistPosition: 5
    };

    let score = 0;
    let maxScore = 0;

    for (const [field, weight] of Object.entries(weights)) {
      maxScore += weight;
      
      if (this.hasValidValue(video[field])) {
        score += weight;
        
        // Bonus points for high-quality data
        if (field === 'thumbnails' && video[field]) {
          const thumbCount = Object.keys(video[field]).length;
          if (thumbCount >= 3) score += 2; // Bonus for multiple thumbnail sizes
        }
        
        if (field === 'title' && video[field] && video[field].length > 10) {
          score += 2; // Bonus for descriptive titles
        }
      }
    }

    return Math.min(Math.round((score / maxScore) * 100), 100);
  }

  /**
   * Check if a value is valid (not null, undefined, empty string, or 0 for certain fields)
   * @param {*} value - Value to check
   * @returns {boolean} Whether value is valid
   */
  hasValidValue(value) {
    if (value === null || value === undefined) {
      return false;
    }
    
    if (typeof value === 'string' && value.trim() === '') {
      return false;
    }
    
    if (typeof value === 'object' && Object.keys(value).length === 0) {
      return false;
    }
    
    return true;
  }

  /**
   * Validate batch of videos
   * @param {Array<Object>} videos - Array of video records
   * @param {string} level - Validation level
   * @returns {Object} Batch validation result
   */
  validateBatch(videos, level = 'STANDARD') {
    const result = {
      totalVideos: videos.length,
      validVideos: 0,
      invalidVideos: 0,
      averageScore: 0,
      errors: [],
      warnings: [],
      videoResults: []
    };

    let totalScore = 0;

    for (let i = 0; i < videos.length; i++) {
      const videoResult = this.validateVideo(videos[i], level);
      videoResult.index = i;
      videoResult.videoId = videos[i]?.videoId || `unknown-${i}`;
      
      result.videoResults.push(videoResult);
      
      if (videoResult.isValid) {
        result.validVideos++;
      } else {
        result.invalidVideos++;
      }
      
      totalScore += videoResult.score;
      
      // Collect unique errors and warnings
      videoResult.errors.forEach(error => {
        if (!result.errors.includes(error)) {
          result.errors.push(error);
        }
      });
      
      videoResult.warnings.forEach(warning => {
        if (!result.warnings.includes(warning)) {
          result.warnings.push(warning);
        }
      });
    }

    result.averageScore = videos.length > 0 ? Math.round(totalScore / videos.length) : 0;

    return result;
  }

  /**
   * Get validation summary for a video
   * @param {Object} video - VideoRecord object
   * @returns {string} Human-readable validation summary
   */
  getValidationSummary(video) {
    const validation = this.validateVideo(video);
    
    if (validation.isValid) {
      return `✅ Valid (Score: ${validation.score}/100)`;
    } else {
      const errorCount = validation.errors.length;
      const warningCount = validation.warnings.length;
      return `❌ Invalid (${errorCount} errors, ${warningCount} warnings, Score: ${validation.score}/100)`;
    }
  }
}

// Export for different environments
if (typeof module !== 'undefined' && module.exports) {
  module.exports = DataValidator;
} else {
  window.DataValidator = DataValidator;
}
