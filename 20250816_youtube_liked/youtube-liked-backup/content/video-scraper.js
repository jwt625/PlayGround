/**
 * Video Scraper for YouTube Liked Videos Backup Extension
 * Extracts metadata from YouTube DOM elements
 */

class VideoScraper {
  constructor() {
    this.selectors = window.Constants?.YOUTUBE_SELECTORS || {};
    this.patterns = window.Constants?.YOUTUBE_PATTERNS || {};
    this.dataSchemas = window.DataSchemas || {};
  }

  /**
   * Scrape all visible videos on the current page
   * @returns {Promise<Array<Object>>} Array of VideoRecord objects
   */
  async scrapeVisibleVideos() {
    const videoElements = this.getVideoElements();
    const videos = [];
    
    for (const element of videoElements) {
      try {
        const video = await this.scrapeVideoElement(element);
        if (video && video.videoId) {
          videos.push(video);
        }
      } catch (error) {
        console.error('Failed to scrape video element:', error);
      }
    }
    
    return videos;
  }

  /**
   * Get all video elements on the page
   * @returns {Array<Element>} Array of video container elements
   */
  getVideoElements() {
    // Try primary selector first
    let elements = document.querySelectorAll(this.selectors.VIDEO_CONTAINER);
    
    // Fallback to alternative selector
    if (elements.length === 0) {
      elements = document.querySelectorAll(this.selectors.VIDEO_CONTAINER_ALT);
    }
    
    return Array.from(elements);
  }

  /**
   * Scrape metadata from a single video element
   * @param {Element} element - Video container element
   * @returns {Promise<Object>} VideoRecord object
   */
  async scrapeVideoElement(element) {
    try {
      const videoData = {};
      
      // Extract basic video information
      const titleElement = element.querySelector(this.selectors.VIDEO_TITLE);
      if (titleElement) {
        videoData.title = this.cleanText(titleElement.textContent);
        videoData.url = titleElement.href;
        videoData.videoId = this.extractVideoId(titleElement.href);
      }
      
      if (!videoData.videoId) {
        return null; // Skip if we can't get video ID
      }
      
      // Extract duration
      const durationElement = element.querySelector(this.selectors.VIDEO_DURATION);
      if (durationElement) {
        videoData.durationFormatted = this.cleanText(durationElement.textContent);
        videoData.duration = this.parseDuration(videoData.durationFormatted);
      }
      
      // Extract channel information
      const channelElement = element.querySelector(this.selectors.CHANNEL_NAME);
      if (channelElement) {
        videoData.channelName = this.cleanText(channelElement.textContent);
        videoData.channelUrl = channelElement.href;
        videoData.channelId = this.extractChannelId(channelElement.href);
      }
      
      // Extract thumbnails
      const thumbnailElement = element.querySelector(this.selectors.VIDEO_THUMBNAIL);
      if (thumbnailElement) {
        videoData.thumbnails = this.extractThumbnails(thumbnailElement);
      }
      
      // Extract view count and upload date
      const viewCountElement = element.querySelector(this.selectors.VIEW_COUNT);
      if (viewCountElement) {
        videoData.viewCount = this.parseViewCount(viewCountElement.textContent);
      }
      
      const uploadDateElement = element.querySelector(this.selectors.UPLOAD_DATE);
      if (uploadDateElement) {
        videoData.uploadDate = this.parseUploadDate(uploadDateElement.textContent);
      }
      
      // Extract playlist position
      videoData.playlistPosition = this.getPlaylistPosition(element);
      
      // Create complete video record
      const video = this.dataSchemas.createVideoRecord(videoData);
      
      // Set data source and verification score
      video.dataSource = [this.dataSchemas.DataSource.DOM_SCRAPING];
      video.verificationScore = this.calculateVerificationScore(video);
      video.missingFields = this.findMissingFields(video);
      video.backupStatus = this.dataSchemas.BackupStatus.SCRAPED;
      
      return video;
    } catch (error) {
      console.error('Error scraping video element:', error);
      return null;
    }
  }

  /**
   * Extract video ID from YouTube URL
   * @param {string} url - YouTube video URL
   * @returns {string|null} Video ID or null
   */
  extractVideoId(url) {
    if (!url) return null;
    
    const match = url.match(this.patterns.VIDEO_URL_PATTERN);
    return match ? match[1] : null;
  }

  /**
   * Extract channel ID from channel URL
   * @param {string} url - YouTube channel URL
   * @returns {string|null} Channel ID or null
   */
  extractChannelId(url) {
    if (!url) return null;
    
    // Extract from various channel URL formats
    const patterns = [
      /\/channel\/([a-zA-Z0-9_-]+)/,
      /\/c\/([a-zA-Z0-9_-]+)/,
      /\/user\/([a-zA-Z0-9_-]+)/,
      /\/@([a-zA-Z0-9_-]+)/
    ];
    
    for (const pattern of patterns) {
      const match = url.match(pattern);
      if (match) return match[1];
    }
    
    return null;
  }

  /**
   * Extract thumbnails from thumbnail element
   * @param {Element} element - Thumbnail image element
   * @returns {Object} ThumbnailSet object
   */
  extractThumbnails(element) {
    const thumbnails = this.dataSchemas.createThumbnailSet();
    
    if (element.src) {
      // YouTube thumbnail URLs follow predictable patterns
      const baseUrl = element.src.replace(/\/[^\/]*\.jpg.*$/, '');
      
      thumbnails.default = `${baseUrl}/default.jpg`;
      thumbnails.medium = `${baseUrl}/mqdefault.jpg`;
      thumbnails.high = `${baseUrl}/hqdefault.jpg`;
      thumbnails.standard = `${baseUrl}/sddefault.jpg`;
      thumbnails.maxres = `${baseUrl}/maxresdefault.jpg`;
    }
    
    return thumbnails;
  }

  /**
   * Parse duration string to seconds
   * @param {string} durationStr - Duration string (e.g., "10:23")
   * @returns {number} Duration in seconds
   */
  parseDuration(durationStr) {
    if (!durationStr) return 0;
    
    const parts = durationStr.split(':').map(part => parseInt(part, 10));
    let seconds = 0;
    
    if (parts.length === 3) {
      // Hours:Minutes:Seconds
      seconds = parts[0] * 3600 + parts[1] * 60 + parts[2];
    } else if (parts.length === 2) {
      // Minutes:Seconds
      seconds = parts[0] * 60 + parts[1];
    } else if (parts.length === 1) {
      // Just seconds
      seconds = parts[0];
    }
    
    return seconds;
  }

  /**
   * Parse view count string to number
   * @param {string} viewStr - View count string (e.g., "1.2M views")
   * @returns {number|null} View count or null
   */
  parseViewCount(viewStr) {
    if (!viewStr) return null;
    
    const match = viewStr.match(/([\d,\.]+)\s*([KMB]?)/i);
    if (!match) return null;
    
    let count = parseFloat(match[1].replace(/,/g, ''));
    const multiplier = match[2].toUpperCase();
    
    switch (multiplier) {
      case 'K':
        count *= 1000;
        break;
      case 'M':
        count *= 1000000;
        break;
      case 'B':
        count *= 1000000000;
        break;
    }
    
    return Math.floor(count);
  }

  /**
   * Parse upload date string to Date object
   * @param {string} dateStr - Upload date string (e.g., "2 years ago")
   * @returns {Date|null} Upload date or null
   */
  parseUploadDate(dateStr) {
    if (!dateStr) return null;
    
    const now = new Date();
    const match = dateStr.match(/(\d+)\s+(second|minute|hour|day|week|month|year)s?\s+ago/i);
    
    if (!match) return null;
    
    const amount = parseInt(match[1], 10);
    const unit = match[2].toLowerCase();
    
    switch (unit) {
      case 'second':
        return new Date(now.getTime() - amount * 1000);
      case 'minute':
        return new Date(now.getTime() - amount * 60 * 1000);
      case 'hour':
        return new Date(now.getTime() - amount * 60 * 60 * 1000);
      case 'day':
        return new Date(now.getTime() - amount * 24 * 60 * 60 * 1000);
      case 'week':
        return new Date(now.getTime() - amount * 7 * 24 * 60 * 60 * 1000);
      case 'month':
        return new Date(now.getTime() - amount * 30 * 24 * 60 * 60 * 1000);
      case 'year':
        return new Date(now.getTime() - amount * 365 * 24 * 60 * 60 * 1000);
      default:
        return null;
    }
  }

  /**
   * Get playlist position of video element
   * @param {Element} element - Video container element
   * @returns {number|null} Playlist position or null
   */
  getPlaylistPosition(element) {
    // Try to find position indicator in the element
    const positionElement = element.querySelector('[data-index]');
    if (positionElement) {
      return parseInt(positionElement.dataset.index, 10);
    }
    
    // Fallback: count position among siblings
    const parent = element.parentElement;
    if (parent) {
      const siblings = Array.from(parent.children);
      return siblings.indexOf(element) + 1;
    }
    
    return null;
  }

  /**
   * Calculate verification score based on available data
   * @param {Object} video - VideoRecord object
   * @returns {number} Verification score (0-100)
   */
  calculateVerificationScore(video) {
    let score = 0;
    const weights = {
      videoId: 20,
      title: 15,
      channelName: 15,
      duration: 10,
      thumbnails: 10,
      viewCount: 10,
      uploadDate: 10,
      channelId: 5,
      playlistPosition: 5
    };
    
    for (const [field, weight] of Object.entries(weights)) {
      if (video[field] && video[field] !== '' && video[field] !== 0) {
        score += weight;
      }
    }
    
    return Math.min(score, 100);
  }

  /**
   * Find missing fields in video record
   * @param {Object} video - VideoRecord object
   * @returns {Array<string>} Array of missing field names
   */
  findMissingFields(video) {
    const requiredFields = window.Constants?.METADATA_LEVELS?.STANDARD || [];
    const missingFields = [];
    
    for (const field of requiredFields) {
      if (!video[field] || video[field] === '' || video[field] === 0) {
        missingFields.push(field);
      }
    }
    
    return missingFields;
  }

  /**
   * Clean text content by removing extra whitespace
   * @param {string} text - Text to clean
   * @returns {string} Cleaned text
   */
  cleanText(text) {
    if (!text) return '';
    return text.trim().replace(/\s+/g, ' ');
  }

  /**
   * Check if the current page is YouTube liked videos
   * @returns {boolean} Whether on liked videos page
   */
  isLikedVideosPage() {
    return window.Constants?.UTILS?.isLikedVideosPage() || false;
  }

  /**
   * Get total video count from page
   * @returns {number|null} Total video count or null
   */
  getTotalVideoCount() {
    const countElement = document.querySelector(this.selectors.VIDEO_COUNT);
    if (countElement) {
      const match = countElement.textContent.match(/(\d+)/);
      return match ? parseInt(match[1], 10) : null;
    }
    return null;
  }
}

// Export for different environments
if (typeof module !== 'undefined' && module.exports) {
  module.exports = VideoScraper;
} else {
  window.VideoScraper = VideoScraper;
}
