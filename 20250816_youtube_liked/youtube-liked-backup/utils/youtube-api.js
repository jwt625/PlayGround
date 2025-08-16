/**
 * YouTube API Utilities for YouTube Liked Videos Backup Extension
 * DOM interaction utilities and YouTube-specific helpers
 */

class YouTubeAPI {
  constructor() {
    // Use self for service worker compatibility, but fall back to window for content scripts
    const globalScope = typeof window !== 'undefined' ? window : self;
    this.selectors = globalScope.Constants?.YOUTUBE_SELECTORS || {};
    this.patterns = globalScope.Constants?.YOUTUBE_PATTERNS || {};
  }

  /**
   * Check if current page is YouTube liked videos playlist
   * @returns {boolean} Whether on liked videos page
   */
  isLikedVideosPage() {
    // Only available in content script context where window.location exists
    if (typeof window !== 'undefined' && window.location) {
      return window.location.pathname.includes('/playlist') &&
             window.location.search.includes('list=LL');
    }
    return false;
  }

  /**
   * Wait for element to appear in DOM
   * @param {string} selector - CSS selector
   * @param {number} timeout - Timeout in milliseconds
   * @returns {Promise<Element|null>} Element or null if timeout
   */
  async waitForElement(selector, timeout = 10000) {
    return new Promise((resolve) => {
      const element = document.querySelector(selector);
      if (element) {
        resolve(element);
        return;
      }

      const observer = new MutationObserver((mutations, obs) => {
        const element = document.querySelector(selector);
        if (element) {
          obs.disconnect();
          resolve(element);
        }
      });

      observer.observe(document.body, {
        childList: true,
        subtree: true
      });

      setTimeout(() => {
        observer.disconnect();
        resolve(null);
      }, timeout);
    });
  }

  /**
   * Wait for multiple elements to appear
   * @param {Array<string>} selectors - Array of CSS selectors
   * @param {number} timeout - Timeout in milliseconds
   * @returns {Promise<Array<Element>>} Array of elements
   */
  async waitForElements(selectors, timeout = 10000) {
    const promises = selectors.map(selector => this.waitForElement(selector, timeout));
    return Promise.all(promises);
  }

  /**
   * Scroll element into view smoothly
   * @param {Element} element - Element to scroll to
   * @param {Object} options - Scroll options
   */
  scrollIntoView(element, options = {}) {
    const defaultOptions = {
      behavior: 'smooth',
      block: 'center',
      inline: 'nearest'
    };

    element.scrollIntoView({ ...defaultOptions, ...options });
  }

  /**
   * Click element with retry logic
   * @param {Element|string} elementOrSelector - Element or CSS selector
   * @param {number} retries - Number of retries
   * @returns {Promise<boolean>} Whether click was successful
   */
  async clickElement(elementOrSelector, retries = 3) {
    let element;
    
    if (typeof elementOrSelector === 'string') {
      element = await this.waitForElement(elementOrSelector, 5000);
    } else {
      element = elementOrSelector;
    }

    if (!element) {
      return false;
    }

    for (let i = 0; i < retries; i++) {
      try {
        // Scroll element into view
        this.scrollIntoView(element);
        
        // Wait a bit for scroll to complete
        await this.delay(500);
        
        // Try different click methods
        if (element.click) {
          element.click();
        } else {
          // Fallback to dispatching click event
          const clickEvent = new MouseEvent('click', {
            view: window,
            bubbles: true,
            cancelable: true
          });
          element.dispatchEvent(clickEvent);
        }
        
        return true;
      } catch (error) {
        console.warn(`Click attempt ${i + 1} failed:`, error);
        if (i < retries - 1) {
          await this.delay(1000);
        }
      }
    }

    return false;
  }

  /**
   * Extract video ID from various YouTube URL formats
   * @param {string} url - YouTube URL
   * @returns {string|null} Video ID or null
   */
  extractVideoId(url) {
    if (!url) return null;

    const patterns = [
      /(?:youtube\.com\/watch\?v=|youtu\.be\/|youtube\.com\/embed\/)([a-zA-Z0-9_-]{11})/,
      /youtube\.com\/v\/([a-zA-Z0-9_-]{11})/,
      /youtube\.com\/watch\?.*v=([a-zA-Z0-9_-]{11})/
    ];

    for (const pattern of patterns) {
      const match = url.match(pattern);
      if (match) return match[1];
    }

    return null;
  }

  /**
   * Extract channel ID from channel URL
   * @param {string} url - Channel URL
   * @returns {string|null} Channel ID or null
   */
  extractChannelId(url) {
    if (!url) return null;

    const patterns = [
      /youtube\.com\/channel\/([a-zA-Z0-9_-]+)/,
      /youtube\.com\/c\/([a-zA-Z0-9_-]+)/,
      /youtube\.com\/user\/([a-zA-Z0-9_-]+)/,
      /youtube\.com\/@([a-zA-Z0-9_-]+)/
    ];

    for (const pattern of patterns) {
      const match = url.match(pattern);
      if (match) return match[1];
    }

    return null;
  }

  /**
   * Get video thumbnail URLs for a video ID
   * @param {string} videoId - YouTube video ID
   * @returns {Object} Thumbnail URLs object
   */
  getThumbnailUrls(videoId) {
    if (!videoId || videoId.length !== 11) {
      return {};
    }

    const baseUrl = `https://i.ytimg.com/vi/${videoId}`;
    
    return {
      default: `${baseUrl}/default.jpg`,      // 120x90
      medium: `${baseUrl}/mqdefault.jpg`,     // 320x180
      high: `${baseUrl}/hqdefault.jpg`,       // 480x360
      standard: `${baseUrl}/sddefault.jpg`,   // 640x480
      maxres: `${baseUrl}/maxresdefault.jpg`  // 1280x720
    };
  }

  /**
   * Check if thumbnail URL is accessible
   * @param {string} url - Thumbnail URL
   * @returns {Promise<boolean>} Whether thumbnail is accessible
   */
  async checkThumbnailAccessibility(url) {
    try {
      const response = await fetch(url, { method: 'HEAD' });
      return response.ok;
    } catch (error) {
      return false;
    }
  }

  /**
   * Parse YouTube duration string to seconds
   * @param {string} durationStr - Duration string (e.g., "10:23", "1:30:45")
   * @returns {number} Duration in seconds
   */
  parseDuration(durationStr) {
    if (!durationStr || typeof durationStr !== 'string') {
      return 0;
    }

    const parts = durationStr.split(':').map(part => parseInt(part, 10) || 0);
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

    return Math.max(0, seconds);
  }

  /**
   * Format seconds to duration string
   * @param {number} seconds - Duration in seconds
   * @returns {string} Formatted duration string
   */
  formatDuration(seconds) {
    if (typeof seconds !== 'number' || seconds < 0) {
      return '0:00';
    }

    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    const secs = seconds % 60;

    if (hours > 0) {
      return `${hours}:${minutes.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
    } else {
      return `${minutes}:${secs.toString().padStart(2, '0')}`;
    }
  }

  /**
   * Parse view count string to number
   * @param {string} viewStr - View count string (e.g., "1.2M views", "1,234 views")
   * @returns {number|null} View count or null
   */
  parseViewCount(viewStr) {
    if (!viewStr || typeof viewStr !== 'string') {
      return null;
    }

    // Remove "views" and other text, keep only numbers and multipliers
    const cleanStr = viewStr.replace(/[^\d.,KMB]/gi, '');
    const match = cleanStr.match(/([\d,\.]+)\s*([KMB]?)/i);
    
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
   * Parse relative date string to Date object
   * @param {string} dateStr - Relative date string (e.g., "2 years ago")
   * @returns {Date|null} Date object or null
   */
  parseRelativeDate(dateStr) {
    if (!dateStr || typeof dateStr !== 'string') {
      return null;
    }

    const now = new Date();
    const match = dateStr.match(/(\d+)\s+(second|minute|hour|day|week|month|year)s?\s+ago/i);

    if (!match) return null;

    const amount = parseInt(match[1], 10);
    const unit = match[2].toLowerCase();

    const msPerUnit = {
      second: 1000,
      minute: 60 * 1000,
      hour: 60 * 60 * 1000,
      day: 24 * 60 * 60 * 1000,
      week: 7 * 24 * 60 * 60 * 1000,
      month: 30 * 24 * 60 * 60 * 1000,
      year: 365 * 24 * 60 * 60 * 1000
    };

    const ms = msPerUnit[unit];
    if (!ms) return null;

    return new Date(now.getTime() - (amount * ms));
  }

  /**
   * Get page metadata
   * @returns {Object} Page metadata
   */
  getPageMetadata() {
    return {
      url: window.location.href,
      title: document.title,
      isLikedVideosPage: this.isLikedVideosPage(),
      timestamp: new Date()
    };
  }

  /**
   * Monitor for page navigation changes
   * @param {Function} callback - Callback function for navigation changes
   * @returns {Function} Cleanup function
   */
  onNavigationChange(callback) {
    let lastUrl = location.href;
    
    const observer = new MutationObserver(() => {
      const url = location.href;
      if (url !== lastUrl) {
        lastUrl = url;
        callback(url);
      }
    });

    observer.observe(document, { subtree: true, childList: true });

    return () => observer.disconnect();
  }

  /**
   * Utility delay function
   * @param {number} ms - Milliseconds to delay
   * @returns {Promise<void>}
   */
  delay(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
  }

  /**
   * Get element text content safely
   * @param {Element} element - DOM element
   * @returns {string} Cleaned text content
   */
  getTextContent(element) {
    if (!element) return '';
    
    return element.textContent?.trim().replace(/\s+/g, ' ') || '';
  }

  /**
   * Get element attribute safely
   * @param {Element} element - DOM element
   * @param {string} attribute - Attribute name
   * @returns {string|null} Attribute value or null
   */
  getAttribute(element, attribute) {
    if (!element || !attribute) return null;
    
    return element.getAttribute(attribute);
  }

  /**
   * Check if element is visible
   * @param {Element} element - DOM element
   * @returns {boolean} Whether element is visible
   */
  isElementVisible(element) {
    if (!element) return false;
    
    const rect = element.getBoundingClientRect();
    const style = window.getComputedStyle(element);
    
    return rect.width > 0 && 
           rect.height > 0 && 
           style.visibility !== 'hidden' && 
           style.display !== 'none' &&
           element.offsetParent !== null;
  }

  /**
   * Find element by text content
   * @param {string} text - Text to search for
   * @param {string} tag - Tag name to search within (optional)
   * @returns {Element|null} Found element or null
   */
  findElementByText(text, tag = '*') {
    const xpath = `//${tag}[contains(text(), "${text}")]`;
    const result = document.evaluate(xpath, document, null, XPathResult.FIRST_ORDERED_NODE_TYPE, null);
    return result.singleNodeValue;
  }
}

// Export for different environments
if (typeof module !== 'undefined' && module.exports) {
  module.exports = YouTubeAPI;
} else {
  // Use self for service worker compatibility, but fall back to window for content scripts
  const globalScope = typeof window !== 'undefined' ? window : self;
  globalScope.YouTubeAPI = YouTubeAPI;
}
