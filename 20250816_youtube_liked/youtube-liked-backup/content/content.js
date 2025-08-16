/**
 * Content Script for YouTube Liked Videos Backup Extension
 * Main coordinator for scraping operations on YouTube
 */

class YouTubeBackupContent {
  constructor() {
    this.videoScraper = new VideoScraper();
    this.paginationHandler = new PaginationHandler(this.videoScraper);
    this.isBackupActive = false;
    this.currentSession = null;
    this.backupProgress = {
      processed: 0,
      total: 0,
      batch: 0,
      errors: []
    };
    
    this.init();
  }

  /**
   * Initialize content script
   */
  async init() {
    // Only run on YouTube liked videos page
    if (!this.isLikedVideosPage()) {
      return;
    }
    
    console.log('YouTube Liked Videos Backup Extension loaded');
    
    // Set up message listeners
    this.setupMessageListeners();
    
    // Wait for page to fully load
    if (document.readyState === 'loading') {
      document.addEventListener('DOMContentLoaded', () => this.onPageReady());
    } else {
      this.onPageReady();
    }
  }

  /**
   * Handle page ready state
   */
  async onPageReady() {
    // Notify background script that we're ready
    this.sendMessage({
      type: window.Constants?.MESSAGE_TYPES?.BACKUP_STATUS || 'backupStatus',
      data: {
        ready: true,
        url: window.location.href,
        totalVideos: this.videoScraper.getTotalVideoCount()
      }
    });
  }

  /**
   * Set up message listeners for communication with background script
   */
  setupMessageListeners() {
    chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
      this.handleMessage(message, sender, sendResponse);
      return true; // Keep message channel open for async responses
    });
  }

  /**
   * Handle messages from background script and popup
   * @param {Object} message - Message object
   * @param {Object} sender - Message sender
   * @param {Function} sendResponse - Response callback
   */
  async handleMessage(message, sender, sendResponse) {
    try {
      switch (message.type) {
        case window.Constants?.MESSAGE_TYPES?.START_BACKUP:
          await this.startBackup(message.data);
          sendResponse({ success: true });
          break;
          
        case window.Constants?.MESSAGE_TYPES?.PAUSE_BACKUP:
          await this.pauseBackup();
          sendResponse({ success: true });
          break;
          
        case window.Constants?.MESSAGE_TYPES?.STOP_BACKUP:
          await this.stopBackup();
          sendResponse({ success: true });
          break;
          
        case window.Constants?.MESSAGE_TYPES?.GET_STATUS:
          sendResponse(this.getBackupStatus());
          break;
          
        case window.Constants?.MESSAGE_TYPES?.SETTINGS_UPDATED:
          await this.updateSettings(message.data);
          sendResponse({ success: true });
          break;
          
        default:
          console.warn('Unknown message type:', message.type);
          sendResponse({ success: false, error: 'Unknown message type' });
      }
    } catch (error) {
      console.error('Error handling message:', error);
      sendResponse({ success: false, error: error.message });
    }
  }

  /**
   * Start backup process
   * @param {Object} options - Backup options
   */
  async startBackup(options = {}) {
    if (this.isBackupActive) {
      throw new Error('Backup already in progress');
    }
    
    if (!this.isLikedVideosPage()) {
      throw new Error('Not on YouTube liked videos page');
    }
    
    this.isBackupActive = true;
    this.backupProgress = {
      processed: 0,
      total: 0,
      batch: 0,
      errors: [],
      startTime: new Date()
    };
    
    try {
      console.log('Starting backup process...');
      
      // Reset pagination handler
      this.paginationHandler.reset();
      
      // Start processing all videos
      const allVideos = await this.paginationHandler.processAllVideos(
        (batch, batchNumber) => this.onBatchProcessed(batch, batchNumber),
        (progress) => this.onProgress(progress)
      );
      
      // Complete backup
      await this.completeBackup(allVideos);
      
    } catch (error) {
      console.error('Backup failed:', error);
      this.backupProgress.errors.push({
        type: 'BACKUP_ERROR',
        message: error.message,
        timestamp: new Date()
      });
      
      await this.sendMessage({
        type: window.Constants?.MESSAGE_TYPES?.ERROR_OCCURRED,
        data: {
          error: error.message,
          progress: this.backupProgress
        }
      });
    } finally {
      this.isBackupActive = false;
    }
  }

  /**
   * Handle batch processing
   * @param {Array<Object>} videos - Batch of videos
   * @param {number} batchNumber - Batch number
   */
  async onBatchProcessed(videos, batchNumber) {
    console.log(`Processing batch ${batchNumber}: ${videos.length} videos`);
    
    // Send videos to background script for storage
    await this.sendMessage({
      type: window.Constants?.MESSAGE_TYPES?.VIDEO_SCRAPED,
      data: {
        videos: videos,
        batchNumber: batchNumber
      }
    });
    
    // Update progress
    this.backupProgress.batch = batchNumber;
    this.backupProgress.processed += videos.length;
    
    // Send progress update
    await this.sendMessage({
      type: window.Constants?.MESSAGE_TYPES?.BATCH_COMPLETE,
      data: {
        batchNumber: batchNumber,
        videosInBatch: videos.length,
        totalProcessed: this.backupProgress.processed,
        progress: this.backupProgress
      }
    });
  }

  /**
   * Handle progress updates
   * @param {Object} progress - Progress information
   */
  async onProgress(progress) {
    this.backupProgress.processed = progress.processed;
    this.backupProgress.hasMore = progress.hasMore;
    
    // Send progress to background script
    await this.sendMessage({
      type: window.Constants?.MESSAGE_TYPES?.BACKUP_STATUS,
      data: {
        progress: this.backupProgress,
        isActive: this.isBackupActive
      }
    });
  }

  /**
   * Complete backup process
   * @param {Array<Object>} allVideos - All processed videos
   */
  async completeBackup(allVideos) {
    this.backupProgress.endTime = new Date();
    this.backupProgress.total = allVideos.length;
    
    console.log(`Backup complete: ${allVideos.length} videos processed`);
    
    // Send completion message
    await this.sendMessage({
      type: window.Constants?.MESSAGE_TYPES?.BACKUP_COMPLETE,
      data: {
        totalVideos: allVideos.length,
        progress: this.backupProgress,
        duration: this.backupProgress.endTime - this.backupProgress.startTime
      }
    });
  }

  /**
   * Pause backup process
   */
  async pauseBackup() {
    if (!this.isBackupActive) {
      return;
    }
    
    this.isBackupActive = false;
    console.log('Backup paused');
    
    await this.sendMessage({
      type: window.Constants?.MESSAGE_TYPES?.BACKUP_STATUS,
      data: {
        paused: true,
        progress: this.backupProgress
      }
    });
  }

  /**
   * Stop backup process
   */
  async stopBackup() {
    this.isBackupActive = false;
    this.paginationHandler.reset();
    
    console.log('Backup stopped');
    
    await this.sendMessage({
      type: window.Constants?.MESSAGE_TYPES?.BACKUP_STATUS,
      data: {
        stopped: true,
        progress: this.backupProgress
      }
    });
  }

  /**
   * Get current backup status
   * @returns {Object} Backup status
   */
  getBackupStatus() {
    return {
      isActive: this.isBackupActive,
      progress: this.backupProgress,
      isLikedVideosPage: this.isLikedVideosPage(),
      totalVideosOnPage: this.videoScraper.getTotalVideoCount(),
      paginationStats: this.paginationHandler.getStats()
    };
  }

  /**
   * Update settings
   * @param {Object} settings - New settings
   */
  async updateSettings(settings) {
    // Update local settings that affect content script behavior
    if (settings.batchSize) {
      this.paginationHandler.batchSize = settings.batchSize;
    }
    
    if (settings.scrollDelay) {
      this.paginationHandler.scrollDelay = settings.scrollDelay;
    }
    
    console.log('Settings updated:', settings);
  }

  /**
   * Send message to background script
   * @param {Object} message - Message to send
   * @returns {Promise<Object>} Response from background script
   */
  async sendMessage(message) {
    try {
      return await chrome.runtime.sendMessage(message);
    } catch (error) {
      console.error('Failed to send message:', error);
      return { success: false, error: error.message };
    }
  }

  /**
   * Check if current page is YouTube liked videos
   * @returns {boolean} Whether on liked videos page
   */
  isLikedVideosPage() {
    return window.Constants?.UTILS?.isLikedVideosPage() || 
           (window.location.pathname.includes('/playlist') && 
            window.location.search.includes('list=LL'));
  }

  /**
   * Get page information
   * @returns {Object} Page information
   */
  getPageInfo() {
    return {
      url: window.location.href,
      isLikedVideosPage: this.isLikedVideosPage(),
      totalVideos: this.videoScraper.getTotalVideoCount(),
      visibleVideos: this.videoScraper.getVideoElements().length,
      scrollInfo: this.paginationHandler.getScrollInfo()
    };
  }

  /**
   * Cleanup when page unloads
   */
  cleanup() {
    this.paginationHandler.removeScrollListener();
    this.isBackupActive = false;
  }
}

// Initialize content script when DOM is ready
let youtubeBackupContent = null;

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', initializeContent);
} else {
  initializeContent();
}

function initializeContent() {
  // Only initialize on YouTube liked videos page
  if (window.location.hostname === 'www.youtube.com' && 
      (window.location.pathname.includes('/playlist') && 
       window.location.search.includes('list=LL'))) {
    youtubeBackupContent = new YouTubeBackupContent();
  }
}

// Handle page navigation (YouTube is a SPA)
let lastUrl = location.href;
new MutationObserver(() => {
  const url = location.href;
  if (url !== lastUrl) {
    lastUrl = url;
    
    // Cleanup previous instance
    if (youtubeBackupContent) {
      youtubeBackupContent.cleanup();
      youtubeBackupContent = null;
    }
    
    // Initialize new instance if on liked videos page
    setTimeout(initializeContent, 1000);
  }
}).observe(document, { subtree: true, childList: true });

// Cleanup on page unload
window.addEventListener('beforeunload', () => {
  if (youtubeBackupContent) {
    youtubeBackupContent.cleanup();
  }
});
