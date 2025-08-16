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
    console.log('YouTube Liked Videos Backup Extension content script loaded on:', window.location.href);

    // Always set up message listeners so popup can communicate
    this.setupMessageListeners();
    console.log('Message listeners set up');

    // Only initialize backup functionality on liked videos page
    if (this.isLikedVideosPage()) {
      console.log('On YouTube liked videos page - initializing backup functionality');

      // Wait for page to fully load
      if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', () => this.onPageReady());
      } else {
        this.onPageReady();
      }
    } else {
      console.log('Not on YouTube liked videos page - backup functionality disabled');
      console.log('Current URL:', window.location.href);
      console.log('Expected URL pattern: youtube.com/playlist?list=LL');
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
    console.log('🔧 Setting up message listeners in content script');
    chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
      console.log('📨 Content script received message:', message);
      console.log('📨 Message sender:', sender);
      this.handleMessage(message, sender, sendResponse);
      return true; // Keep message channel open for async responses
    });
    console.log('✅ Message listeners set up successfully');
  }

  /**
   * Handle messages from background script and popup
   * @param {Object} message - Message object
   * @param {Object} sender - Message sender
   * @param {Function} sendResponse - Response callback
   */
  async handleMessage(message, sender, sendResponse) {
    try {
      console.log('🔧 handleMessage called with:', message.type);
      console.log('🔧 Full message:', message);
      console.log('🔧 Sender:', sender);
      console.log('🔧 Current page check - isLikedVideosPage():', this.isLikedVideosPage());

      // Use direct string comparison as fallback if Constants not available
      const messageTypes = window.Constants?.MESSAGE_TYPES || {
        START_BACKUP: 'startBackup',
        PAUSE_BACKUP: 'pauseBackup',
        STOP_BACKUP: 'stopBackup',
        GET_STATUS: 'getStatus',
        SETTINGS_UPDATED: 'settingsUpdated'
      };

      console.log('🔧 Available message types:', messageTypes);
      console.log('🔧 Checking message type:', message.type);

      switch (message.type) {
        case messageTypes.START_BACKUP:
        case 'startBackup': // Direct fallback
          console.log('🚀 START_BACKUP message received');
          if (!this.isLikedVideosPage()) {
            console.log('❌ Not on liked videos page, sending error response');
            sendResponse({
              success: false,
              error: 'Please navigate to YouTube liked videos page (youtube.com/playlist?list=LL)'
            });
          } else {
            console.log('✅ On liked videos page, starting backup');
            try {
              // First notify background script to start session
              console.log('📤 Notifying background script to start backup session...');
              const sessionResult = await this.sendMessage({
                type: 'startBackup',
                data: message.data
              });

              if (!sessionResult.success) {
                throw new Error(sessionResult.error || 'Failed to start backup session');
              }

              console.log('✅ Background session started, beginning content scraping...');
              // Now start the actual scraping process
              await this.startBackup(message.data);
              console.log('✅ Backup started successfully, sending success response');
              sendResponse({ success: true });
            } catch (error) {
              console.error('❌ Failed to start backup:', error);
              sendResponse({
                success: false,
                error: error.message
              });
            }
          }
          break;

        case messageTypes.PAUSE_BACKUP:
        case 'pauseBackup': // Direct fallback
          console.log('⏸️ PAUSE_BACKUP message received');
          await this.pauseBackup();
          sendResponse({ success: true });
          break;

        case messageTypes.STOP_BACKUP:
        case 'stopBackup': // Direct fallback
          console.log('⏹️ STOP_BACKUP message received');
          await this.stopBackup();
          sendResponse({ success: true });
          break;

        case messageTypes.GET_STATUS:
        case 'getStatus': // Direct fallback
          console.log('📊 GET_STATUS message received');
          const status = this.getBackupStatus();
          console.log('📊 Sending status response:', status);
          sendResponse(status);
          break;

        case messageTypes.SETTINGS_UPDATED:
        case 'settingsUpdated': // Direct fallback
          console.log('⚙️ SETTINGS_UPDATED message received');
          await this.updateSettings(message.data);
          sendResponse({ success: true });
          break;

        default:
          console.warn('❓ Unknown message type:', message.type);
          console.warn('❓ Available types:', Object.values(messageTypes));
          sendResponse({ success: false, error: 'Unknown message type' });
      }
    } catch (error) {
      console.error('❌ Error handling message:', error);
      console.error('❌ Error stack:', error.stack);
      console.error('❌ Message that caused error:', message);
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
    console.log('🧹 YouTubeBackupContent cleanup called');
    if (this.paginationHandler && this.paginationHandler.removeScrollListener) {
      this.paginationHandler.removeScrollListener();
    }
    this.isBackupActive = false;
    console.log('🧹 Cleanup completed');
  }
}

// Initialize content script when DOM is ready
let youtubeBackupContent = null;

console.log('🔧 Content script file loaded, DOM state:', document.readyState);
console.log('🔧 Current URL:', window.location.href);
console.log('🔧 Hostname:', window.location.hostname);

if (document.readyState === 'loading') {
  console.log('🔧 DOM still loading, waiting for DOMContentLoaded');
  document.addEventListener('DOMContentLoaded', initializeContent);
} else {
  console.log('🔧 DOM ready, initializing immediately');
  initializeContent();
}

function initializeContent() {
  console.log('🔧 initializeContent called');
  console.log('🔧 Current hostname:', window.location.hostname);
  console.log('🔧 Current pathname:', window.location.pathname);
  console.log('🔧 Current search:', window.location.search);

  // Always initialize on YouTube pages to handle popup communication
  if (window.location.hostname === 'www.youtube.com') {
    console.log('✅ On YouTube, creating YouTubeBackupContent instance');
    youtubeBackupContent = new YouTubeBackupContent();
    console.log('✅ YouTubeBackupContent instance created:', !!youtubeBackupContent);
  } else {
    console.log('❌ Not on YouTube, skipping initialization');
  }
}

// Handle page navigation (YouTube is a SPA)
let lastUrl = location.href;
console.log('🔧 Setting up navigation observer, initial URL:', lastUrl);

new MutationObserver(() => {
  const url = location.href;
  if (url !== lastUrl) {
    console.log('🔄 URL changed from:', lastUrl, 'to:', url);
    lastUrl = url;

    // Cleanup previous instance
    if (youtubeBackupContent) {
      console.log('🧹 Cleaning up previous content script instance');
      if (youtubeBackupContent.cleanup) {
        youtubeBackupContent.cleanup();
      }
      youtubeBackupContent = null;
    }

    // Initialize new instance
    console.log('🔄 Reinitializing content script after navigation');
    setTimeout(initializeContent, 1000);
  }
}).observe(document, { subtree: true, childList: true });

// Cleanup on page unload
window.addEventListener('beforeunload', () => {
  if (youtubeBackupContent) {
    youtubeBackupContent.cleanup();
  }
});
