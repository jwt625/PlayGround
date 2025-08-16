/**
 * Background Service Worker for YouTube Liked Videos Backup Extension
 * Handles data management and coordination between components
 */

// Import required modules
try {
  importScripts(
    '../utils/data-schemas.js',
    '../utils/constants.js',
    'storage-manager.js',
    'export-manager.js'
  );
  console.log('Background scripts loaded successfully');
} catch (error) {
  console.error('Failed to load background scripts:', error);
}

class YouTubeBackupBackground {
  constructor() {
    this.storageManager = new YouTubeStorageManager();
    this.exportManager = new YouTubeExportManager(this.storageManager);
    this.currentSession = null;
    this.extensionState = 'idle';
    this.backupStats = {
      totalVideos: 0,
      sessionsCount: 0,
      lastBackupDate: null
    };

    this.init();
  }

  /**
   * Initialize background service worker
   */
  async init() {
    console.log('YouTube Liked Videos Backup Extension background script loaded');
    
    // Set up message listeners
    this.setupMessageListeners();
    
    // Set up extension lifecycle listeners
    this.setupExtensionListeners();
    
    // Load initial state
    await this.loadInitialState();
  }

  /**
   * Set up message listeners
   */
  setupMessageListeners() {
    chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
      this.handleMessage(message, sender, sendResponse);
      return true; // Keep message channel open for async responses
    });
  }

  /**
   * Set up extension lifecycle listeners
   */
  setupExtensionListeners() {
    // Handle extension installation
    chrome.runtime.onInstalled.addListener((details) => {
      this.handleInstallation(details);
    });
    
    // Handle extension startup
    chrome.runtime.onStartup.addListener(() => {
      this.handleStartup();
    });
  }

  /**
   * Handle messages from content scripts and popup
   * @param {Object} message - Message object
   * @param {Object} sender - Message sender
   * @param {Function} sendResponse - Response callback
   */
  async handleMessage(message, sender, sendResponse) {
    try {
      switch (message.type) {
        case 'startBackup':
          await this.startBackupSession(message.data, sender.tab);
          sendResponse({ success: true });
          break;
          
        case 'videoScraped':
          await this.handleVideosBatch(message.data);
          sendResponse({ success: true });
          break;
          
        case 'batchComplete':
          await this.handleBatchComplete(message.data);
          sendResponse({ success: true });
          break;
          
        case 'backupComplete':
          await this.handleBackupComplete(message.data);
          sendResponse({ success: true });
          break;
          
        case 'errorOccurred':
          await this.handleError(message.data);
          sendResponse({ success: true });
          break;
          
        case 'getStatus':
          const status = await this.getExtensionStatus();
          sendResponse(status);
          break;
          
        case 'getStats':
          const stats = await this.getBackupStats();
          sendResponse(stats);
          break;
          
        case 'exportData':
          const exportResult = await this.exportManager.exportData(message.data);
          sendResponse(exportResult);
          break;
          
        case 'saveSettings':
          await this.saveSettings(message.data);
          sendResponse({ success: true });
          break;
          
        case 'loadSettings':
          const settings = await this.loadSettings();
          sendResponse(settings);
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
   * Start a new backup session
   * @param {Object} options - Backup options
   * @param {Object} tab - Tab information
   */
  async startBackupSession(options = {}, tab) {
    if (this.currentSession) {
      throw new Error('Backup session already in progress');
    }
    
    // Create new session
    this.currentSession = DataSchemas.createBackupSession({
      settings: await this.storageManager.getSettings()
    });
    
    // Save session
    await this.storageManager.saveSession(this.currentSession);
    
    // Update extension state
    this.extensionState = 'scraping';
    await this.updateExtensionState();
    
    console.log('Backup session started:', this.currentSession.sessionId);
  }

  /**
   * Handle batch of scraped videos
   * @param {Object} data - Batch data
   */
  async handleVideosBatch(data) {
    if (!this.currentSession) {
      throw new Error('No active backup session');
    }
    
    const { videos, batchNumber } = data;
    
    // Save videos to storage
    const savedCount = await this.saveVideos(videos);
    
    // Update session progress
    this.currentSession.videosProcessed += savedCount;
    await this.storageManager.saveSession(this.currentSession);
    
    console.log(`Batch ${batchNumber} processed: ${savedCount}/${videos.length} videos saved`);
  }

  /**
   * Handle batch completion
   * @param {Object} data - Batch completion data
   */
  async handleBatchComplete(data) {
    if (!this.currentSession) return;
    
    const { batchNumber, videosInBatch, totalProcessed } = data;
    
    // Update backup stats
    this.backupStats.totalVideos = totalProcessed;
    
    // Notify popup of progress
    await this.notifyProgress({
      batchNumber,
      videosInBatch,
      totalProcessed,
      sessionId: this.currentSession.sessionId
    });
  }

  /**
   * Handle backup completion
   * @param {Object} data - Completion data
   */
  async handleBackupComplete(data) {
    if (!this.currentSession) return;
    
    const { totalVideos, duration } = data;
    
    // Update session
    this.currentSession.endTime = new Date();
    this.currentSession.videosProcessed = totalVideos;
    await this.storageManager.saveSession(this.currentSession);
    
    // Update extension state
    this.extensionState = 'idle';
    this.backupStats.lastBackupDate = new Date();
    this.backupStats.sessionsCount++;
    await this.updateExtensionState();
    
    console.log(`Backup completed: ${totalVideos} videos in ${duration}ms`);
    
    // Clear current session
    this.currentSession = null;
  }

  /**
   * Handle errors during backup
   * @param {Object} data - Error data
   */
  async handleError(data) {
    if (this.currentSession) {
      this.currentSession.errors.push(DataSchemas.createErrorLog(
        data.type || 'UNKNOWN_ERROR',
        data.message || 'Unknown error occurred',
        data.context || {}
      ));
      
      await this.storageManager.saveSession(this.currentSession);
    }
    
    console.error('Backup error:', data);
  }

  /**
   * Save videos to storage
   * @param {Array<Object>} videos - Videos to save
   * @returns {Promise<number>} Number of successfully saved videos
   */
  async saveVideos(videos) {
    let savedCount = 0;
    
    for (const video of videos) {
      try {
        const success = await this.storageManager.saveVideo(video);
        if (success) {
          savedCount++;
        }
      } catch (error) {
        console.error('Failed to save video:', video.videoId, error);
      }
    }
    
    return savedCount;
  }

  /**
   * Get extension status
   * @returns {Promise<Object>} Extension status
   */
  async getExtensionStatus() {
    const state = await this.storageManager.getState();
    const storageUsage = await this.storageManager.getStorageUsage();
    
    return {
      extensionState: this.extensionState,
      currentSession: this.currentSession,
      backupStats: this.backupStats,
      storageUsage: storageUsage,
      state: state
    };
  }

  /**
   * Get backup statistics
   * @returns {Promise<Object>} Backup statistics
   */
  async getBackupStats() {
    const allVideos = await this.storageManager.getAllVideos();
    const storageUsage = await this.storageManager.getStorageUsage();
    
    return {
      totalVideos: allVideos.length,
      totalSessions: this.backupStats.sessionsCount,
      lastBackupDate: this.backupStats.lastBackupDate,
      storageUsage: storageUsage,
      videosByStatus: this.groupVideosByStatus(allVideos),
      videosByChannel: this.groupVideosByChannel(allVideos)
    };
  }

  /**
   * Export data
   * @param {Object} options - Export options
   */
  async exportData(options = {}) {
    const { format = 'json', includeAll = true } = options;
    
    // Get videos to export
    const videos = await this.storageManager.getAllVideos();
    
    // Create export data
    const exportData = {
      metadata: {
        exportDate: new Date(),
        totalVideos: videos.length,
        extensionVersion: chrome.runtime.getManifest().version
      },
      videos: videos
    };
    
    // Generate filename
    const timestamp = new Date().toISOString().split('T')[0];
    const filename = `youtube-liked-backup-${timestamp}.${format}`;
    
    // Create and download file
    let blob;
    if (format === 'json') {
      blob = new Blob([JSON.stringify(exportData, null, 2)], { type: 'application/json' });
    } else if (format === 'csv') {
      const csv = this.convertToCSV(videos);
      blob = new Blob([csv], { type: 'text/csv' });
    }
    
    // Use Chrome downloads API
    const url = URL.createObjectURL(blob);
    await chrome.downloads.download({
      url: url,
      filename: filename,
      saveAs: true
    });
    
    console.log(`Data exported: ${filename}`);
  }

  /**
   * Save settings
   * @param {Object} settings - Settings to save
   */
  async saveSettings(settings) {
    await this.storageManager.saveSettings(settings);
    console.log('Settings saved:', settings);
  }

  /**
   * Load settings
   * @returns {Promise<Object>} Current settings
   */
  async loadSettings() {
    return await this.storageManager.getSettings();
  }

  /**
   * Load initial state
   */
  async loadInitialState() {
    const state = await this.storageManager.getState();
    this.backupStats = {
      totalVideos: state.totalVideosBackedUp || 0,
      sessionsCount: 0, // Will be calculated from stored sessions
      lastBackupDate: state.lastBackupDate ? new Date(state.lastBackupDate) : null
    };
    
    this.extensionState = state.extensionState || 'idle';
  }

  /**
   * Update extension state
   */
  async updateExtensionState() {
    const state = {
      lastBackupDate: this.backupStats.lastBackupDate,
      totalVideosBackedUp: this.backupStats.totalVideos,
      currentSession: this.currentSession?.sessionId || null,
      extensionState: this.extensionState
    };
    
    await this.storageManager.saveState(state);
  }

  /**
   * Notify progress to popup
   * @param {Object} progress - Progress data
   */
  async notifyProgress(progress) {
    // Send message to popup if it's open
    try {
      await chrome.runtime.sendMessage({
        type: 'progressUpdate',
        data: progress
      });
    } catch (error) {
      // Popup might not be open, ignore error
    }
  }

  /**
   * Handle extension installation
   * @param {Object} details - Installation details
   */
  async handleInstallation(details) {
    if (details.reason === 'install') {
      console.log('Extension installed');
      
      // Initialize default settings
      const defaultSettings = DataSchemas.createBackupSettings();
      await this.storageManager.saveSettings(defaultSettings);
      
      // Initialize state
      await this.updateExtensionState();
    }
  }

  /**
   * Handle extension startup
   */
  async handleStartup() {
    console.log('Extension started');
    await this.loadInitialState();
  }

  /**
   * Group videos by status
   * @param {Array<Object>} videos - Videos to group
   * @returns {Object} Videos grouped by status
   */
  groupVideosByStatus(videos) {
    const groups = {};
    videos.forEach(video => {
      const status = video.backupStatus || 'unknown';
      groups[status] = (groups[status] || 0) + 1;
    });
    return groups;
  }

  /**
   * Group videos by channel
   * @param {Array<Object>} videos - Videos to group
   * @returns {Object} Videos grouped by channel
   */
  groupVideosByChannel(videos) {
    const groups = {};
    videos.forEach(video => {
      const channel = video.channelName || 'Unknown';
      groups[channel] = (groups[channel] || 0) + 1;
    });
    return groups;
  }

  /**
   * Convert videos to CSV format
   * @param {Array<Object>} videos - Videos to convert
   * @returns {string} CSV string
   */
  convertToCSV(videos) {
    if (videos.length === 0) return '';
    
    const headers = [
      'videoId', 'title', 'channelName', 'duration', 'durationFormatted',
      'viewCount', 'uploadDate', 'likedDate', 'url', 'channelUrl'
    ];
    
    const csvRows = [headers.join(',')];
    
    videos.forEach(video => {
      const row = headers.map(header => {
        let value = video[header] || '';
        if (typeof value === 'string') {
          value = `"${value.replace(/"/g, '""')}"`;
        }
        return value;
      });
      csvRows.push(row.join(','));
    });
    
    return csvRows.join('\n');
  }
}

// Initialize background service worker
try {
  const youtubeBackupBackground = new YouTubeBackupBackground();
  console.log('YouTube Backup Background initialized successfully');
} catch (error) {
  console.error('Failed to initialize background service worker:', error);
}
