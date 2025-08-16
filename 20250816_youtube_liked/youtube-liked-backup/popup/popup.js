/**
 * Popup Script for YouTube Liked Videos Backup Extension
 * Handles popup UI interactions and communication with background script
 */

class YouTubeBackupPopup {
  constructor() {
    this.isBackupActive = false;
    this.currentProgress = null;
    this.extensionStatus = null;
    this.refreshInterval = null;
    
    this.init();
  }

  /**
   * Initialize popup
   */
  async init() {
    console.log('YouTube Backup Popup initialized');
    
    // Set up event listeners
    this.setupEventListeners();
    
    // Load initial data
    await this.loadInitialData();
    
    // Start periodic updates
    this.startPeriodicUpdates();
  }

  /**
   * Set up event listeners for UI elements
   */
  setupEventListeners() {
    // Backup control buttons
    document.getElementById('startBackupBtn').addEventListener('click', () => this.startBackup());
    document.getElementById('pauseBackupBtn').addEventListener('click', () => this.pauseBackup());
    document.getElementById('stopBackupBtn').addEventListener('click', () => this.stopBackup());
    
    // Export buttons
    document.getElementById('exportJsonBtn').addEventListener('click', () => this.exportData('json'));
    document.getElementById('exportCsvBtn').addEventListener('click', () => this.exportData('csv'));
    
    // Settings button
    document.getElementById('settingsBtn').addEventListener('click', () => this.openSettings());
    
    // Help link
    document.getElementById('helpLink').addEventListener('click', (e) => {
      e.preventDefault();
      this.openHelp();
    });
  }

  /**
   * Load initial data from background script
   */
  async loadInitialData() {
    try {
      this.showLoading(true);

      // Check current page first (doesn't require background script)
      await this.checkCurrentPage();

      // Try to get extension status with timeout
      try {
        this.extensionStatus = await this.sendMessageWithTimeout({ type: 'getStatus' }, 3000);

        // Get backup statistics
        const stats = await this.sendMessageWithTimeout({ type: 'getStats' }, 3000);

        // Update UI
        this.updateUI();
        this.updateStats(stats);

      } catch (bgError) {
        console.warn('Background script not ready:', bgError);
        // Show basic UI without background data
        this.showBasicUI();
      }

    } catch (error) {
      console.error('Failed to load initial data:', error);
      this.showError('Failed to load extension data');
    } finally {
      this.showLoading(false);
    }
  }

  /**
   * Check if current page is YouTube liked videos
   */
  async checkCurrentPage() {
    try {
      const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });

      if (tab && tab.url) {
        const isLikedVideosPage = tab.url.includes('youtube.com') &&
                                 tab.url.includes('/playlist') &&
                                 tab.url.includes('list=LL');

        this.updatePageStatus(isLikedVideosPage, tab.url);

        if (isLikedVideosPage) {
          // Try to get page info from content script (with timeout)
          try {
            const response = await this.sendMessageToTab(tab.id, { type: 'getStatus' }, 2000);
            if (response && response.totalVideosOnPage) {
              this.updateVideoCount(response.totalVideosOnPage);
            }
          } catch (error) {
            // Content script might not be loaded yet, that's okay
            console.log('Content script not ready, will retry later');
          }
        }
      } else {
        // No tab info available
        this.updatePageStatus(false, '');
      }
    } catch (error) {
      console.error('Failed to check current page:', error);
      this.updatePageStatus(false, '');
    }
  }

  /**
   * Update page status display
   * @param {boolean} isLikedVideosPage - Whether on liked videos page
   * @param {string} url - Current page URL
   */
  updatePageStatus(isLikedVideosPage, url) {
    const pageIcon = document.getElementById('pageIcon');
    const pageText = document.getElementById('pageText');
    const startBtn = document.getElementById('startBackupBtn');
    const videoCountEl = document.getElementById('videoCount');
    
    if (isLikedVideosPage) {
      pageIcon.textContent = '✅';
      pageText.textContent = 'YouTube Liked Videos Page';
      pageText.className = 'text text-success';
      startBtn.disabled = false;
      videoCountEl.style.display = 'flex';
    } else {
      pageIcon.textContent = '📍';
      pageText.textContent = 'Navigate to YouTube Liked Videos';
      pageText.className = 'text';
      startBtn.disabled = true;
      videoCountEl.style.display = 'none';
    }
  }

  /**
   * Update video count display
   * @param {number} count - Number of videos
   */
  updateVideoCount(count) {
    const totalVideosEl = document.getElementById('totalVideos');
    totalVideosEl.textContent = count || 0;
  }

  /**
   * Update UI based on current state
   */
  updateUI() {
    if (!this.extensionStatus) return;
    
    const { extensionState, currentSession } = this.extensionStatus;
    
    // Update status indicator
    this.updateStatusIndicator(extensionState);
    
    // Update backup controls
    this.updateBackupControls(extensionState, currentSession);
    
    // Update progress section
    if (currentSession && extensionState === 'scraping') {
      this.showProgress(true);
      this.isBackupActive = true;
    } else {
      this.showProgress(false);
      this.isBackupActive = false;
    }
  }

  /**
   * Update status indicator
   * @param {string} state - Extension state
   */
  updateStatusIndicator(state) {
    const statusDot = document.getElementById('statusDot');
    const statusText = document.getElementById('statusText');
    
    statusDot.className = 'status-dot';
    
    switch (state) {
      case 'scraping':
        statusDot.classList.add('active');
        statusText.textContent = 'Backing up...';
        break;
      case 'error':
        statusDot.classList.add('error');
        statusText.textContent = 'Error';
        break;
      case 'idle':
      default:
        statusText.textContent = 'Ready';
        break;
    }
  }

  /**
   * Update backup control buttons
   * @param {string} state - Extension state
   * @param {Object} session - Current session
   */
  updateBackupControls(state, session) {
    const startBtn = document.getElementById('startBackupBtn');
    const pauseBtn = document.getElementById('pauseBackupBtn');
    const stopBtn = document.getElementById('stopBackupBtn');
    
    if (state === 'scraping' && session) {
      startBtn.style.display = 'none';
      pauseBtn.style.display = 'block';
      stopBtn.style.display = 'block';
    } else {
      startBtn.style.display = 'block';
      pauseBtn.style.display = 'none';
      stopBtn.style.display = 'none';
    }
  }

  /**
   * Show/hide progress section
   * @param {boolean} show - Whether to show progress
   */
  showProgress(show) {
    const progressSection = document.getElementById('progressSection');
    progressSection.style.display = show ? 'block' : 'none';
  }

  /**
   * Show basic UI when background script is not available
   */
  showBasicUI() {
    // Update status indicator
    const statusDot = document.getElementById('statusDot');
    const statusText = document.getElementById('statusText');
    statusDot.className = 'status-dot warning';
    statusText.textContent = 'Loading...';

    // Show default stats
    document.getElementById('totalBackedUp').textContent = '0';
    document.getElementById('lastBackupDate').textContent = 'Never';
    document.getElementById('storageUsed').textContent = '0%';

    // Disable backup controls initially
    document.getElementById('startBackupBtn').disabled = true;
  }

  /**
   * Update statistics display
   * @param {Object} stats - Backup statistics
   */
  updateStats(stats) {
    if (!stats) return;

    // Total backed up
    document.getElementById('totalBackedUp').textContent = stats.totalVideos || 0;

    // Last backup date
    const lastBackupEl = document.getElementById('lastBackupDate');
    if (stats.lastBackupDate) {
      const date = new Date(stats.lastBackupDate);
      lastBackupEl.textContent = date.toLocaleDateString();
    } else {
      lastBackupEl.textContent = 'Never';
    }

    // Storage usage
    const storageUsedEl = document.getElementById('storageUsed');
    if (stats.storageUsage) {
      storageUsedEl.textContent = `${Math.round(stats.storageUsage.percentage)}%`;
    } else {
      storageUsedEl.textContent = '0%';
    }
  }

  /**
   * Update progress display
   * @param {Object} progress - Progress data
   */
  updateProgress(progress) {
    if (!progress) return;
    
    this.currentProgress = progress;
    
    // Update progress bar
    const progressFill = document.getElementById('progressFill');
    const progressPercentage = document.getElementById('progressPercentage');
    
    let percentage = 0;
    if (progress.total && progress.processed) {
      percentage = Math.round((progress.processed / progress.total) * 100);
    }
    
    progressFill.style.width = `${percentage}%`;
    progressPercentage.textContent = `${percentage}%`;
    
    // Update progress details
    document.getElementById('processedCount').textContent = progress.processed || 0;
    document.getElementById('batchCount').textContent = progress.batch || 0;
    document.getElementById('errorCount').textContent = progress.errors?.length || 0;
  }

  /**
   * Start backup process
   */
  async startBackup() {
    try {
      this.showLoading(true);
      
      // Get current tab
      const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
      
      if (!tab || !tab.url.includes('youtube.com')) {
        throw new Error('Please navigate to YouTube liked videos page');
      }
      
      // Send start backup message to content script
      const response = await chrome.tabs.sendMessage(tab.id, {
        type: 'startBackup',
        data: {}
      });
      
      if (response && response.success) {
        this.isBackupActive = true;
        this.showProgress(true);
        this.updateBackupControls('scraping', { active: true });
      } else {
        throw new Error(response?.error || 'Failed to start backup');
      }
      
    } catch (error) {
      console.error('Failed to start backup:', error);
      this.showError(error.message);
    } finally {
      this.showLoading(false);
    }
  }

  /**
   * Pause backup process
   */
  async pauseBackup() {
    try {
      const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
      
      const response = await chrome.tabs.sendMessage(tab.id, {
        type: 'pauseBackup'
      });
      
      if (response && response.success) {
        this.isBackupActive = false;
        this.updateBackupControls('idle', null);
      }
      
    } catch (error) {
      console.error('Failed to pause backup:', error);
      this.showError('Failed to pause backup');
    }
  }

  /**
   * Stop backup process
   */
  async stopBackup() {
    try {
      const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
      
      const response = await chrome.tabs.sendMessage(tab.id, {
        type: 'stopBackup'
      });
      
      if (response && response.success) {
        this.isBackupActive = false;
        this.showProgress(false);
        this.updateBackupControls('idle', null);
      }
      
    } catch (error) {
      console.error('Failed to stop backup:', error);
      this.showError('Failed to stop backup');
    }
  }

  /**
   * Export data in specified format
   * @param {string} format - Export format ('json' or 'csv')
   */
  async exportData(format) {
    try {
      this.showLoading(true);
      
      const response = await this.sendMessage({
        type: 'exportData',
        data: { format }
      });
      
      if (response && response.success) {
        // Export initiated successfully
        console.log(`Export ${format} initiated`);
      } else {
        throw new Error(response?.error || 'Export failed');
      }
      
    } catch (error) {
      console.error('Failed to export data:', error);
      this.showError(`Failed to export ${format.toUpperCase()}`);
    } finally {
      this.showLoading(false);
    }
  }

  /**
   * Open settings page
   */
  openSettings() {
    chrome.runtime.openOptionsPage();
  }

  /**
   * Open help page
   */
  openHelp() {
    chrome.tabs.create({
      url: 'https://github.com/jwt625/PlayGround/tree/main/20250816_youtube_liked'
    });
  }

  /**
   * Send message to background script with timeout
   * @param {Object} message - Message to send
   * @param {number} timeout - Timeout in milliseconds
   * @returns {Promise<Object>} Response from background script
   */
  async sendMessageWithTimeout(message, timeout = 5000) {
    return new Promise((resolve, reject) => {
      const timeoutId = setTimeout(() => {
        reject(new Error('Message timeout'));
      }, timeout);

      chrome.runtime.sendMessage(message, (response) => {
        clearTimeout(timeoutId);
        if (chrome.runtime.lastError) {
          reject(new Error(chrome.runtime.lastError.message));
        } else {
          resolve(response);
        }
      });
    });
  }

  /**
   * Send message to background script
   * @param {Object} message - Message to send
   * @returns {Promise<Object>} Response from background script
   */
  async sendMessage(message) {
    return this.sendMessageWithTimeout(message, 5000);
  }

  /**
   * Send message to content script in tab
   * @param {number} tabId - Tab ID
   * @param {Object} message - Message to send
   * @param {number} timeout - Timeout in milliseconds
   * @returns {Promise<Object>} Response from content script
   */
  async sendMessageToTab(tabId, message, timeout = 3000) {
    return new Promise((resolve, reject) => {
      const timeoutId = setTimeout(() => {
        reject(new Error('Tab message timeout'));
      }, timeout);

      chrome.tabs.sendMessage(tabId, message, (response) => {
        clearTimeout(timeoutId);
        if (chrome.runtime.lastError) {
          reject(new Error(chrome.runtime.lastError.message));
        } else {
          resolve(response);
        }
      });
    });
  }

  /**
   * Show/hide loading overlay
   * @param {boolean} show - Whether to show loading
   */
  showLoading(show) {
    const loadingOverlay = document.getElementById('loadingOverlay');
    loadingOverlay.style.display = show ? 'flex' : 'none';
  }

  /**
   * Show error message
   * @param {string} message - Error message
   */
  showError(message) {
    const errorSection = document.getElementById('errorSection');
    const errorText = document.getElementById('errorText');
    
    errorText.textContent = message;
    errorSection.style.display = 'block';
    
    // Hide error after 5 seconds
    setTimeout(() => {
      errorSection.style.display = 'none';
    }, 5000);
  }

  /**
   * Start periodic updates
   */
  startPeriodicUpdates() {
    this.refreshInterval = setInterval(async () => {
      if (this.isBackupActive) {
        try {
          this.extensionStatus = await this.sendMessage({ type: 'getStatus' });
          this.updateUI();
        } catch (error) {
          console.error('Failed to refresh status:', error);
        }
      }
    }, 2000); // Update every 2 seconds during backup
  }

  /**
   * Stop periodic updates
   */
  stopPeriodicUpdates() {
    if (this.refreshInterval) {
      clearInterval(this.refreshInterval);
      this.refreshInterval = null;
    }
  }
}

// Initialize popup when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
  new YouTubeBackupPopup();
});
