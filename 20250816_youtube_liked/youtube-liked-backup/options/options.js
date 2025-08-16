/**
 * Options Page Script for YouTube Liked Videos Backup Extension
 * Handles settings configuration and data management
 */

class YouTubeBackupOptions {
  constructor() {
    this.settings = null;
    this.stats = null;
    this.pendingAction = null;
    
    this.init();
  }

  /**
   * Initialize options page
   */
  async init() {
    console.log('YouTube Backup Options page initialized');
    
    // Set up event listeners
    this.setupEventListeners();
    
    // Load current settings and stats
    await this.loadData();
    
    // Update UI
    this.updateUI();
  }

  /**
   * Set up event listeners for all UI elements
   */
  setupEventListeners() {
    // Range inputs
    document.getElementById('verificationThreshold').addEventListener('input', (e) => {
      document.getElementById('verificationThresholdValue').textContent = e.target.value;
    });
    
    // Save button
    document.getElementById('saveBtn').addEventListener('click', () => this.saveSettings());
    
    // Reset button
    document.getElementById('resetBtn').addEventListener('click', () => this.resetSettings());
    
    // Data management buttons
    document.getElementById('exportAllBtn').addEventListener('click', () => this.exportAllData());
    document.getElementById('viewDataBtn').addEventListener('click', () => this.viewData());
    document.getElementById('clearDataBtn').addEventListener('click', () => this.confirmClearData());
    
    // Footer links
    document.getElementById('helpLink').addEventListener('click', (e) => {
      e.preventDefault();
      this.openHelp();
    });
    
    document.getElementById('githubLink').addEventListener('click', (e) => {
      e.preventDefault();
      this.openGitHub();
    });
    
    document.getElementById('issuesLink').addEventListener('click', (e) => {
      e.preventDefault();
      this.openIssues();
    });
    
    // Modal buttons
    document.getElementById('modalCancelBtn').addEventListener('click', () => this.hideModal());
    document.getElementById('modalConfirmBtn').addEventListener('click', () => this.confirmModalAction());
  }

  /**
   * Load current settings and statistics
   */
  async loadData() {
    try {
      // Load settings
      this.settings = await this.sendMessage({ type: 'loadSettings' });
      
      // Load statistics
      this.stats = await this.sendMessage({ type: 'getStats' });
      
      console.log('Loaded settings:', this.settings);
      console.log('Loaded stats:', this.stats);
      
    } catch (error) {
      console.error('Failed to load data:', error);
      this.showStatus('Failed to load settings', 'error');
    }
  }

  /**
   * Update UI with current settings and stats
   */
  updateUI() {
    if (!this.settings) return;
    
    // Update backup settings
    document.getElementById('autoRemoveAfterBackup').checked = this.settings.autoRemoveAfterBackup || false;
    document.getElementById('verificationThreshold').value = this.settings.verificationThreshold || 80;
    document.getElementById('verificationThresholdValue').textContent = this.settings.verificationThreshold || 80;
    document.getElementById('removalRateLimit').value = this.settings.removalRateLimit || 20;
    document.getElementById('batchSize').value = this.settings.batchSize || 50;
    
    // Update export settings
    document.getElementById('exportFormat').value = this.settings.exportFormat || 'json';
    document.getElementById('includeDescriptions').checked = this.settings.includeDescriptions !== false;
    document.getElementById('downloadThumbnails').checked = this.settings.downloadThumbnails || false;
    
    // Update performance settings
    document.getElementById('scrollDelay').value = this.settings.scrollDelay || 2000;
    document.getElementById('navigationDelay').value = this.settings.navigationDelay || 5000;
    document.getElementById('maxRetries').value = this.settings.maxRetries || 3;
    
    // Update advanced settings
    document.getElementById('enableDebugLogging').checked = this.settings.enableDebugLogging || false;
    document.getElementById('enableNetworkInterception').checked = this.settings.enableNetworkInterception !== false;
    
    // Update statistics
    this.updateStats();
  }

  /**
   * Update statistics display
   */
  updateStats() {
    if (!this.stats) return;
    
    // Total videos
    document.getElementById('totalVideos').textContent = this.stats.totalVideos || 0;
    
    // Storage usage
    const storageUsage = this.stats.storageUsage?.percentage || 0;
    document.getElementById('storageUsed').textContent = `${Math.round(storageUsage)}%`;
    
    // Last backup date
    const lastBackupEl = document.getElementById('lastBackup');
    if (this.stats.lastBackupDate) {
      const date = new Date(this.stats.lastBackupDate);
      lastBackupEl.textContent = date.toLocaleDateString();
    } else {
      lastBackupEl.textContent = 'Never';
    }
  }

  /**
   * Collect settings from UI
   * @returns {Object} Settings object
   */
  collectSettings() {
    return {
      // Backup settings
      autoRemoveAfterBackup: document.getElementById('autoRemoveAfterBackup').checked,
      verificationThreshold: parseInt(document.getElementById('verificationThreshold').value, 10),
      removalRateLimit: parseInt(document.getElementById('removalRateLimit').value, 10),
      batchSize: parseInt(document.getElementById('batchSize').value, 10),
      
      // Export settings
      exportFormat: document.getElementById('exportFormat').value,
      includeDescriptions: document.getElementById('includeDescriptions').checked,
      downloadThumbnails: document.getElementById('downloadThumbnails').checked,
      
      // Performance settings
      scrollDelay: parseInt(document.getElementById('scrollDelay').value, 10),
      navigationDelay: parseInt(document.getElementById('navigationDelay').value, 10),
      maxRetries: parseInt(document.getElementById('maxRetries').value, 10),
      
      // Advanced settings
      enableDebugLogging: document.getElementById('enableDebugLogging').checked,
      enableNetworkInterception: document.getElementById('enableNetworkInterception').checked
    };
  }

  /**
   * Save settings
   */
  async saveSettings() {
    try {
      const settings = this.collectSettings();
      
      // Validate settings
      const validation = this.validateSettings(settings);
      if (!validation.isValid) {
        this.showStatus(`Invalid settings: ${validation.errors.join(', ')}`, 'error');
        return;
      }
      
      // Save settings
      const response = await this.sendMessage({
        type: 'saveSettings',
        data: settings
      });
      
      if (response && response.success !== false) {
        this.settings = settings;
        this.showStatus('Settings saved successfully', 'success');
      } else {
        throw new Error(response?.error || 'Failed to save settings');
      }
      
    } catch (error) {
      console.error('Failed to save settings:', error);
      this.showStatus('Failed to save settings', 'error');
    }
  }

  /**
   * Reset settings to defaults
   */
  async resetSettings() {
    try {
      // Create default settings
      const defaultSettings = {
        autoRemoveAfterBackup: false,
        verificationThreshold: 80,
        removalRateLimit: 20,
        batchSize: 50,
        exportFormat: 'json',
        includeDescriptions: true,
        downloadThumbnails: false,
        scrollDelay: 2000,
        navigationDelay: 5000,
        maxRetries: 3,
        enableDebugLogging: false,
        enableNetworkInterception: true
      };
      
      // Save default settings
      const response = await this.sendMessage({
        type: 'saveSettings',
        data: defaultSettings
      });
      
      if (response && response.success !== false) {
        this.settings = defaultSettings;
        this.updateUI();
        this.showStatus('Settings reset to defaults', 'success');
      } else {
        throw new Error(response?.error || 'Failed to reset settings');
      }
      
    } catch (error) {
      console.error('Failed to reset settings:', error);
      this.showStatus('Failed to reset settings', 'error');
    }
  }

  /**
   * Export all data
   */
  async exportAllData() {
    try {
      const response = await this.sendMessage({
        type: 'exportData',
        data: { format: 'both' }
      });
      
      if (response && response.success !== false) {
        this.showStatus('Export initiated successfully', 'success');
      } else {
        throw new Error(response?.error || 'Export failed');
      }
      
    } catch (error) {
      console.error('Failed to export data:', error);
      this.showStatus('Failed to export data', 'error');
    }
  }

  /**
   * View data (open data viewer)
   */
  viewData() {
    // For now, just show stats
    const message = `
      Total Videos: ${this.stats?.totalVideos || 0}
      Storage Used: ${Math.round(this.stats?.storageUsage?.percentage || 0)}%
      Last Backup: ${this.stats?.lastBackupDate ? new Date(this.stats.lastBackupDate).toLocaleDateString() : 'Never'}
    `;
    
    alert(message);
  }

  /**
   * Confirm clear data action
   */
  confirmClearData() {
    this.pendingAction = 'clearData';
    this.showModal(
      'Clear All Data',
      'Are you sure you want to clear all backed up data? This action cannot be undone.',
      'Clear Data'
    );
  }

  /**
   * Clear all data
   */
  async clearData() {
    try {
      // Note: This would need to be implemented in the background script
      this.showStatus('Clear data functionality not yet implemented', 'error');
      
    } catch (error) {
      console.error('Failed to clear data:', error);
      this.showStatus('Failed to clear data', 'error');
    }
  }

  /**
   * Validate settings
   * @param {Object} settings - Settings to validate
   * @returns {Object} Validation result
   */
  validateSettings(settings) {
    const errors = [];
    
    if (settings.verificationThreshold < 0 || settings.verificationThreshold > 100) {
      errors.push('Verification threshold must be between 0 and 100');
    }
    
    if (settings.removalRateLimit < 1 || settings.removalRateLimit > 60) {
      errors.push('Removal rate limit must be between 1 and 60');
    }
    
    if (settings.batchSize < 10 || settings.batchSize > 200) {
      errors.push('Batch size must be between 10 and 200');
    }
    
    if (settings.scrollDelay < 500 || settings.scrollDelay > 10000) {
      errors.push('Scroll delay must be between 500 and 10000 milliseconds');
    }
    
    if (settings.navigationDelay < 1000 || settings.navigationDelay > 30000) {
      errors.push('Navigation delay must be between 1000 and 30000 milliseconds');
    }
    
    if (settings.maxRetries < 1 || settings.maxRetries > 10) {
      errors.push('Max retries must be between 1 and 10');
    }
    
    return {
      isValid: errors.length === 0,
      errors: errors
    };
  }

  /**
   * Show modal dialog
   * @param {string} title - Modal title
   * @param {string} text - Modal text
   * @param {string} confirmText - Confirm button text
   */
  showModal(title, text, confirmText = 'Confirm') {
    document.getElementById('modalTitle').textContent = title;
    document.getElementById('modalText').textContent = text;
    document.getElementById('modalConfirmBtn').textContent = confirmText;
    document.getElementById('modalOverlay').style.display = 'flex';
  }

  /**
   * Hide modal dialog
   */
  hideModal() {
    document.getElementById('modalOverlay').style.display = 'none';
    this.pendingAction = null;
  }

  /**
   * Confirm modal action
   */
  async confirmModalAction() {
    this.hideModal();
    
    if (this.pendingAction === 'clearData') {
      await this.clearData();
    }
    
    this.pendingAction = null;
  }

  /**
   * Show status message
   * @param {string} message - Status message
   * @param {string} type - Message type ('success' or 'error')
   */
  showStatus(message, type = 'success') {
    const statusMessage = document.getElementById('statusMessage');
    const statusIcon = document.getElementById('statusIcon');
    const statusText = document.getElementById('statusText');
    
    statusMessage.className = `status-message ${type}`;
    statusIcon.textContent = type === 'success' ? '✅' : '❌';
    statusText.textContent = message;
    
    statusMessage.style.display = 'flex';
    
    // Hide after 3 seconds
    setTimeout(() => {
      statusMessage.style.display = 'none';
    }, 3000);
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
   * Open GitHub repository
   */
  openGitHub() {
    chrome.tabs.create({
      url: 'https://github.com/jwt625/PlayGround'
    });
  }

  /**
   * Open issues page
   */
  openIssues() {
    chrome.tabs.create({
      url: 'https://github.com/jwt625/PlayGround/issues'
    });
  }

  /**
   * Send message to background script
   * @param {Object} message - Message to send
   * @returns {Promise<Object>} Response from background script
   */
  async sendMessage(message) {
    return new Promise((resolve, reject) => {
      chrome.runtime.sendMessage(message, (response) => {
        if (chrome.runtime.lastError) {
          reject(new Error(chrome.runtime.lastError.message));
        } else {
          resolve(response);
        }
      });
    });
  }
}

// Initialize options page when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
  new YouTubeBackupOptions();
});
