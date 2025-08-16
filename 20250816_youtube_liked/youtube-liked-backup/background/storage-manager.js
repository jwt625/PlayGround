/**
 * Storage Manager for YouTube Liked Videos Backup Extension
 * Handles Chrome storage API and quota management
 */

class StorageManager {
  constructor() {
    this.isIndexedDBAvailable = false;
    this.db = null;
    this.initIndexedDB();
  }

  /**
   * Initialize IndexedDB for large datasets
   */
  async initIndexedDB() {
    try {
      const request = indexedDB.open('YouTubeLikedBackup', 1);
      
      request.onerror = () => {
        console.warn('IndexedDB not available, using Chrome storage only');
      };
      
      request.onsuccess = (event) => {
        this.db = event.target.result;
        this.isIndexedDBAvailable = true;
      };
      
      request.onupgradeneeded = (event) => {
        const db = event.target.result;
        
        // Videos store
        if (!db.objectStoreNames.contains('videos')) {
          const videosStore = db.createObjectStore('videos', { keyPath: 'videoId' });
          videosStore.createIndex('channelId', 'channelId', { unique: false });
          videosStore.createIndex('likedDate', 'likedDate', { unique: false });
          videosStore.createIndex('backupStatus', 'backupStatus', { unique: false });
        }
        
        // Sessions store
        if (!db.objectStoreNames.contains('sessions')) {
          const sessionsStore = db.createObjectStore('sessions', { keyPath: 'sessionId' });
          sessionsStore.createIndex('startTime', 'startTime', { unique: false });
        }
        
        // Exports store
        if (!db.objectStoreNames.contains('exports')) {
          const exportsStore = db.createObjectStore('exports', { keyPath: 'exportId' });
          exportsStore.createIndex('createdDate', 'createdDate', { unique: false });
          exportsStore.createIndex('format', 'format', { unique: false });
        }
      };
    } catch (error) {
      console.warn('Failed to initialize IndexedDB:', error);
    }
  }

  /**
   * Save a video record
   * @param {Object} video - VideoRecord object
   * @returns {Promise<boolean>} Success status
   */
  async saveVideo(video) {
    try {
      // Validate video record
      const validation = window.DataSchemas?.validateVideoRecord(video);
      if (!validation?.isValid) {
        throw new Error(`Invalid video record: ${validation?.errors?.join(', ')}`);
      }

      const key = `video_${video.videoId}`;
      
      // Try Chrome storage first
      try {
        await chrome.storage.local.set({ [key]: video });
        return true;
      } catch (error) {
        if (error.message.includes('QUOTA_EXCEEDED')) {
          // Fallback to IndexedDB
          return await this.saveToIndexedDB('videos', video);
        }
        throw error;
      }
    } catch (error) {
      console.error('Failed to save video:', error);
      return false;
    }
  }

  /**
   * Get a video record
   * @param {string} videoId - Video ID
   * @returns {Promise<Object|null>} VideoRecord or null
   */
  async getVideo(videoId) {
    try {
      const key = `video_${videoId}`;
      
      // Try Chrome storage first
      const result = await chrome.storage.local.get(key);
      if (result[key]) {
        return result[key];
      }
      
      // Try IndexedDB
      return await this.getFromIndexedDB('videos', videoId);
    } catch (error) {
      console.error('Failed to get video:', error);
      return null;
    }
  }

  /**
   * Get multiple videos
   * @param {Array<string>} videoIds - Array of video IDs
   * @returns {Promise<Array<Object>>} Array of VideoRecords
   */
  async getVideos(videoIds) {
    const videos = [];
    
    // Batch get from Chrome storage
    const keys = videoIds.map(id => `video_${id}`);
    try {
      const result = await chrome.storage.local.get(keys);
      
      for (const key of keys) {
        if (result[key]) {
          videos.push(result[key]);
        }
      }
    } catch (error) {
      console.error('Failed to batch get videos from Chrome storage:', error);
    }
    
    // Get remaining from IndexedDB
    const foundVideoIds = videos.map(v => v.videoId);
    const missingIds = videoIds.filter(id => !foundVideoIds.includes(id));
    
    for (const videoId of missingIds) {
      const video = await this.getFromIndexedDB('videos', videoId);
      if (video) {
        videos.push(video);
      }
    }
    
    return videos;
  }

  /**
   * Get all videos with optional filtering
   * @param {Object} filter - Filter criteria
   * @returns {Promise<Array<Object>>} Array of VideoRecords
   */
  async getAllVideos(filter = {}) {
    const videos = [];
    
    // Get from Chrome storage
    try {
      const result = await chrome.storage.local.get(null);
      for (const [key, value] of Object.entries(result)) {
        if (key.startsWith('video_') && this.matchesFilter(value, filter)) {
          videos.push(value);
        }
      }
    } catch (error) {
      console.error('Failed to get videos from Chrome storage:', error);
    }
    
    // Get from IndexedDB
    const indexedDBVideos = await this.getAllFromIndexedDB('videos', filter);
    videos.push(...indexedDBVideos);
    
    return videos;
  }

  /**
   * Save backup session
   * @param {Object} session - BackupSession object
   * @returns {Promise<boolean>} Success status
   */
  async saveSession(session) {
    try {
      const key = `session_${session.sessionId}`;
      await chrome.storage.local.set({ [key]: session });
      return true;
    } catch (error) {
      console.error('Failed to save session:', error);
      return false;
    }
  }

  /**
   * Get backup session
   * @param {string} sessionId - Session ID
   * @returns {Promise<Object|null>} BackupSession or null
   */
  async getSession(sessionId) {
    try {
      const key = `session_${sessionId}`;
      const result = await chrome.storage.local.get(key);
      return result[key] || null;
    } catch (error) {
      console.error('Failed to get session:', error);
      return null;
    }
  }

  /**
   * Save extension settings
   * @param {Object} settings - BackupSettings object
   * @returns {Promise<boolean>} Success status
   */
  async saveSettings(settings) {
    try {
      await chrome.storage.local.set({ settings: settings });
      return true;
    } catch (error) {
      console.error('Failed to save settings:', error);
      return false;
    }
  }

  /**
   * Get extension settings
   * @returns {Promise<Object>} BackupSettings object
   */
  async getSettings() {
    try {
      const result = await chrome.storage.local.get('settings');
      return result.settings || window.DataSchemas?.createBackupSettings();
    } catch (error) {
      console.error('Failed to get settings:', error);
      return window.DataSchemas?.createBackupSettings();
    }
  }

  /**
   * Save extension state
   * @param {Object} state - Extension state
   * @returns {Promise<boolean>} Success status
   */
  async saveState(state) {
    try {
      await chrome.storage.local.set({ state: state });
      return true;
    } catch (error) {
      console.error('Failed to save state:', error);
      return false;
    }
  }

  /**
   * Get extension state
   * @returns {Promise<Object>} Extension state
   */
  async getState() {
    try {
      const result = await chrome.storage.local.get('state');
      return result.state || {
        lastBackupDate: null,
        totalVideosBackedUp: 0,
        currentSession: null,
        extensionState: 'idle'
      };
    } catch (error) {
      console.error('Failed to get state:', error);
      return {
        lastBackupDate: null,
        totalVideosBackedUp: 0,
        currentSession: null,
        extensionState: 'idle'
      };
    }
  }

  /**
   * Get storage usage statistics
   * @returns {Promise<Object>} Storage usage info
   */
  async getStorageUsage() {
    try {
      const usage = await chrome.storage.local.getBytesInUse();
      const quota = chrome.storage.local.QUOTA_BYTES || 5242880; // 5MB default
      
      return {
        usage: usage,
        quota: quota,
        percentage: (usage / quota) * 100,
        available: quota - usage,
        isNearLimit: (usage / quota) > 0.8
      };
    } catch (error) {
      console.error('Failed to get storage usage:', error);
      return {
        usage: 0,
        quota: 5242880,
        percentage: 0,
        available: 5242880,
        isNearLimit: false
      };
    }
  }

  /**
   * Clear all extension data
   * @returns {Promise<boolean>} Success status
   */
  async clearAllData() {
    try {
      await chrome.storage.local.clear();
      
      if (this.isIndexedDBAvailable && this.db) {
        const transaction = this.db.transaction(['videos', 'sessions', 'exports'], 'readwrite');
        await Promise.all([
          transaction.objectStore('videos').clear(),
          transaction.objectStore('sessions').clear(),
          transaction.objectStore('exports').clear()
        ]);
      }
      
      return true;
    } catch (error) {
      console.error('Failed to clear data:', error);
      return false;
    }
  }

  // IndexedDB helper methods
  async saveToIndexedDB(storeName, data) {
    if (!this.isIndexedDBAvailable || !this.db) return false;
    
    try {
      const transaction = this.db.transaction([storeName], 'readwrite');
      const store = transaction.objectStore(storeName);
      await store.put(data);
      return true;
    } catch (error) {
      console.error('Failed to save to IndexedDB:', error);
      return false;
    }
  }

  async getFromIndexedDB(storeName, key) {
    if (!this.isIndexedDBAvailable || !this.db) return null;
    
    try {
      const transaction = this.db.transaction([storeName], 'readonly');
      const store = transaction.objectStore(storeName);
      const request = store.get(key);
      
      return new Promise((resolve, reject) => {
        request.onsuccess = () => resolve(request.result || null);
        request.onerror = () => reject(request.error);
      });
    } catch (error) {
      console.error('Failed to get from IndexedDB:', error);
      return null;
    }
  }

  async getAllFromIndexedDB(storeName, filter = {}) {
    if (!this.isIndexedDBAvailable || !this.db) return [];
    
    try {
      const transaction = this.db.transaction([storeName], 'readonly');
      const store = transaction.objectStore(storeName);
      const request = store.getAll();
      
      return new Promise((resolve, reject) => {
        request.onsuccess = () => {
          const results = request.result || [];
          const filtered = results.filter(item => this.matchesFilter(item, filter));
          resolve(filtered);
        };
        request.onerror = () => reject(request.error);
      });
    } catch (error) {
      console.error('Failed to get all from IndexedDB:', error);
      return [];
    }
  }

  /**
   * Check if item matches filter criteria
   * @param {Object} item - Item to check
   * @param {Object} filter - Filter criteria
   * @returns {boolean} Whether item matches filter
   */
  matchesFilter(item, filter) {
    for (const [key, value] of Object.entries(filter)) {
      if (item[key] !== value) {
        return false;
      }
    }
    return true;
  }
}

// Export for different environments
if (typeof module !== 'undefined' && module.exports) {
  module.exports = StorageManager;
} else {
  window.StorageManager = StorageManager;
}
