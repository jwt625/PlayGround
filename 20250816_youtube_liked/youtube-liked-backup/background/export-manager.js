/**
 * Export Manager for YouTube Liked Videos Backup Extension
 * Handles data export functionality using Chrome downloads API
 */

class ExportManager {
  constructor(storageManager) {
    this.storageManager = storageManager;
    this.exportFormats = {
      JSON: 'json',
      CSV: 'csv',
      BOTH: 'both'
    };
  }

  /**
   * Export data in specified format
   * @param {Object} options - Export options
   * @returns {Promise<Object>} Export result
   */
  async exportData(options = {}) {
    const {
      format = 'json',
      includeMetadata = true,
      includeDescriptions = true,
      filter = {},
      filename = null
    } = options;

    try {
      // Get videos to export
      const videos = await this.storageManager.getAllVideos(filter);
      
      if (videos.length === 0) {
        throw new Error('No videos found to export');
      }

      // Prepare export data
      const exportData = await this.prepareExportData(videos, {
        includeMetadata,
        includeDescriptions
      });

      // Generate exports based on format
      const results = [];

      if (format === 'json' || format === 'both') {
        const jsonResult = await this.exportJSON(exportData, filename);
        results.push(jsonResult);
      }

      if (format === 'csv' || format === 'both') {
        const csvResult = await this.exportCSV(exportData.videos, filename);
        results.push(csvResult);
      }

      return {
        success: true,
        exports: results,
        totalVideos: videos.length
      };

    } catch (error) {
      console.error('Export failed:', error);
      return {
        success: false,
        error: error.message
      };
    }
  }

  /**
   * Prepare data for export
   * @param {Array<Object>} videos - Videos to export
   * @param {Object} options - Preparation options
   * @returns {Object} Prepared export data
   */
  async prepareExportData(videos, options = {}) {
    const { includeMetadata = true, includeDescriptions = true } = options;

    // Filter and clean video data
    const cleanedVideos = videos.map(video => this.cleanVideoForExport(video, {
      includeDescriptions
    }));

    const exportData = {
      videos: cleanedVideos
    };

    if (includeMetadata) {
      exportData.metadata = await this.generateExportMetadata(videos);
    }

    return exportData;
  }

  /**
   * Clean video data for export
   * @param {Object} video - Video record
   * @param {Object} options - Cleaning options
   * @returns {Object} Cleaned video record
   */
  cleanVideoForExport(video, options = {}) {
    const { includeDescriptions = true } = options;

    const cleaned = {
      videoId: video.videoId,
      title: video.title,
      channelName: video.channelName,
      channelId: video.channelId,
      channelUrl: video.channelUrl,
      duration: video.duration,
      durationFormatted: video.durationFormatted,
      url: video.url,
      uploadDate: video.uploadDate,
      likedDate: video.likedDate,
      scrapedDate: video.scrapedDate,
      viewCount: video.viewCount,
      likeCount: video.likeCount,
      thumbnails: video.thumbnails,
      backupStatus: video.backupStatus,
      verificationScore: video.verificationScore
    };

    if (includeDescriptions && video.description) {
      cleaned.description = video.description;
    }

    if (video.tags && video.tags.length > 0) {
      cleaned.tags = video.tags;
    }

    if (video.category) {
      cleaned.category = video.category;
    }

    if (video.language) {
      cleaned.language = video.language;
    }

    if (video.playlistPosition) {
      cleaned.playlistPosition = video.playlistPosition;
    }

    return cleaned;
  }

  /**
   * Generate export metadata
   * @param {Array<Object>} videos - Videos being exported
   * @returns {Object} Export metadata
   */
  async generateExportMetadata(videos) {
    const settings = await this.storageManager.getSettings();
    const state = await this.storageManager.getState();

    return {
      exportDate: new Date().toISOString(),
      extensionVersion: chrome.runtime.getManifest().version,
      totalVideos: videos.length,
      exportSettings: {
        format: 'multiple',
        includeDescriptions: true,
        includeMetadata: true
      },
      statistics: {
        videosByStatus: this.groupVideosByStatus(videos),
        videosByChannel: this.groupVideosByChannel(videos),
        dateRange: this.getDateRange(videos),
        averageVerificationScore: this.calculateAverageScore(videos)
      },
      backupInfo: {
        lastBackupDate: state.lastBackupDate,
        totalSessions: state.totalSessions || 0,
        extensionState: state.extensionState
      }
    };
  }

  /**
   * Export data as JSON
   * @param {Object} data - Data to export
   * @param {string} customFilename - Custom filename
   * @returns {Promise<Object>} Export result
   */
  async exportJSON(data, customFilename = null) {
    const timestamp = new Date().toISOString().split('T')[0];
    const filename = customFilename || `youtube-liked-backup-${timestamp}.json`;

    const jsonString = JSON.stringify(data, null, 2);
    const blob = new Blob([jsonString], { type: 'application/json' });
    
    return await this.downloadFile(blob, filename);
  }

  /**
   * Export data as CSV
   * @param {Array<Object>} videos - Videos to export
   * @param {string} customFilename - Custom filename
   * @returns {Promise<Object>} Export result
   */
  async exportCSV(videos, customFilename = null) {
    const timestamp = new Date().toISOString().split('T')[0];
    const filename = customFilename || `youtube-liked-backup-${timestamp}.csv`;

    const csvString = this.convertToCSV(videos);
    const blob = new Blob([csvString], { type: 'text/csv;charset=utf-8;' });
    
    return await this.downloadFile(blob, filename);
  }

  /**
   * Convert videos to CSV format
   * @param {Array<Object>} videos - Videos to convert
   * @returns {string} CSV string
   */
  convertToCSV(videos) {
    if (videos.length === 0) {
      return 'No videos to export';
    }

    // Define CSV headers
    const headers = [
      'videoId',
      'title',
      'channelName',
      'channelId',
      'duration',
      'durationFormatted',
      'viewCount',
      'uploadDate',
      'likedDate',
      'url',
      'channelUrl',
      'verificationScore',
      'backupStatus',
      'thumbnailDefault',
      'thumbnailHigh'
    ];

    // Create CSV rows
    const csvRows = [headers.join(',')];

    videos.forEach(video => {
      const row = headers.map(header => {
        let value = '';

        switch (header) {
          case 'thumbnailDefault':
            value = video.thumbnails?.default || '';
            break;
          case 'thumbnailHigh':
            value = video.thumbnails?.high || '';
            break;
          case 'uploadDate':
          case 'likedDate':
            value = video[header] ? new Date(video[header]).toISOString() : '';
            break;
          default:
            value = video[header] || '';
        }

        // Escape CSV values
        if (typeof value === 'string') {
          // Escape quotes and wrap in quotes if contains comma, quote, or newline
          if (value.includes(',') || value.includes('"') || value.includes('\n')) {
            value = `"${value.replace(/"/g, '""')}"`;
          }
        }

        return value;
      });

      csvRows.push(row.join(','));
    });

    return csvRows.join('\n');
  }

  /**
   * Download file using Chrome downloads API
   * @param {Blob} blob - File blob
   * @param {string} filename - Filename
   * @returns {Promise<Object>} Download result
   */
  async downloadFile(blob, filename) {
    try {
      const url = URL.createObjectURL(blob);
      
      const downloadId = await chrome.downloads.download({
        url: url,
        filename: filename,
        saveAs: true
      });

      // Clean up object URL after a delay
      setTimeout(() => {
        URL.revokeObjectURL(url);
      }, 10000);

      return {
        success: true,
        downloadId: downloadId,
        filename: filename,
        size: blob.size
      };

    } catch (error) {
      console.error('Download failed:', error);
      return {
        success: false,
        error: error.message,
        filename: filename
      };
    }
  }

  /**
   * Group videos by backup status
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
   * @returns {Object} Videos grouped by channel (top 10)
   */
  groupVideosByChannel(videos) {
    const groups = {};
    videos.forEach(video => {
      const channel = video.channelName || 'Unknown';
      groups[channel] = (groups[channel] || 0) + 1;
    });

    // Return top 10 channels
    return Object.entries(groups)
      .sort(([,a], [,b]) => b - a)
      .slice(0, 10)
      .reduce((obj, [channel, count]) => {
        obj[channel] = count;
        return obj;
      }, {});
  }

  /**
   * Get date range of videos
   * @param {Array<Object>} videos - Videos to analyze
   * @returns {Object} Date range information
   */
  getDateRange(videos) {
    const dates = videos
      .map(video => video.likedDate)
      .filter(date => date)
      .map(date => new Date(date))
      .sort((a, b) => a - b);

    if (dates.length === 0) {
      return { earliest: null, latest: null, span: null };
    }

    const earliest = dates[0];
    const latest = dates[dates.length - 1];
    const spanDays = Math.ceil((latest - earliest) / (1000 * 60 * 60 * 24));

    return {
      earliest: earliest.toISOString(),
      latest: latest.toISOString(),
      span: `${spanDays} days`
    };
  }

  /**
   * Calculate average verification score
   * @param {Array<Object>} videos - Videos to analyze
   * @returns {number} Average verification score
   */
  calculateAverageScore(videos) {
    const scores = videos
      .map(video => video.verificationScore)
      .filter(score => typeof score === 'number');

    if (scores.length === 0) return 0;

    const sum = scores.reduce((total, score) => total + score, 0);
    return Math.round(sum / scores.length);
  }

  /**
   * Get export statistics
   * @returns {Promise<Object>} Export statistics
   */
  async getExportStats() {
    const videos = await this.storageManager.getAllVideos();
    
    return {
      totalVideos: videos.length,
      videosByStatus: this.groupVideosByStatus(videos),
      videosByChannel: this.groupVideosByChannel(videos),
      dateRange: this.getDateRange(videos),
      averageScore: this.calculateAverageScore(videos),
      estimatedSizes: {
        json: Math.round(JSON.stringify(videos).length / 1024) + ' KB',
        csv: Math.round(this.convertToCSV(videos).length / 1024) + ' KB'
      }
    };
  }

  /**
   * Validate export options
   * @param {Object} options - Export options to validate
   * @returns {Object} Validation result
   */
  validateExportOptions(options) {
    const errors = [];
    const warnings = [];

    if (options.format && !['json', 'csv', 'both'].includes(options.format)) {
      errors.push('Invalid export format. Must be "json", "csv", or "both"');
    }

    if (options.filename && typeof options.filename !== 'string') {
      errors.push('Filename must be a string');
    }

    if (options.filter && typeof options.filter !== 'object') {
      errors.push('Filter must be an object');
    }

    return {
      isValid: errors.length === 0,
      errors,
      warnings
    };
  }
}

// Export for different environments
if (typeof module !== 'undefined' && module.exports) {
  module.exports = ExportManager;
} else {
  window.ExportManager = ExportManager;
}
