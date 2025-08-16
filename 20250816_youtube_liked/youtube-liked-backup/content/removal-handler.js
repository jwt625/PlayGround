/**
 * Removal Handler for YouTube Liked Videos Backup Extension
 * Handles safe removal of videos from liked list
 */

class RemovalHandler {
  constructor() {
    this.isRemoving = false;
    this.removalQueue = [];
    this.removedVideos = [];
    this.failedRemovals = [];
    this.removalDelay = window.Constants?.TIMING?.REMOVAL_DELAY || 3000;
    this.maxRetries = window.Constants?.LIMITS?.MAX_RETRIES || 3;
    this.youtubeAPI = new YouTubeAPI();
  }

  /**
   * Queue videos for removal
   * @param {Array<Object>} videos - Videos to queue for removal
   * @param {Object} options - Removal options
   */
  queueForRemoval(videos, options = {}) {
    const { verifyBeforeRemoval = true, batchSize = 10 } = options;
    
    videos.forEach(video => {
      if (this.shouldQueueVideo(video, verifyBeforeRemoval)) {
        this.removalQueue.push({
          video: video,
          retries: 0,
          queuedAt: new Date(),
          options: options
        });
      }
    });
    
    console.log(`Queued ${videos.length} videos for removal`);
  }

  /**
   * Check if video should be queued for removal
   * @param {Object} video - Video record
   * @param {boolean} verifyBeforeRemoval - Whether to verify before removal
   * @returns {boolean} Whether video should be queued
   */
  shouldQueueVideo(video, verifyBeforeRemoval) {
    // Don't queue if already removed or failed
    if (video.removalStatus === 'removed' || video.removalStatus === 'failed') {
      return false;
    }
    
    // If verification required, check verification score
    if (verifyBeforeRemoval) {
      const minScore = window.Constants?.LIMITS?.MIN_VERIFICATION_SCORE || 60;
      if (video.verificationScore < minScore) {
        console.warn(`Video ${video.videoId} has low verification score: ${video.verificationScore}`);
        return false;
      }
    }
    
    // Check if video is properly backed up
    if (video.backupStatus !== 'verified' && video.backupStatus !== 'exported') {
      console.warn(`Video ${video.videoId} is not properly backed up: ${video.backupStatus}`);
      return false;
    }
    
    return true;
  }

  /**
   * Start processing removal queue
   * @param {Function} onProgress - Progress callback
   * @param {Function} onComplete - Completion callback
   * @returns {Promise<Object>} Removal results
   */
  async processRemovalQueue(onProgress, onComplete) {
    if (this.isRemoving) {
      throw new Error('Removal process already in progress');
    }
    
    if (this.removalQueue.length === 0) {
      throw new Error('No videos queued for removal');
    }
    
    this.isRemoving = true;
    const totalQueued = this.removalQueue.length;
    let processed = 0;
    
    try {
      console.log(`Starting removal of ${totalQueued} videos`);
      
      while (this.removalQueue.length > 0 && this.isRemoving) {
        const item = this.removalQueue.shift();
        
        try {
          const result = await this.removeVideo(item);
          
          if (result.success) {
            this.removedVideos.push({
              video: item.video,
              removedAt: new Date(),
              result: result
            });
          } else {
            // Retry if not exceeded max retries
            if (item.retries < this.maxRetries) {
              item.retries++;
              this.removalQueue.push(item);
              console.log(`Retrying removal of video ${item.video.videoId} (attempt ${item.retries + 1})`);
            } else {
              this.failedRemovals.push({
                video: item.video,
                error: result.error,
                retries: item.retries
              });
            }
          }
          
          processed++;
          
          // Call progress callback
          if (onProgress) {
            onProgress({
              processed: processed,
              total: totalQueued,
              removed: this.removedVideos.length,
              failed: this.failedRemovals.length,
              remaining: this.removalQueue.length
            });
          }
          
          // Delay between removals to avoid rate limiting
          if (this.removalQueue.length > 0) {
            await this.youtubeAPI.delay(this.removalDelay);
          }
          
        } catch (error) {
          console.error(`Error removing video ${item.video.videoId}:`, error);
          this.failedRemovals.push({
            video: item.video,
            error: error.message,
            retries: item.retries
          });
          processed++;
        }
      }
      
      const results = {
        totalProcessed: processed,
        removed: this.removedVideos.length,
        failed: this.failedRemovals.length,
        removedVideos: this.removedVideos,
        failedRemovals: this.failedRemovals
      };
      
      console.log('Removal process completed:', results);
      
      if (onComplete) {
        onComplete(results);
      }
      
      return results;
      
    } finally {
      this.isRemoving = false;
    }
  }

  /**
   * Remove a single video from liked list
   * @param {Object} item - Queue item containing video and options
   * @returns {Promise<Object>} Removal result
   */
  async removeVideo(item) {
    const { video } = item;
    
    try {
      console.log(`Removing video: ${video.title} (${video.videoId})`);
      
      // Find the video element on the page
      const videoElement = await this.findVideoElement(video.videoId);
      
      if (!videoElement) {
        return {
          success: false,
          error: 'Video element not found on page'
        };
      }
      
      // Find and click the unlike button
      const unlikeButton = await this.findUnlikeButton(videoElement);
      
      if (!unlikeButton) {
        return {
          success: false,
          error: 'Unlike button not found'
        };
      }
      
      // Click the unlike button
      const clickSuccess = await this.youtubeAPI.clickElement(unlikeButton);
      
      if (!clickSuccess) {
        return {
          success: false,
          error: 'Failed to click unlike button'
        };
      }
      
      // Wait for removal to take effect
      await this.youtubeAPI.delay(1000);
      
      // Verify removal
      const isRemoved = await this.verifyRemoval(video.videoId);
      
      if (isRemoved) {
        return {
          success: true,
          removedAt: new Date()
        };
      } else {
        return {
          success: false,
          error: 'Removal verification failed'
        };
      }
      
    } catch (error) {
      console.error(`Error removing video ${video.videoId}:`, error);
      return {
        success: false,
        error: error.message
      };
    }
  }

  /**
   * Find video element on page by video ID
   * @param {string} videoId - Video ID to find
   * @returns {Promise<Element|null>} Video element or null
   */
  async findVideoElement(videoId) {
    const selectors = [
      `a[href*="watch?v=${videoId}"]`,
      `a[href*="v=${videoId}"]`
    ];
    
    for (const selector of selectors) {
      const linkElement = document.querySelector(selector);
      if (linkElement) {
        // Find the parent video container
        const videoContainer = linkElement.closest(
          window.Constants?.YOUTUBE_SELECTORS?.VIDEO_CONTAINER ||
          'ytd-playlist-video-renderer'
        );
        
        if (videoContainer) {
          return videoContainer;
        }
      }
    }
    
    return null;
  }

  /**
   * Find unlike button within video element
   * @param {Element} videoElement - Video container element
   * @returns {Promise<Element|null>} Unlike button or null
   */
  async findUnlikeButton(videoElement) {
    // Try different selectors for unlike button
    const selectors = [
      'button[aria-label*="Remove from"]',
      'button[aria-label*="Unlike"]',
      'button[title*="Remove from"]',
      '.ytd-menu-renderer button',
      'ytd-toggle-button-renderer button'
    ];
    
    for (const selector of selectors) {
      const button = videoElement.querySelector(selector);
      if (button && this.youtubeAPI.isElementVisible(button)) {
        return button;
      }
    }
    
    // Try to find menu button and open it
    const menuButton = videoElement.querySelector('button[aria-label="Action menu"]');
    if (menuButton) {
      await this.youtubeAPI.clickElement(menuButton);
      await this.youtubeAPI.delay(500);
      
      // Look for unlike option in menu
      const menuUnlikeButton = document.querySelector('tp-yt-paper-listbox button[aria-label*="Remove"]');
      if (menuUnlikeButton) {
        return menuUnlikeButton;
      }
    }
    
    return null;
  }

  /**
   * Verify that video was successfully removed
   * @param {string} videoId - Video ID to verify
   * @returns {Promise<boolean>} Whether video was removed
   */
  async verifyRemoval(videoId) {
    // Wait a bit for DOM to update
    await this.youtubeAPI.delay(2000);
    
    // Check if video element is still present
    const videoElement = await this.findVideoElement(videoId);
    
    // If element is not found or not visible, consider it removed
    return !videoElement || !this.youtubeAPI.isElementVisible(videoElement);
  }

  /**
   * Stop removal process
   */
  stopRemoval() {
    this.isRemoving = false;
    console.log('Removal process stopped');
  }

  /**
   * Clear removal queue
   */
  clearQueue() {
    this.removalQueue = [];
    console.log('Removal queue cleared');
  }

  /**
   * Get removal statistics
   * @returns {Object} Removal statistics
   */
  getStats() {
    return {
      isRemoving: this.isRemoving,
      queueLength: this.removalQueue.length,
      removedCount: this.removedVideos.length,
      failedCount: this.failedRemovals.length,
      removalDelay: this.removalDelay,
      maxRetries: this.maxRetries
    };
  }

  /**
   * Get removal history
   * @returns {Object} Removal history
   */
  getHistory() {
    return {
      removed: this.removedVideos,
      failed: this.failedRemovals
    };
  }

  /**
   * Restore a removed video (add back to liked list)
   * @param {Object} video - Video to restore
   * @returns {Promise<Object>} Restoration result
   */
  async restoreVideo(video) {
    try {
      // Open video page
      const videoUrl = `https://www.youtube.com/watch?v=${video.videoId}`;
      const tab = await chrome.tabs.create({ url: videoUrl, active: false });
      
      // Wait for page to load
      await this.youtubeAPI.delay(3000);
      
      // Find and click like button
      const likeButton = await this.youtubeAPI.waitForElement('button[aria-label*="like this video"]', 10000);
      
      if (likeButton) {
        await this.youtubeAPI.clickElement(likeButton);
        await this.youtubeAPI.delay(1000);
        
        // Close tab
        await chrome.tabs.remove(tab.id);
        
        return {
          success: true,
          restoredAt: new Date()
        };
      } else {
        await chrome.tabs.remove(tab.id);
        return {
          success: false,
          error: 'Like button not found'
        };
      }
      
    } catch (error) {
      console.error(`Error restoring video ${video.videoId}:`, error);
      return {
        success: false,
        error: error.message
      };
    }
  }

  /**
   * Reset removal handler state
   */
  reset() {
    this.isRemoving = false;
    this.removalQueue = [];
    this.removedVideos = [];
    this.failedRemovals = [];
  }
}

// Export for different environments
if (typeof module !== 'undefined' && module.exports) {
  module.exports = RemovalHandler;
} else {
  window.RemovalHandler = RemovalHandler;
}
