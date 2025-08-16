/**
 * Pagination Handler for YouTube Liked Videos Backup Extension
 * Manages infinite scroll and batch processing of videos
 */

class PaginationHandler {
  constructor(videoScraper) {
    this.videoScraper = videoScraper;
    this.isScrolling = false;
    this.hasMoreContent = true;
    this.lastVideoCount = 0;
    this.scrollAttempts = 0;
    this.maxScrollAttempts = 5;
    this.batchSize = window.Constants?.LIMITS?.BATCH_SIZE || 50;
    this.scrollDelay = window.Constants?.TIMING?.SCROLL_DELAY || 2000;
    this.processedVideoIds = new Set();
    
    // Bind methods
    this.handleScroll = this.handleScroll.bind(this);
    this.checkForNewContent = this.checkForNewContent.bind(this);
  }

  /**
   * Start pagination process to load all videos
   * @param {Function} onBatchProcessed - Callback for each batch of videos
   * @param {Function} onProgress - Progress callback
   * @returns {Promise<Array<Object>>} All scraped videos
   */
  async processAllVideos(onBatchProcessed, onProgress) {
    const allVideos = [];
    let batchCount = 0;
    
    try {
      // Initial scrape of visible videos
      let currentVideos = await this.videoScraper.scrapeVisibleVideos();
      let newVideos = this.filterNewVideos(currentVideos);
      
      if (newVideos.length > 0) {
        allVideos.push(...newVideos);
        this.markVideosAsProcessed(newVideos);
        
        if (onBatchProcessed) {
          await onBatchProcessed(newVideos, ++batchCount);
        }
        
        if (onProgress) {
          onProgress({
            processed: allVideos.length,
            batch: batchCount,
            hasMore: this.hasMoreContent
          });
        }
      }
      
      // Continue scrolling and processing until no more content
      while (this.hasMoreContent && this.scrollAttempts < this.maxScrollAttempts) {
        await this.scrollToLoadMore();
        
        // Wait for content to load
        await this.waitForContentLoad();
        
        // Scrape new videos
        console.log('🔄 Scraping current batch of videos...');
        currentVideos = await this.videoScraper.scrapeVisibleVideos();
        console.log(`🔄 Found ${currentVideos.length} total videos on page`);

        newVideos = this.filterNewVideos(currentVideos);
        console.log(`🔄 Found ${newVideos.length} new videos (${this.processedVideoIds.size} already processed)`);
        
        if (newVideos.length > 0) {
          allVideos.push(...newVideos);
          this.markVideosAsProcessed(newVideos);
          this.scrollAttempts = 0; // Reset attempts on successful load
          
          if (onBatchProcessed) {
            await onBatchProcessed(newVideos, ++batchCount);
          }
          
          if (onProgress) {
            onProgress({
              processed: allVideos.length,
              batch: batchCount,
              hasMore: this.hasMoreContent
            });
          }
        } else {
          this.scrollAttempts++;
          console.log(`No new videos found, attempt ${this.scrollAttempts}/${this.maxScrollAttempts}`);
        }
        
        // Check if we've reached the end
        if (!await this.hasMoreContentToLoad()) {
          this.hasMoreContent = false;
          break;
        }
      }
      
      console.log(`Pagination complete. Processed ${allVideos.length} videos in ${batchCount} batches.`);
      return allVideos;
      
    } catch (error) {
      console.error('Error during pagination:', error);
      throw error;
    }
  }

  /**
   * Scroll to load more content
   * @returns {Promise<void>}
   */
  async scrollToLoadMore() {
    if (this.isScrolling) return;
    
    this.isScrolling = true;
    
    try {
      // Scroll to bottom of page
      window.scrollTo({
        top: document.body.scrollHeight,
        behavior: 'smooth'
      });
      
      // Wait for scroll to complete
      await this.delay(this.scrollDelay);
      
      // Alternative: trigger load more button if present
      const loadMoreButton = document.querySelector(window.Constants?.YOUTUBE_SELECTORS?.LOAD_MORE_BUTTON);
      if (loadMoreButton && loadMoreButton.offsetParent !== null) {
        loadMoreButton.click();
        await this.delay(1000);
      }
      
    } finally {
      this.isScrolling = false;
    }
  }

  /**
   * Wait for new content to load after scrolling
   * @returns {Promise<void>}
   */
  async waitForContentLoad() {
    const maxWaitTime = 10000; // 10 seconds max wait
    const checkInterval = 500; // Check every 500ms
    let waitTime = 0;
    
    while (waitTime < maxWaitTime) {
      // Check if spinner is visible (content loading)
      const spinner = document.querySelector(window.Constants?.YOUTUBE_SELECTORS?.SPINNER);
      if (spinner && spinner.offsetParent !== null) {
        await this.delay(checkInterval);
        waitTime += checkInterval;
        continue;
      }
      
      // Check if new videos have appeared
      const currentVideoCount = this.videoScraper.getVideoElements().length;
      if (currentVideoCount > this.lastVideoCount) {
        this.lastVideoCount = currentVideoCount;
        break;
      }
      
      await this.delay(checkInterval);
      waitTime += checkInterval;
    }
  }

  /**
   * Check if there's more content to load
   * @returns {Promise<boolean>} Whether more content is available
   */
  async hasMoreContentToLoad() {
    // Check for load more button
    const loadMoreButton = document.querySelector(window.Constants?.YOUTUBE_SELECTORS?.LOAD_MORE_BUTTON);
    if (loadMoreButton && loadMoreButton.offsetParent !== null) {
      return true;
    }
    
    // Check if we're at the bottom and no new content loaded
    const isAtBottom = (window.innerHeight + window.scrollY) >= document.body.offsetHeight - 1000;
    if (!isAtBottom) {
      return true;
    }
    
    // Check for continuation elements that indicate more content
    const continuations = document.querySelectorAll('[data-continuation]');
    if (continuations.length > 0) {
      return true;
    }
    
    // If we've made multiple scroll attempts without new content, assume we're done
    return this.scrollAttempts < this.maxScrollAttempts;
  }

  /**
   * Filter out videos that have already been processed
   * @param {Array<Object>} videos - Array of video records
   * @returns {Array<Object>} New videos only
   */
  filterNewVideos(videos) {
    return videos.filter(video => {
      if (!video.videoId || this.processedVideoIds.has(video.videoId)) {
        return false;
      }
      return true;
    });
  }

  /**
   * Mark videos as processed to avoid duplicates
   * @param {Array<Object>} videos - Array of video records
   */
  markVideosAsProcessed(videos) {
    videos.forEach(video => {
      if (video.videoId) {
        this.processedVideoIds.add(video.videoId);
      }
    });
  }

  /**
   * Process videos in batches
   * @param {Array<Object>} videos - All videos to process
   * @param {Function} processor - Function to process each batch
   * @returns {Promise<Array<Object>>} Processed results
   */
  async processBatches(videos, processor) {
    const results = [];
    
    for (let i = 0; i < videos.length; i += this.batchSize) {
      const batch = videos.slice(i, i + this.batchSize);
      
      try {
        const batchResults = await processor(batch, Math.floor(i / this.batchSize) + 1);
        if (batchResults) {
          results.push(...batchResults);
        }
      } catch (error) {
        console.error(`Error processing batch ${Math.floor(i / this.batchSize) + 1}:`, error);
      }
      
      // Small delay between batches to avoid overwhelming the system
      await this.delay(100);
    }
    
    return results;
  }

  /**
   * Get current scroll position and page info
   * @returns {Object} Scroll and page information
   */
  getScrollInfo() {
    return {
      scrollTop: window.pageYOffset || document.documentElement.scrollTop,
      scrollHeight: document.body.scrollHeight,
      clientHeight: window.innerHeight,
      scrollPercentage: ((window.pageYOffset || document.documentElement.scrollTop) / 
                        (document.body.scrollHeight - window.innerHeight)) * 100,
      isAtBottom: (window.innerHeight + window.scrollY) >= document.body.offsetHeight - 100,
      videoCount: this.videoScraper.getVideoElements().length
    };
  }

  /**
   * Scroll to specific position
   * @param {number} position - Scroll position (0-1, where 1 is bottom)
   * @returns {Promise<void>}
   */
  async scrollToPosition(position) {
    const targetScroll = (document.body.scrollHeight - window.innerHeight) * position;
    
    window.scrollTo({
      top: targetScroll,
      behavior: 'smooth'
    });
    
    await this.delay(1000);
  }

  /**
   * Reset pagination state
   */
  reset() {
    this.isScrolling = false;
    this.hasMoreContent = true;
    this.lastVideoCount = 0;
    this.scrollAttempts = 0;
    this.processedVideoIds.clear();
  }

  /**
   * Handle scroll events
   * @param {Event} event - Scroll event
   */
  handleScroll(event) {
    // Throttle scroll events
    if (this.scrollTimeout) {
      clearTimeout(this.scrollTimeout);
    }
    
    this.scrollTimeout = setTimeout(() => {
      this.checkForNewContent();
    }, 250);
  }

  /**
   * Check for new content after scroll
   */
  async checkForNewContent() {
    if (this.isScrolling) return;
    
    const scrollInfo = this.getScrollInfo();
    
    // If near bottom, try to load more content
    if (scrollInfo.scrollPercentage > 80 && this.hasMoreContent) {
      await this.scrollToLoadMore();
    }
  }

  /**
   * Add scroll listener
   */
  addScrollListener() {
    window.addEventListener('scroll', this.handleScroll, { passive: true });
  }

  /**
   * Remove scroll listener
   */
  removeScrollListener() {
    window.removeEventListener('scroll', this.handleScroll);
    if (this.scrollTimeout) {
      clearTimeout(this.scrollTimeout);
    }
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
   * Get pagination statistics
   * @returns {Object} Pagination stats
   */
  getStats() {
    return {
      processedVideos: this.processedVideoIds.size,
      scrollAttempts: this.scrollAttempts,
      hasMoreContent: this.hasMoreContent,
      isScrolling: this.isScrolling,
      lastVideoCount: this.lastVideoCount
    };
  }
}

// Export for different environments
if (typeof module !== 'undefined' && module.exports) {
  module.exports = PaginationHandler;
} else {
  window.PaginationHandler = PaginationHandler;
}
