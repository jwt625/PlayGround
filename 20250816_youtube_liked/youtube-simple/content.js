console.log('YouTube Simple Extension content script loaded');

// Listen for messages from popup
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  console.log('Content script received message:', message);

  if (message.type === 'getVideoTitles') {
    try {
      const videos = getVideoTitles();
      console.log('Found videos:', videos);
      sendResponse({ success: true, videos: videos });
    } catch (error) {
      console.error('Error getting video titles:', error);
      sendResponse({ success: false, error: error.message });
    }
    return false; // Synchronous response
  } else if (message.type === 'scrollToBottom') {
    console.log('Scrolling to bottom of page...');

    try {
      // Simple scroll to bottom
      window.scrollTo(0, document.body.scrollHeight);
      console.log('Scrolled to bottom successfully');
      sendResponse({ success: true });
    } catch (error) {
      console.error('Error scrolling:', error);
      sendResponse({ success: false, error: error.message });
    }
    return false; // Synchronous response
  } else if (message.type === 'removeTopVideo') {
    console.log('Attempting to remove top video...');

    // Handle async removeTopVideo function
    removeTopVideo()
      .then(result => {
        sendResponse(result);
      })
      .catch(error => {
        console.error('Error removing video:', error);
        sendResponse({ success: false, error: error.message });
      });

    return true; // Async response
  } else if (message.type === 'scrapeVideoInfo') {
    console.log('Scraping video info from current page...');

    try {
      const videoInfo = scrapeVideoInfo();
      sendResponse(videoInfo);
    } catch (error) {
      console.error('Error scraping video info:', error);
      sendResponse({ success: false, error: error.message });
    }
    return false; // Synchronous response
  }

  return false; // Default to synchronous
});

function getVideoTitles() {
  console.log('Getting video titles from page...');
  console.log('Current URL:', window.location.href);
  
  const videos = [];
  
  // Try multiple selectors for different YouTube layouts
  const selectors = [
    'ytd-playlist-video-renderer',  // Playlist view
    'ytd-grid-video-renderer',      // Grid view
    'ytd-video-renderer',           // Search results, etc.
    'ytd-compact-video-renderer'    // Sidebar, etc.
  ];
  
  let videoElements = [];
  
  for (const selector of selectors) {
    const elements = document.querySelectorAll(selector);
    console.log(`Selector "${selector}": found ${elements.length} elements`);
    
    if (elements.length > 0) {
      videoElements = Array.from(elements);
      console.log(`Using selector: ${selector}`);
      break;
    }
  }
  
  if (videoElements.length === 0) {
    console.log('No video elements found, trying generic selectors...');
    
    // Try some generic selectors
    const genericSelectors = [
      '[data-video-id]',
      'a[href*="/watch?v="]',
      'a[href*="youtube.com/watch"]'
    ];
    
    for (const selector of genericSelectors) {
      const elements = document.querySelectorAll(selector);
      console.log(`Generic selector "${selector}": found ${elements.length} elements`);
      
      if (elements.length > 0) {
        // For generic selectors, we need to find their parent containers
        const containers = new Set();
        elements.forEach(el => {
          let container = el.closest('ytd-playlist-video-renderer, ytd-grid-video-renderer, ytd-video-renderer');
          if (container) containers.add(container);
        });
        videoElements = Array.from(containers);
        console.log(`Found ${videoElements.length} video containers from generic selector`);
        break;
      }
    }
  }
  
  console.log(`Processing ${videoElements.length} video elements...`);
  
  videoElements.forEach((element, index) => {
    try {
      const video = extractVideoData(element);
      if (video.title || video.videoId) {
        videos.push(video);
        console.log(`Video ${index + 1}:`, video);
      }
    } catch (error) {
      console.error(`Error processing video ${index + 1}:`, error);
    }
  });
  
  console.log(`Total videos extracted: ${videos.length}`);
  return videos;
}

function extractVideoData(element) {
  const video = {};
  
  // Try to get title
  const titleSelectors = [
    'a#video-title',
    'h3 a',
    '.ytd-video-meta-block h3 a',
    'a[href*="/watch?v="]'
  ];
  
  for (const selector of titleSelectors) {
    const titleElement = element.querySelector(selector);
    if (titleElement) {
      video.title = titleElement.textContent?.trim();
      video.url = titleElement.href;
      if (video.url) {
        video.videoId = extractVideoId(video.url);
      }
      break;
    }
  }
  
  // Try to get channel name
  const channelSelectors = [
    'ytd-channel-name a',
    '.ytd-channel-name a',
    'a[href*="/channel/"]',
    'a[href*="/@"]'
  ];
  
  for (const selector of channelSelectors) {
    const channelElement = element.querySelector(selector);
    if (channelElement) {
      video.channel = channelElement.textContent?.trim();
      break;
    }
  }
  
  return video;
}

function extractVideoId(url) {
  if (!url) return null;

  const match = url.match(/[?&]v=([^&]+)/);
  return match ? match[1] : null;
}

// Helper function to simulate proper mouse clicks
function simulateClick(element) {
  console.log('🖱️ Simulating click on element:', element);

  // Create and dispatch multiple events to ensure YouTube responds
  const events = [
    new MouseEvent('mousedown', { bubbles: true, cancelable: true }),
    new MouseEvent('mouseup', { bubbles: true, cancelable: true }),
    new MouseEvent('click', { bubbles: true, cancelable: true })
  ];

  events.forEach(event => {
    element.dispatchEvent(event);
  });

  // Also try the simple click as fallback
  element.click();
}

async function removeTopVideo() {
  console.log('🗑️ Looking for top video to remove...');

  try {
    // Find the first video element
    const selectors = [
      'ytd-playlist-video-renderer',
      'ytd-grid-video-renderer',
      'ytd-video-renderer',
      '[data-video-id]'
    ];

    let firstVideo = null;
    for (const selector of selectors) {
      const videos = document.querySelectorAll(selector);
      console.log(`🔍 Checking selector "${selector}": found ${videos.length} elements`);
      if (videos.length > 0) {
        firstVideo = videos[0];
        console.log(`🎯 Found first video using selector: ${selector}`);
        break;
      }
    }

    if (!firstVideo) {
      return { success: false, error: 'No videos found on page' };
    }

    // Get video title for confirmation
    const titleElement = firstVideo.querySelector('a#video-title, h3 a, a[href*="/watch?v="]');
    const videoTitle = titleElement ? titleElement.textContent?.trim() : 'Unknown video';
    console.log(`🎬 Video to remove: "${videoTitle}"`);

    // Look for the action menu button
    const actionMenuButton = firstVideo.querySelector('button[aria-label="Action menu"]');

    if (!actionMenuButton) {
      console.log('❌ Action menu button not found');
      return { success: false, error: 'Action menu button not found' };
    }

    console.log('📋 Found action menu button, clicking...');
    simulateClick(actionMenuButton);

    // Wait for the menu to appear and then look for the remove option
    return new Promise((resolve) => {
      const maxAttempts = 5;
      let attempts = 0;
      let resolved = false;

      // Set a timeout to prevent hanging
      const timeoutId = setTimeout(() => {
        if (!resolved) {
          resolved = true;
          console.log('⏰ Timeout reached, resolving with error');
          resolve({ success: false, error: 'Operation timed out after 5 seconds' });
        }
      }, 5000);

      const checkForMenu = () => {
        if (resolved) return;

        attempts++;
        console.log(`🔍 Attempt ${attempts}/${maxAttempts} to find menu...`);

        try {
          // Look for the remove option in ytd-popup-container
          const popupContainer = document.querySelector('ytd-popup-container');
          if (!popupContainer) {
            console.log('❌ Popup container not found');
            if (attempts < maxAttempts) {
              setTimeout(checkForMenu, 300);
              return;
            }
            if (!resolved) {
              resolved = true;
              clearTimeout(timeoutId);
              resolve({ success: false, error: 'Menu popup not found after multiple attempts' });
            }
            return;
          }

          console.log('📋 Found popup container, looking for remove option...');

          // Look for the specific menu item with "Remove from Liked videos" text
          const removeMenuItems = popupContainer.querySelectorAll('tp-yt-paper-item');
          console.log(`📋 Found ${removeMenuItems.length} paper items in popup`);

          let removeItem = null;
          for (const item of removeMenuItems) {
            const text = item.textContent || '';
            console.log(`📋 Menu item text: "${text.trim()}"`);

            if (text.includes('Remove from Liked videos') || text.includes('Remove from liked videos')) {
              removeItem = item;
              break;
            }
          }

          if (removeItem) {
            console.log('🗑️ Found "Remove from Liked videos" option, clicking...');
            simulateClick(removeItem);
            console.log('✅ Successfully clicked remove option');
            if (!resolved) {
              resolved = true;
              clearTimeout(timeoutId);
              resolve({ success: true, videoTitle: videoTitle });
            }
          } else {
            if (!resolved) {
              resolved = true;
              clearTimeout(timeoutId);
              resolve({ success: false, error: 'Remove option not found in menu' });
            }
          }
        } catch (error) {
          console.error('❌ Error in menu handling:', error);
          if (!resolved) {
            resolved = true;
            clearTimeout(timeoutId);
            resolve({ success: false, error: `Menu handling error: ${error.message}` });
          }
        }
      };

      // Start checking for menu after initial delay
      setTimeout(checkForMenu, 500);
    });

  } catch (error) {
    console.error('❌ Error in removeTopVideo:', error);
    return { success: false, error: `Removal error: ${error.message}` };
  }
}

function scrapeVideoInfo() {
  console.log('🎬 Scraping video info from current page...');
  console.log('Current URL:', window.location.href);

  // Check if we're on a video page
  if (!window.location.href.includes('/watch?v=')) {
    return { success: false, error: 'Not on a video page. Please navigate to a YouTube video first.' };
  }

  const videoInfo = {
    scrapedAt: new Date().toISOString(),
    clickedAt: new Date().toISOString(), // When the scrape button was clicked
    source: 'YouTube Video Info Scraper',
    url: window.location.href
  };

  try {
    // Extract video ID from URL
    const urlParams = new URLSearchParams(window.location.search);
    videoInfo.videoId = urlParams.get('v');
    console.log('Video ID:', videoInfo.videoId);

    // Find the main ytd-watch-metadata element
    const watchMetadata = document.querySelector('ytd-watch-metadata');
    if (!watchMetadata) {
      return { success: false, error: 'ytd-watch-metadata element not found. Page may not be fully loaded.' };
    }

    console.log('✅ Found ytd-watch-metadata element');

    // Extract video title from ytd-watch-metadata
    const titleElement = watchMetadata.querySelector('h1 yt-formatted-string');
    if (titleElement && titleElement.textContent) {
      videoInfo.title = titleElement.textContent.trim();
      console.log('Found title:', videoInfo.title);
    }

    // Extract channel information from ytd-watch-metadata
    const channelElement = watchMetadata.querySelector('ytd-channel-name a');
    if (channelElement) {
      videoInfo.channel = channelElement.textContent?.trim();
      videoInfo.channelUrl = channelElement.href;
      console.log('Found channel:', videoInfo.channel);
    }

    // Extract subscriber count
    const subscriberElement = watchMetadata.querySelector('#owner-sub-count');
    if (subscriberElement && subscriberElement.textContent) {
      videoInfo.subscriberCount = subscriberElement.textContent.trim();
      console.log('Found subscriber count:', videoInfo.subscriberCount);
    }

    // Extract view count and upload date from ytd-watch-info-text
    const watchInfoText = watchMetadata.querySelector('ytd-watch-info-text');
    if (watchInfoText) {
      // Get the formatted info string that contains both views and date
      const infoElement = watchInfoText.querySelector('#info yt-formatted-string, #info');
      if (infoElement && infoElement.textContent) {
        const infoText = infoElement.textContent.trim();
        console.log('Found info text:', infoText);

        // Parse views and date from the combined text (e.g., "229K views  2 days ago")
        const parts = infoText.split(/\s{2,}/); // Split on multiple spaces
        if (parts.length >= 2) {
          videoInfo.viewCount = parts[0];
          videoInfo.uploadDate = parts[1];
        } else {
          // Fallback: try to extract views and date separately
          if (infoText.includes('view')) {
            const viewMatch = infoText.match(/[\d,KMB.]+\s*views?/i);
            if (viewMatch) videoInfo.viewCount = viewMatch[0];
          }
          if (infoText.includes('ago') || infoText.includes('Premiered') || infoText.includes('Published')) {
            const dateMatch = infoText.match(/\d+\s+(second|minute|hour|day|week|month|year)s?\s+ago|Premiered\s+.+|Published\s+.+/i);
            if (dateMatch) videoInfo.uploadDate = dateMatch[0];
          }
        }
      }

      // Also try to get the tooltip for more precise date
      const tooltip = watchInfoText.querySelector('tp-yt-paper-tooltip #tooltip');
      if (tooltip && tooltip.textContent) {
        const tooltipText = tooltip.textContent.trim();
        console.log('Found tooltip with precise date:', tooltipText);
        videoInfo.preciseDate = tooltipText;
      }
    }

    // Extract like and dislike counts from the like/dislike buttons
    const likeButton = watchMetadata.querySelector('like-button-view-model button, segmented-like-dislike-button-view-model like-button-view-model button');
    if (likeButton) {
      const likeText = likeButton.querySelector('.yt-spec-button-shape-next__button-text-content');
      if (likeText && likeText.textContent && likeText.textContent.trim() !== '') {
        videoInfo.likeCount = likeText.textContent.trim();
        console.log('Found like count:', videoInfo.likeCount);
      }

      // Get aria-label for more detailed like info
      const ariaLabel = likeButton.getAttribute('aria-label');
      if (ariaLabel) {
        videoInfo.likeAriaLabel = ariaLabel;
        console.log('Found like aria-label:', ariaLabel);
      }
    }

    const dislikeButton = watchMetadata.querySelector('dislike-button-view-model button, segmented-like-dislike-button-view-model dislike-button-view-model button');
    if (dislikeButton) {
      const dislikeText = dislikeButton.querySelector('.yt-spec-button-shape-next__button-text-content');
      if (dislikeText && dislikeText.textContent && dislikeText.textContent.trim() !== '') {
        videoInfo.dislikeCount = dislikeText.textContent.trim();
        console.log('Found dislike count:', videoInfo.dislikeCount);
      }
    }

    // Extract video description from multiple possible locations within ytd-watch-metadata
    let descriptionFound = false;

    // Method 1: Try the expanded description first (most complete)
    const expandedDesc = watchMetadata.querySelector('ytd-text-inline-expander #expanded yt-attributed-string');
    if (expandedDesc && expandedDesc.textContent && expandedDesc.textContent.trim()) {
      videoInfo.description = expandedDesc.textContent.trim();
      videoInfo.descriptionSource = 'expanded';
      descriptionFound = true;
      console.log('Found expanded description length:', videoInfo.description.length);
    }

    // Method 2: Try the snippet description (visible portion)
    if (!descriptionFound) {
      const snippetDesc = watchMetadata.querySelector('ytd-text-inline-expander #attributed-snippet-text');
      if (snippetDesc && snippetDesc.textContent && snippetDesc.textContent.trim()) {
        videoInfo.description = snippetDesc.textContent.trim();
        videoInfo.descriptionSource = 'snippet';
        descriptionFound = true;
        console.log('Found snippet description length:', videoInfo.description.length);
      }
    }

    // Method 3: Try the description text container
    if (!descriptionFound) {
      const descContainer = watchMetadata.querySelector('#description-text-container #attributed-description-text');
      if (descContainer && descContainer.textContent && descContainer.textContent.trim()) {
        videoInfo.description = descContainer.textContent.trim();
        videoInfo.descriptionSource = 'container';
        descriptionFound = true;
        console.log('Found container description length:', videoInfo.description.length);
      }
    }

    // Method 4: Fallback to any attributed string in the description area
    if (!descriptionFound) {
      const fallbackDesc = watchMetadata.querySelector('#description yt-attributed-string, ytd-text-inline-expander yt-attributed-string');
      if (fallbackDesc && fallbackDesc.textContent && fallbackDesc.textContent.trim()) {
        videoInfo.description = fallbackDesc.textContent.trim();
        videoInfo.descriptionSource = 'fallback';
        descriptionFound = true;
        console.log('Found fallback description length:', videoInfo.description.length);
      }
    }

    if (!descriptionFound) {
      console.log('⚠️ No description found');
      videoInfo.description = '';
      videoInfo.descriptionSource = 'none';
    }

    // Extract video duration from the player
    const durationElement = document.querySelector('.ytp-time-duration');
    if (durationElement && durationElement.textContent) {
      videoInfo.duration = durationElement.textContent.trim();
      console.log('Found duration:', videoInfo.duration);
    }

    // Extract comment count (this might be below the fold)
    const commentCountElement = document.querySelector('ytd-comments-header-renderer #count yt-formatted-string');
    if (commentCountElement && commentCountElement.textContent) {
      videoInfo.commentCount = commentCountElement.textContent.trim();
      console.log('Found comment count:', videoInfo.commentCount);
    }

    console.log('✅ Successfully scraped video info:', videoInfo);
    return { success: true, videoInfo: videoInfo };

  } catch (error) {
    console.error('❌ Error scraping video info:', error);
    return { success: false, error: `Scraping error: ${error.message}` };
  }
}


