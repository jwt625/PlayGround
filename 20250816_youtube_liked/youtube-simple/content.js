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


