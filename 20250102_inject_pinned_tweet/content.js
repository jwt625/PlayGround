
// content.js
let pinnedTweet = null;
let isInjected = false;

// Function to extract pinned tweet
async function getPinnedTweet(profileUrl) {
  try {
    // Wait for the profile page to load
    await waitForElement('[data-testid="pinned-tweet-text"]');
    
    // Find the pinned tweet container
    const pinnedTweetElement = document.querySelector('[data-testid="pinned-tweet"]');
    if (!pinnedTweetElement) return null;
    
    // Clone the pinned tweet
    return pinnedTweetElement.cloneNode(true);
  } catch (error) {
    console.error('Error getting pinned tweet:', error);
    return null;
  }
}

// Function to inject pinned tweet at the top of the feed
async function injectPinnedTweet() {
  if (isInjected || !pinnedTweet) return;
  
  const timeline = await waitForElement('[data-testid="primaryColumn"]');
  if (!timeline) return;
  
  // Create container for pinned tweet
  const container = document.createElement('div');
  container.classList.add('pinned-tweet-container');
  container.style.marginBottom = '20px';
  container.style.border = '1px solid rgb(239, 243, 244)';
  container.style.borderRadius = '16px';
  container.appendChild(pinnedTweet);
  
  // Insert at the top of the timeline
  const firstChild = timeline.firstChild;
  timeline.insertBefore(container, firstChild);
  isInjected = true;
}

// Utility function to wait for element
function waitForElement(selector) {
  return new Promise(resolve => {
    if (document.querySelector(selector)) {
      return resolve(document.querySelector(selector));
    }

    const observer = new MutationObserver(mutations => {
      if (document.querySelector(selector)) {
        observer.disconnect();
        resolve(document.querySelector(selector));
      }
    });

    observer.observe(document.body, {
      childList: true,
      subtree: true
    });
  });
}

// Initialize
async function init() {
  // Get saved profile URL
  const { profileUrl } = await chrome.storage.sync.get(['profileUrl']);
  if (!profileUrl) return;
  
  // Get pinned tweet first
  const pinned = await getPinnedTweet(profileUrl);
  if (pinned) {
    pinnedTweet = pinned;
    // Watch for feed changes and inject when possible
    const observer = new MutationObserver(() => {
      if (window.location.pathname === '/home' || window.location.pathname === '/') {
        injectPinnedTweet();
      }
    });
    
    observer.observe(document.body, {
      childList: true,
      subtree: true
    });
  }
}

init();