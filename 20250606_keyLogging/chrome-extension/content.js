// Content Script for URL changes within single-page applications
// This handles navigation that doesn't trigger tab events (e.g., React Router, Vue Router)

let lastUrl = location.href;

// Monitor for URL changes using history API and hash changes
function detectUrlChange() {
  if (location.href !== lastUrl) {
    const oldUrl = lastUrl;
    lastUrl = location.href;
    
    // Send URL change to background script
    chrome.runtime.sendMessage({
      type: 'url_change',
      oldUrl: oldUrl,
      newUrl: lastUrl,
      timestamp: Date.now() / 1000
    });
    
    console.log('SPA navigation detected:', oldUrl, '→', lastUrl);
  }
}

// Listen for browser navigation events
window.addEventListener('popstate', detectUrlChange);

// Override pushState and replaceState to catch programmatic navigation
const originalPushState = history.pushState;
const originalReplaceState = history.replaceState;

history.pushState = function(...args) {
  originalPushState.apply(history, args);
  setTimeout(detectUrlChange, 0); // Use setTimeout to ensure URL has changed
};

history.replaceState = function(...args) {
  originalReplaceState.apply(history, args);
  setTimeout(detectUrlChange, 0);
};

// Also check periodically in case we missed something
setInterval(detectUrlChange, 1000);

console.log('Content script loaded for URL:', location.href);