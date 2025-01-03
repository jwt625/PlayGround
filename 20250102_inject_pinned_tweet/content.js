let pinnedTweet = null;
let isInjected = false;

// Debug logging function
function debugLog(step, message, data = null) {
    const logMessage = `[Twitter Pin Extension] ${step}: ${message}`;
    if (data) {
        console.log(logMessage, data);
    } else {
        console.log(logMessage);
    }
}

// Function to fetch profile page and extract pinned tweet
async function getPinnedTweet(profileUrl) {
    try {
        debugLog('Fetch', `Starting to fetch profile page: ${profileUrl}`);
        
        // Fetch the profile page content
        const response = await fetch(profileUrl);
        if (!response.ok) {
            debugLog('Fetch Error', `Failed to fetch profile page. Status: ${response.status}`);
            return null;
        }
        debugLog('Fetch', 'Successfully fetched profile page');
        
        const text = await response.text();
        debugLog('Parse', 'Got page content, length:', text.length);
        
        // Create a temporary DOM parser
        const parser = new DOMParser();
        const doc = parser.parseFromString(text, 'text/html');
        debugLog('Parse', 'Parsed HTML document');
        
        // Find the pinned tweet in the parsed document
        const pinnedTweetElement = doc.querySelector('[data-testid="pinned-tweet"]');
        if (!pinnedTweetElement) {
            debugLog('Extract', 'No pinned tweet found on profile');
            return null;
        }
        debugLog('Extract', 'Found pinned tweet element', pinnedTweetElement);
        
        // Clone the pinned tweet
        const clonedTweet = pinnedTweetElement.cloneNode(true);
        debugLog('Clone', 'Cloned pinned tweet element');
        
        // Fix any relative URLs in the tweet content
        const links = clonedTweet.querySelectorAll('a');
        debugLog('URLs', `Fixing ${links.length} links in tweet`);
        links.forEach(link => {
            if (link.href && link.href.startsWith('/')) {
                const oldHref = link.href;
                link.href = `https://twitter.com${link.href}`;
                debugLog('URLs', `Updated link from ${oldHref} to ${link.href}`);
            }
        });
        
        const images = clonedTweet.querySelectorAll('img');
        debugLog('URLs', `Fixing ${images.length} images in tweet`);
        images.forEach(img => {
            if (img.src && img.src.startsWith('/')) {
                const oldSrc = img.src;
                img.src = `https://twitter.com${img.src}`;
                debugLog('URLs', `Updated image from ${oldSrc} to ${img.src}`);
            }
        });
        
        debugLog('Success', 'Successfully processed pinned tweet');
        return clonedTweet;
    } catch (error) {
        debugLog('Error', 'Error getting pinned tweet:', error);
        return null;
    }
}

// Function to inject pinned tweet at the top of the feed
async function injectPinnedTweet() {
    if (isInjected || !pinnedTweet) {
        debugLog('Inject', 'Skipping injection - already injected or no tweet available');
        return;
    }
    
    debugLog('Inject', 'Waiting for timeline element');
    const timeline = await waitForElement('[data-testid="primaryColumn"]');
    if (!timeline) {
        debugLog('Inject', 'Failed to find timeline element');
        return;
    }
    debugLog('Inject', 'Found timeline element');
    
    // Create container for pinned tweet
    const container = document.createElement('div');
    container.classList.add('pinned-tweet-container');
    container.style.marginBottom = '20px';
    container.style.border = '1px solid rgb(239, 243, 244)';
    container.style.borderRadius = '16px';
    container.style.padding = '12px';
    container.style.backgroundColor = 'white';
    
    // Add a "Pinned Tweet" label
    const label = document.createElement('div');
    label.textContent = '📌 Your Pinned Tweet';
    label.style.marginBottom = '8px';
    label.style.fontWeight = 'bold';
    label.style.color = 'rgb(83, 100, 113)';
    container.appendChild(label);
    
    // Add the tweet
    container.appendChild(pinnedTweet);
    debugLog('Inject', 'Created container with tweet');
    
    // Find the right place to insert
    const composeBox = timeline.querySelector('[data-testid="tweetTextarea_0"]');
    const insertAfter = composeBox ? 
        composeBox.closest('[data-testid="cellInnerDiv"]') : 
        timeline.firstChild;
    
    timeline.insertBefore(container, insertAfter.nextSibling);
    isInjected = true;
    debugLog('Inject', 'Successfully injected pinned tweet into timeline');
}

// Utility function to wait for element
function waitForElement(selector, timeout = 5000) {
    debugLog('Wait', `Starting to wait for element: ${selector}`);
    return new Promise((resolve) => {
        if (document.querySelector(selector)) {
            debugLog('Wait', `Element already exists: ${selector}`);
            return resolve(document.querySelector(selector));
        }

        const observer = new MutationObserver((mutations) => {
            if (document.querySelector(selector)) {
                observer.disconnect();
                debugLog('Wait', `Element found: ${selector}`);
                resolve(document.querySelector(selector));
            }
        });

        observer.observe(document.body, {
            childList: true,
            subtree: true
        });

        // Add timeout to avoid infinite waiting
        setTimeout(() => {
            observer.disconnect();
            debugLog('Wait', `Timeout waiting for element: ${selector}`);
            resolve(null);
        }, timeout);
    });
}

// Function to check if we're on the home page
function isHomePage() {
    const isHome = window.location.pathname === '/home' || 
                  window.location.pathname === '/' || 
                  window.location.pathname === '/x';
    debugLog('Navigation', `Checking if home page: ${isHome}`, window.location.pathname);
    return isHome;
}

// Initialize
async function init() {
    try {
        debugLog('Init', 'Starting extension initialization');
        
        // Get saved profile URL
        const { profileUrl } = await chrome.storage.sync.get(['profileUrl']);
        if (!profileUrl) {
            debugLog('Init', 'No profile URL found in storage');
            return;
        }
        debugLog('Init', `Found profile URL: ${profileUrl}`);
        
        // Get pinned tweet first
        debugLog('Init', 'Attempting to get pinned tweet');
        const pinned = await getPinnedTweet(profileUrl);
        if (pinned) {
            debugLog('Init', 'Successfully got pinned tweet');
            pinnedTweet = pinned;
            
            // Initial check for home page
            if (isHomePage()) {
                debugLog('Init', 'On home page, attempting initial injection');
                await injectPinnedTweet();
            }
            
            // Watch for navigation changes
            debugLog('Init', 'Setting up navigation observer');
            const observer = new MutationObserver(async (mutations) => {
                if (isHomePage() && !isInjected) {
                    debugLog('Navigation', 'Detected navigation to home page, attempting injection');
                    await injectPinnedTweet();
                }
            });
            
            observer.observe(document.body, {
                childList: true,
                subtree: true
            });
            debugLog('Init', 'Navigation observer setup complete');
        } else {
            debugLog('Init', 'Failed to get pinned tweet');
        }
    } catch (error) {
        debugLog('Error', 'Error in initialization:', error);
    }
}

// Reset injection state on navigation
let lastPath = window.location.pathname;
setInterval(() => {
    if (window.location.pathname !== lastPath) {
        debugLog('Navigation', `Path changed from ${lastPath} to ${window.location.pathname}`);
        lastPath = window.location.pathname;
        isInjected = false;
    }
}, 1000);

debugLog('Load', 'Content script loaded');
init();