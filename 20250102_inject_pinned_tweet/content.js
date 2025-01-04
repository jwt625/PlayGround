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

// Function to get tweet from the actual rendered page
async function getTweetFromDOM() {
    debugLog('Scrape', 'Starting DOM scraping');
    
    // Wait for the tweet to be rendered
    const selectors = [
        '[data-testid="tweet"]',
        '[role="article"]',
        '[data-testid="tweetText"]',
        '.css-175oi2r.r-18u37iz',
        'article'
    ];
    
    let tweetElement = null;
    
    // Try multiple times with a delay
    for (let attempt = 0; attempt < 10; attempt++) {
        debugLog('Scrape', `Attempt ${attempt + 1} to find tweet`);
        
        for (const selector of selectors) {
            const element = document.querySelector(selector);
            if (element) {
                debugLog('Scrape', `Found element with selector: ${selector}`);
                tweetElement = element;
                break;
            }
        }
        
        if (tweetElement) break;
        await new Promise(r => setTimeout(r, 1000)); // Wait 1 second between attempts
    }
    
    if (!tweetElement) {
        debugLog('Scrape', 'Failed to find tweet after all attempts');
        return null;
    }
    
    // Send both raw and rendered content to popup
    chrome.runtime.sendMessage({
        type: 'SCRAPED_CONTENT',
        rawHtml: document.documentElement.outerHTML,
        parsedContent: tweetElement.outerHTML,
        timestamp: new Date().toISOString()
    });
    
    return tweetElement.cloneNode(true);
}

// Function to inject tweet at the top of the feed
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
    label.textContent = '📌 Pinned Tweet';
    label.style.marginBottom = '8px';
    label.style.fontWeight = 'bold';
    label.style.color = 'rgb(83, 100, 113)';
    container.appendChild(label);
    
    // Add tweet
    container.appendChild(pinnedTweet);
    debugLog('Inject', 'Created container with tweet');
    
    // Find the right place to insert
    const composeBox = timeline.querySelector('[data-testid="tweetTextarea_0"]');
    const insertAfter = composeBox ? 
        composeBox.closest('[data-testid="cellInnerDiv"]') : 
        timeline.firstChild;
    
    timeline.insertBefore(container, insertAfter.nextSibling);
    isInjected = true;
    debugLog('Inject', 'Successfully injected tweet into timeline');
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
        // If we're on a tweet page, try to scrape it
        if (window.location.pathname.includes('/status/')) {
            debugLog('Init', 'On tweet page, attempting to scrape');
            const tweet = await getTweetFromDOM();
            if (tweet) {
                debugLog('Init', 'Successfully scraped tweet from page');
            }
            return;
        }
        
        // Otherwise check if we need to inject the pinned tweet
        debugLog('Init', 'Starting extension initialization');
        
        // Get saved tweet URL
        const { tweetUrl } = await chrome.storage.sync.get(['tweetUrl']);
        if (!tweetUrl) {
            debugLog('Init', 'No tweet URL found in storage');
            return;
        }
        debugLog('Init', `Found tweet URL: ${tweetUrl}`);
        
        if (isHomePage() && pinnedTweet) {
            debugLog('Init', 'On home page with pinned tweet, attempting injection');
            await injectPinnedTweet();
            
            // Watch for navigation changes
            debugLog('Init', 'Setting up navigation observer');
            const observer = new MutationObserver(async () => {
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