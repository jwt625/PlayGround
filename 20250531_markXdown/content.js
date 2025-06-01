// Configuration for marked.js
marked.setOptions({
    breaks: true,
    gfm: true,
    headerIds: false,
    mangle: false,
    tables: true,
    sanitize: false,
    smartLists: true,
    smartypants: true,
    pedantic: false,
    xhtml: false
});

// Store original tweet contents
const originalContents = new Map();

// Function to sanitize HTML
function sanitizeHtml(html) {
    const div = document.createElement('div');
    div.innerHTML = html;
    
    // Remove any script tags
    const scripts = div.getElementsByTagName('script');
    while (scripts.length > 0) {
        scripts[0].parentNode.removeChild(scripts[0]);
    }
    
    // Remove any style tags
    const styles = div.getElementsByTagName('style');
    while (styles.length > 0) {
        styles[0].parentNode.removeChild(styles[0]);
    }
    
    return div.innerHTML;
}

// Queue for tweets that need MathJax processing
let mathJaxQueue = [];

// Function to process MathJax queue
function processMathJaxQueue() {
    if (mathJaxQueue.length > 0 && window.MathJax && window.MathJax.typesetPromise) {
        const elements = mathJaxQueue.splice(0, mathJaxQueue.length);
        MathJax.typesetPromise(elements).catch(function (err) {
            console.error('MathJax error:', err);
        });
    }
}

// Function to clean URLs in text
function cleanUrls(text) {
    return text.replace(/(https?:\/\/[^\s]+)…/g, '$1');
}

// Function to process a tweet's text content
function processTweet(tweetElement) {
    const tweetText = tweetElement.querySelector('[data-testid="tweetText"]');
    if (!tweetText || tweetText.dataset.markdownProcessed) return;

    // Store original content
    const originalContent = tweetText.innerHTML;
    
    // Extract text content and clean URLs
    const textContent = cleanUrls(tweetText.textContent);
    
    // Process markdown
    const processedContent = marked.parse(textContent);
    
    // Update the content
    tweetText.innerHTML = processedContent;
    
    // Mark as processed
    tweetText.dataset.markdownProcessed = 'true';
    
    // Trigger MathJax to process the content
    if (window.MathJax && window.MathJax.typesetPromise) {
        MathJax.typesetPromise([tweetText]).catch(function (err) {
            console.error('MathJax error:', err);
        });
    }
}

// Create a MutationObserver to watch for new tweets
const observer = new MutationObserver((mutations) => {
    mutations.forEach((mutation) => {
        mutation.addedNodes.forEach((node) => {
            if (node.nodeType === Node.ELEMENT_NODE) {
                // Check if the added node is a tweet
                const tweets = node.querySelectorAll('[data-testid="tweet"]');
                tweets.forEach(tweet => processTweet(tweet));
                
                // Also check the node itself if it's a tweet
                if (node.matches('[data-testid="tweet"]')) {
                    processTweet(node);
                }
            }
        });
    });
});

// Start observing the document
observer.observe(document.body, {
    childList: true,
    subtree: true
});

// Track markdown enabled state
let isMarkdownEnabled = true;

// Load initial state
chrome.storage.sync.get(['markdownEnabled'], function(result) {
    console.log('Content script loaded state:', result);
    isMarkdownEnabled = result.markdownEnabled !== false;
    console.log('Markdown enabled:', isMarkdownEnabled);
    
    // Process existing tweets with current state
    document.querySelectorAll('[data-testid="tweet"]').forEach(tweet => 
        processTweet(tweet)
    );
});

// Listen for toggle messages from popup
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
    console.log('Received message:', request);
    if (request.action === 'toggleMarkdown') {
        isMarkdownEnabled = request.enabled;
        console.log('Markdown toggled to:', isMarkdownEnabled);
        
        // Reprocess all tweets with new state
        document.querySelectorAll('[data-testid="tweet"]').forEach(tweet => 
            processTweet(tweet)
        );
        
        // Send response back to popup
        sendResponse({ success: true });
    }
    return true; // Keep the message channel open for async response
});

// Check for MathJax loading every 100ms
const checkMathJaxInterval = setInterval(() => {
    if (window.MathJax && window.MathJax.typesetPromise) {
        clearInterval(checkMathJaxInterval);
        processMathJaxQueue();
    }
}, 100); 