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
// Track processed tweets to prevent double processing
const processedTweets = new Set();

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
const mathJaxQueue = [];
let isProcessingMathJax = false;
let isMathJaxReady = false;

// Wait for MathJax to be ready
function waitForMathJax() {
    return new Promise((resolve) => {
        if (window.MathJax && window.MathJax.typesetPromise) {
            isMathJaxReady = true;
            resolve();
            return;
        }

        const checkMathJax = setInterval(() => {
            if (window.MathJax && window.MathJax.typesetPromise) {
                clearInterval(checkMathJax);
                isMathJaxReady = true;
                resolve();
            }
        }, 100);

        // Set a timeout to prevent infinite waiting
        setTimeout(() => {
            clearInterval(checkMathJax);
            console.error('MathJax initialization timeout');
        }, 10000);
    });
}

// Process the MathJax queue
async function processMathJaxQueue() {
    if (isProcessingMathJax || mathJaxQueue.length === 0) return;
    
    isProcessingMathJax = true;
    try {
        // Wait for MathJax to be ready
        await waitForMathJax();
        
        if (!isMathJaxReady) {
            console.error('MathJax is not ready');
            return;
        }

        console.log('Processing MathJax queue with', mathJaxQueue.length, 'items');
        await MathJax.typesetPromise(mathJaxQueue);
        console.log('MathJax queue processed successfully');
    } catch (error) {
        console.error('Error processing MathJax queue:', error);
    } finally {
        isProcessingMathJax = false;
        mathJaxQueue.length = 0; // Clear the queue
    }
}

// Function to clean URLs in text
function cleanUrls(text) {
    return text.replace(/(https?:\/\/[^\s]+)…/g, '$1');
}

// Function to process a tweet's text content
async function processTweet(tweet, enabled = true) {
    const textElement = tweet.querySelector('[data-testid="tweetText"]');
    if (!textElement || processedTweets.has(tweet)) return;

    // Store original content if not already stored
    if (!originalContents.has(tweet)) {
        originalContents.set(tweet, textElement.innerHTML);
    }

    if (!enabled) {
        // Restore original content
        textElement.innerHTML = originalContents.get(tweet);
        return;
    }

    // Get the text content and clean it
    const originalContent = textElement.textContent;
    console.log('Original content:', originalContent);
    
    // Clean URLs before markdown processing
    const cleanedText = cleanUrls(originalContent);
    console.log('Cleaned text content:', cleanedText);
    
    // Process markdown
    const processedContent = marked.parse(cleanedText);
    console.log('Processed markdown:', processedContent);
    
    // Sanitize the HTML
    const sanitizedContent = sanitizeHtml(processedContent);
    console.log('Sanitized content:', sanitizedContent);
    
    // Update the tweet content
    textElement.innerHTML = sanitizedContent;

    // Mark as processed
    processedTweets.add(tweet);

    // Check if the tweet contains math
    if (sanitizedContent.includes('$')) {
        console.log('Tweet contains math, adding to MathJax queue');
        mathJaxQueue.push(textElement);
        processMathJaxQueue();
    }
}

// Create a MutationObserver to watch for new tweets
const observer = new MutationObserver((mutations) => {
    mutations.forEach((mutation) => {
        mutation.addedNodes.forEach((node) => {
            if (node.nodeType === Node.ELEMENT_NODE) {
                // Check if the added node is a tweet
                const tweets = node.querySelectorAll('[data-testid="tweet"]');
                tweets.forEach(tweet => processTweet(tweet, isMarkdownEnabled));
                
                // Also check the node itself if it's a tweet
                if (node.matches('[data-testid="tweet"]')) {
                    processTweet(node, isMarkdownEnabled);
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
    document.querySelectorAll('[data-testid="tweet"]').forEach(tweet => processTweet(tweet, isMarkdownEnabled));
});

// Listen for toggle messages from popup
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
    console.log('Received message:', request);
    if (request.action === 'toggleMarkdown') {
        isMarkdownEnabled = request.enabled;
        console.log('Markdown toggled to:', isMarkdownEnabled);
        
        // Clear processed tweets set when toggling
        processedTweets.clear();
        
        // Reprocess all tweets with new state
        document.querySelectorAll('[data-testid="tweet"]').forEach(tweet => processTweet(tweet, isMarkdownEnabled));
        
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