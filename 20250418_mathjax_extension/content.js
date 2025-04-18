let isEnabled = true;
let mathJaxLoaded = false;

// Function to inject MathJax into the page
function injectMathJax() {
    console.log('Injecting MathJax...');
    const script = document.createElement('script');
    script.src = 'https://polyfill.io/v3/polyfill.min.js?features=es6';
    document.head.appendChild(script);

    script.onload = () => {
        console.log('Polyfill loaded, loading MathJax...');
        const mathJaxScript = document.createElement('script');
        mathJaxScript.src = 'https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js';
        mathJaxScript.async = true;
        
        mathJaxScript.onload = () => {
            console.log('MathJax loaded successfully');
            mathJaxLoaded = true;
            // Render equations after MathJax is loaded
            renderEquations();
        };
        
        document.head.appendChild(mathJaxScript);
    };
}

// Function to process text nodes and replace LaTeX equations
function processTextNodes(node) {
    if (!isEnabled) {
        console.log('Processing disabled, skipping...');
        return;
    }

    console.log('Processing text nodes...');
    const walker = document.createTreeWalker(
        node,
        NodeFilter.SHOW_TEXT,
        null,
        false
    );

    let textNode;
    let processedCount = 0;
    while (textNode = walker.nextNode()) {
        const text = textNode.textContent;
        if (text.includes('$')) {
            console.log('Found text with $ symbols:', text);
            const parts = text.split(/(\$[^\$]+\$)/g);
            if (parts.length > 1) {
                console.log('Split parts:', parts);
                const fragment = document.createDocumentFragment();
                parts.forEach(part => {
                    if (part.startsWith('$') && part.endsWith('$')) {
                        // Extract the LaTeX content between $ symbols
                        const latex = part.slice(1, -1);
                        console.log('Found LaTeX equation:', latex);
                        const span = document.createElement('span');
                        span.className = 'math-inline';
                        span.textContent = latex;
                        fragment.appendChild(span);
                        processedCount++;
                    } else {
                        fragment.appendChild(document.createTextNode(part));
                    }
                });
                textNode.parentNode.replaceChild(fragment, textNode);
            }
        }
    }
    console.log(`Processed ${processedCount} equations in this node`);
}

// Function to render equations after processing
function renderEquations() {
    if (!isEnabled) {
        console.log('Rendering disabled, skipping...');
        return;
    }
    
    if (!mathJaxLoaded) {
        console.log('MathJax not loaded yet, skipping rendering');
        return;
    }
    
    if (window.MathJax) {
        console.log('Rendering equations with MathJax...');
        try {
            MathJax.typesetPromise().then(() => {
                console.log('Equations rendered successfully');
            }).catch(error => {
                console.error('Error rendering equations:', error);
            });
        } catch (error) {
            console.error('Error in typesetPromise:', error);
        }
    } else {
        console.error('MathJax not available in window object');
    }
}

// Function to handle enabling/disabling
function setEnabled(enabled) {
    console.log('Setting enabled state to:', enabled);
    isEnabled = enabled;
    if (enabled) {
        processTextNodes(document.body);
        if (mathJaxLoaded) {
            renderEquations();
        } else {
            injectMathJax();
        }
    } else {
        console.log('Disabling, restoring original text...');
        // Remove all math spans and restore original text
        const mathSpans = document.querySelectorAll('.math-inline');
        console.log(`Found ${mathSpans.length} math spans to restore`);
        mathSpans.forEach(span => {
            const text = document.createTextNode('$' + span.textContent + '$');
            span.parentNode.replaceChild(text, span);
        });
    }
}

// Initialize the extension
function init() {
    console.log('Initializing extension...');
    // Load initial state
    chrome.storage.sync.get(['enabled'], function(result) {
        isEnabled = result.enabled !== false; // Default to true if not set
        console.log('Initial enabled state:', isEnabled);
        if (isEnabled) {
            injectMathJax();
            processTextNodes(document.body);
        }
    });
    
    // Set up a MutationObserver to handle dynamic content
    const observer = new MutationObserver((mutations) => {
        if (!isEnabled) return;
        
        let shouldRender = false;
        mutations.forEach((mutation) => {
            if (mutation.addedNodes.length) {
                mutation.addedNodes.forEach((node) => {
                    if (node.nodeType === Node.ELEMENT_NODE) {
                        processTextNodes(node);
                        shouldRender = true;
                    }
                });
            }
        });
        
        if (shouldRender && mathJaxLoaded) {
            renderEquations();
        }
    });

    observer.observe(document.body, {
        childList: true,
        subtree: true
    });

    // Listen for messages from popup
    chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
        console.log('Received message:', message);
        if (message.action === 'toggleEnabled') {
            setEnabled(message.enabled);
            sendResponse({success: true});
        }
        return true; // Keep the message channel open for async response
    });

    // Notify that content script is ready
    console.log('Content script initialized and ready');
}

// Start the extension
init(); 