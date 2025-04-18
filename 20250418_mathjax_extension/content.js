let isEnabled = true;

// Function to inject MathJax into the page
function injectMathJax() {
    const script = document.createElement('script');
    script.src = 'https://polyfill.io/v3/polyfill.min.js?features=es6';
    document.head.appendChild(script);

    script.onload = () => {
        const mathJaxScript = document.createElement('script');
        mathJaxScript.src = 'https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js';
        mathJaxScript.async = true;
        document.head.appendChild(mathJaxScript);
    };
}

// Function to process text nodes and replace LaTeX equations
function processTextNodes(node) {
    if (!isEnabled) return;

    const walker = document.createTreeWalker(
        node,
        NodeFilter.SHOW_TEXT,
        null,
        false
    );

    let textNode;
    while (textNode = walker.nextNode()) {
        const text = textNode.textContent;
        if (text.includes('$')) {
            const parts = text.split(/(\$[^\$]+\$)/g);
            if (parts.length > 1) {
                const fragment = document.createDocumentFragment();
                parts.forEach(part => {
                    if (part.startsWith('$') && part.endsWith('$')) {
                        // Extract the LaTeX content between $ symbols
                        const latex = part.slice(1, -1);
                        const span = document.createElement('span');
                        span.className = 'math-inline';
                        span.textContent = latex;
                        fragment.appendChild(span);
                    } else {
                        fragment.appendChild(document.createTextNode(part));
                    }
                });
                textNode.parentNode.replaceChild(fragment, textNode);
            }
        }
    }
}

// Function to render equations after processing
function renderEquations() {
    if (window.MathJax && isEnabled) {
        MathJax.typesetPromise();
    }
}

// Function to handle enabling/disabling
function setEnabled(enabled) {
    isEnabled = enabled;
    if (enabled) {
        processTextNodes(document.body);
        renderEquations();
    } else {
        // Remove all math spans and restore original text
        const mathSpans = document.querySelectorAll('.math-inline');
        mathSpans.forEach(span => {
            const text = document.createTextNode('$' + span.textContent + '$');
            span.parentNode.replaceChild(text, span);
        });
    }
}

// Initialize the extension
function init() {
    // Load initial state
    chrome.storage.sync.get(['enabled'], function(result) {
        isEnabled = result.enabled !== false; // Default to true if not set
        if (isEnabled) {
            injectMathJax();
            processTextNodes(document.body);
        }
    });
    
    // Set up a MutationObserver to handle dynamic content
    const observer = new MutationObserver((mutations) => {
        if (!isEnabled) return;
        
        mutations.forEach((mutation) => {
            if (mutation.addedNodes.length) {
                mutation.addedNodes.forEach((node) => {
                    if (node.nodeType === Node.ELEMENT_NODE) {
                        processTextNodes(node);
                    }
                });
            }
        });
        renderEquations();
    });

    observer.observe(document.body, {
        childList: true,
        subtree: true
    });

    // Listen for messages from popup
    chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
        if (message.action === 'toggleEnabled') {
            setEnabled(message.enabled);
        }
    });
}

// Start the extension
init(); 