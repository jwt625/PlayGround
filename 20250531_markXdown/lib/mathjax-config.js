window.MathJax = {
    tex: {
        inlineMath: [['$', '$'], ['\\(', '\\)']],
        displayMath: [['$$', '$$'], ['\\[', '\\]']],
        processEscapes: true
    },
    chtml: {
        fontURL: chrome.runtime.getURL('lib/mathjax/output/chtml/fonts/woff-v2'),
        scale: 1,
        minScale: .5
    },
    options: {
        skipHtmlTags: ['script', 'noscript', 'style', 'textarea', 'pre']
    }
}; 