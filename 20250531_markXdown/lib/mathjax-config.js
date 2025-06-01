window.MathJax = {
    loader: {
        load: ['input/tex', 'output/chtml', 'ui/menu']
    },
    tex: {
        inlineMath: [['$', '$'], ['\\(', '\\)']],
        displayMath: [['$$', '$$'], ['\\[', '\\]']],
        processEscapes: true,
        processEnvironments: true,
        packages: ['base', 'ams', 'noerrors', 'noundefined']
    },
    options: {
        skipHtmlTags: ['script', 'noscript', 'style', 'textarea', 'pre'],
        ignoreHtmlClass: 'tex2jax_ignore',
        processHtmlClass: 'tex2jax_process'
    },
    startup: {
        ready: () => {
            console.log('MathJax is loaded and ready');
            // Process any queued elements
            if (window.processMathJaxQueue) {
                window.processMathJaxQueue();
            }
        }
    }
}; 