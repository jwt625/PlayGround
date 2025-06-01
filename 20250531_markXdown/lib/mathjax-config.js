window.MathJax = {
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
            // Process any existing tweets
            document.querySelectorAll('[data-testid="tweet"]').forEach(tweet => {
                const tweetText = tweet.querySelector('[data-testid="tweetText"]');
                if (tweetText && tweetText.dataset.markdownProcessed) {
                    MathJax.typesetPromise([tweetText]).catch(function (err) {
                        console.error('MathJax error:', err);
                    });
                }
            });
        }
    }
}; 