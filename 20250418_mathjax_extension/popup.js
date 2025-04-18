document.addEventListener('DOMContentLoaded', function() {
    const toggle = document.getElementById('enableToggle');
    const status = document.getElementById('status');

    // Load initial state
    chrome.storage.sync.get(['enabled'], function(result) {
        toggle.checked = result.enabled !== false; // Default to true if not set
        updateStatus(result.enabled !== false);
    });

    // Handle toggle changes
    toggle.addEventListener('change', function() {
        const enabled = toggle.checked;
        chrome.storage.sync.set({ enabled: enabled }, function() {
            updateStatus(enabled);
            
            // Notify content script of the change
            chrome.tabs.query({active: true, currentWindow: true}, function(tabs) {
                if (tabs[0]) {
                    chrome.tabs.sendMessage(tabs[0].id, {
                        action: 'toggleEnabled',
                        enabled: enabled
                    }).catch(error => {
                        console.log('Could not send message to content script:', error);
                        // If we can't send the message, we should still update the storage
                        // The content script will pick up the change on next page load
                    });
                }
            });
        });
    });

    function updateStatus(enabled) {
        status.textContent = enabled ? 
            'LaTeX rendering is enabled' : 
            'LaTeX rendering is disabled';
        status.style.color = enabled ? '#4CAF50' : '#666';
    }
}); 