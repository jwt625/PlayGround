document.addEventListener('DOMContentLoaded', function() {
    const toggle = document.getElementById('markdownToggle');
    console.log('Popup loaded');
    
    // Load saved state
    chrome.storage.sync.get(['markdownEnabled'], function(result) {
        console.log('Loaded state:', result);
        toggle.checked = result.markdownEnabled !== false; // Default to true if not set
    });
    
    // Save state when changed
    toggle.addEventListener('change', function() {
        const enabled = toggle.checked;
        console.log('Toggle changed to:', enabled);
        
        chrome.storage.sync.set({ markdownEnabled: enabled }, function() {
            console.log('State saved:', enabled);
        });
        
        // Notify content script
        chrome.tabs.query({active: true, currentWindow: true}, function(tabs) {
            if (tabs[0]) {
                console.log('Sending message to tab:', tabs[0].id);
                chrome.tabs.sendMessage(tabs[0].id, {
                    action: 'toggleMarkdown',
                    enabled: enabled
                }, function(response) {
                    console.log('Message sent, response:', response);
                });
            } else {
                console.log('No active tab found');
            }
        });
    });
}); 