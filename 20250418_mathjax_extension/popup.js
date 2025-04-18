document.addEventListener('DOMContentLoaded', function() {
    console.log('Popup loaded');
    const toggle = document.getElementById('enableToggle');
    const status = document.getElementById('status');

    // Load initial state
    chrome.storage.sync.get(['enabled'], function(result) {
        console.log('Loaded initial state:', result);
        toggle.checked = result.enabled !== false; // Default to true if not set
        updateStatus(result.enabled !== false);
    });

    // Handle toggle changes
    toggle.addEventListener('change', function() {
        const enabled = toggle.checked;
        console.log('Toggle changed to:', enabled);
        
        chrome.storage.sync.set({ enabled: enabled }, function() {
            console.log('Storage updated with enabled:', enabled);
            updateStatus(enabled);
            
            // Notify content script of the change
            chrome.tabs.query({active: true, currentWindow: true}, function(tabs) {
                if (tabs[0]) {
                    console.log('Sending message to tab:', tabs[0].id);
                    try {
                        chrome.tabs.sendMessage(tabs[0].id, {
                            action: 'toggleEnabled',
                            enabled: enabled
                        }, function(response) {
                            if (chrome.runtime.lastError) {
                                console.log('Error sending message:', chrome.runtime.lastError);
                                // If we can't send the message, we should still update the storage
                                // The content script will pick up the change on next page load
                            } else {
                                console.log('Message sent successfully, response:', response);
                            }
                        });
                    } catch (error) {
                        console.error('Error in sendMessage:', error);
                    }
                } else {
                    console.log('No active tab found');
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