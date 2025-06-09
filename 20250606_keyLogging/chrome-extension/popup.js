// Popup Debug Interface for Chrome Extension
console.log('Popup debug interface loaded');

let debugEvents = [];

// Format timestamp for display
function formatTime(timestamp) {
  return new Date(timestamp * 1000).toLocaleTimeString();
}

// Update current tab information
async function updateCurrentTabInfo() {
  try {
    const tabs = await chrome.tabs.query({active: true, currentWindow: true});
    if (tabs[0]) {
      const tab = tabs[0];
      const domain = extractDomain(tab.url);
      
      document.getElementById('currentDomain').textContent = domain;
      document.getElementById('currentUrl').textContent = tab.url;
      document.getElementById('currentTitle').textContent = tab.title || 'No title';
      document.getElementById('lastUpdate').textContent = new Date().toLocaleTimeString();
      
      // Update status indicator
      const statusEl = document.getElementById('status');
      if (domain === 'unknown') {
        statusEl.className = 'status error';
      } else {
        statusEl.className = 'status active';
      }
    }
  } catch (error) {
    console.log('Error getting current tab:', error);
    document.getElementById('currentDomain').textContent = 'Error: ' + error.message;
  }
}

// Extract domain (same logic as background.js)
function extractDomain(url) {
  try {
    const urlObj = new URL(url);
    let domain = urlObj.hostname.toLowerCase();
    
    // Remove common prefixes
    domain = domain.replace(/^www\./, '');
    
    // Sanitize for Prometheus labels
    domain = domain.replace(/[^a-z0-9_]/g, '_');
    
    return domain || 'unknown';
  } catch (error) {
    console.log('Error parsing URL:', url, error);
    return 'unknown';
  }
}

// Test HTTP endpoint instead of native messaging
async function testNativeMessaging() {
  const statusEl = document.getElementById('nativeStatus');
  
  try {
    const testMessage = {
      type: 'test',
      domain: 'test_domain',
      timestamp: Date.now() / 1000
    };
    
    const response = await fetch('http://localhost:8080/chrome-update', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(testMessage)
    });
    
    if (response.ok) {
      const result = await response.json();
      statusEl.textContent = '✅ HTTP Working';
      console.log('HTTP endpoint test response:', result);
    } else {
      throw new Error(`HTTP ${response.status}: ${response.statusText}`);
    }
  } catch (error) {
    statusEl.innerHTML = '<span class="error-msg">❌ ' + error.message + '</span>';
  }
}

// Test storage access
async function testStorage() {
  const statusEl = document.getElementById('storageStatus');
  
  try {
    if (!chrome.storage || !chrome.storage.local) {
      statusEl.innerHTML = '<span class="error-msg">❌ Storage API not available</span>';
      return;
    }
    
    // Try to read storage
    chrome.storage.local.get(['chromeEvents', 'currentDomain', 'lastUpdate'], function(result) {
      if (chrome.runtime.lastError) {
        statusEl.innerHTML = '<span class="error-msg">❌ ' + chrome.runtime.lastError.message + '</span>';
      } else {
        statusEl.textContent = '✅ Working';
        
        // Update events count if available
        if (result.chromeEvents) {
          document.getElementById('eventsSent').textContent = result.chromeEvents.length;
        }
        
        console.log('Storage contents:', result);
      }
    });
  } catch (error) {
    statusEl.innerHTML = '<span class="error-msg">❌ ' + error.message + '</span>';
  }
}

// Load recent events from storage
function loadRecentEvents() {
  const eventsContainer = document.getElementById('recentEvents');
  
  try {
    chrome.storage.local.get(['chromeEvents'], function(result) {
      if (chrome.runtime.lastError) {
        eventsContainer.innerHTML = '<div class="error-msg">Error loading events: ' + chrome.runtime.lastError.message + '</div>';
        return;
      }
      
      const events = result.chromeEvents || [];
      
      if (events.length === 0) {
        eventsContainer.innerHTML = '<div style="color: #888; font-style: italic;">No events recorded yet</div>';
        return;
      }
      
      // Show last 10 events
      const recentEvents = events.slice(-10).reverse();
      eventsContainer.innerHTML = recentEvents.map(event => {
        const timeStr = formatTime(event.timestamp);
        const typeStr = event.type || 'unknown';
        let details = '';
        
        if (event.type === 'domain_switch') {
          details = `${event.from_domain} → ${event.to_domain}`;
        } else if (event.type === 'current_domain') {
          details = event.domain;
        } else {
          details = JSON.stringify(event).substring(0, 50) + '...';
        }
        
        return `
          <div class="event">
            <span class="event-time">${timeStr}</span>
            <span class="event-type">${typeStr}</span>
            <span>${details}</span>
          </div>
        `;
      }).join('');
    });
  } catch (error) {
    eventsContainer.innerHTML = '<div class="error-msg">Error: ' + error.message + '</div>';
  }
}

// Refresh all debug information
function refreshDebugInfo() {
  updateCurrentTabInfo();
  testNativeMessaging();
  testStorage();
  loadRecentEvents();
}

// Initialize popup
document.addEventListener('DOMContentLoaded', function() {
  console.log('Popup DOM loaded, initializing...');
  
  // Initial load
  refreshDebugInfo();
  
  // Set up refresh button
  document.getElementById('refreshBtn').addEventListener('click', refreshDebugInfo);
  
  // Auto-refresh every 5 seconds
  setInterval(refreshDebugInfo, 5000);
});

// Listen for messages from background script
chrome.runtime.onMessage.addListener(function(message, sender, sendResponse) {
  if (message.type === 'debug_update') {
    console.log('Debug update received:', message);
    refreshDebugInfo();
  }
});