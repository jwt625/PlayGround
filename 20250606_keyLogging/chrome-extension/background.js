// Chrome Extension Background Script for Tab Tracking
console.log('Keystroke Tracker Chrome Extension loaded');

let currentDomain = '';
let lastSwitchTime = Date.now();
let eventsSentCount = 0;
let lastNativeMessagingStatus = 'untested';
let lastStorageStatus = 'untested';

// Extract clean domain from URL
function extractDomain(url) {
  try {
    const urlObj = new URL(url);
    let originalHostname = urlObj.hostname.toLowerCase();
    let domain = originalHostname;
    
    // Remove common prefixes
    domain = domain.replace(/^www\./, '');
    
    // Sanitize for Prometheus labels (same as Go code)
    domain = domain.replace(/[^a-z0-9_]/g, '_');
    
    const result = domain || 'unknown';
    console.log(`Domain extraction: ${url} → hostname: ${originalHostname} → cleaned: ${result}`);
    
    return result;
  } catch (error) {
    console.log('Error parsing URL:', url, error);
    return 'unknown';
  }
}

// Send data using HTTP endpoint (no native messaging)
function sendToNativeHost(data) {
  eventsSentCount++;
  console.log(`[Event ${eventsSentCount}] Sending via HTTP:`, data);
  
  // Send to Go application via HTTP
  sendToHttpEndpoint(data);
  
  // Also save to storage for debugging
  saveEventToDebugStorage(data);
  
  lastNativeMessagingStatus = 'disabled (using HTTP)';
  notifyDebugUpdate();
}

// Send data to Go application via HTTP POST
function sendToHttpEndpoint(data) {
  const endpoint = 'http://localhost:8080/chrome-update';
  
  fetch(endpoint, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(data)
  })
  .then(response => {
    if (response.ok) {
      return response.json();
    } else {
      throw new Error(`HTTP ${response.status}: ${response.statusText}`);
    }
  })
  .then(result => {
    console.log('HTTP endpoint success:', result);
    lastStorageStatus = 'working (HTTP)';
    notifyDebugUpdate();
  })
  .catch(error => {
    console.log('HTTP endpoint error:', error);
    lastStorageStatus = 'error: ' + error.message;
    
    // Fall back to file-based approach
    fallbackToStorage(data);
  });
}

// Fallback: Write to file that Go can read  
function fallbackToStorage(data) {
  console.log('Using file-based fallback for:', data);
  
  // Try to write to Downloads folder where Go can read it
  writeToDownloadFile(data);
  
  lastStorageStatus = 'working (file fallback)';
  notifyDebugUpdate();
}

// Simplified storage save for debugging
function saveEventToDebugStorage(data) {
  if (!chrome.storage || !chrome.storage.local) {
    console.log('Chrome storage API not available');
    return;
  }
  
  const timestamp = Date.now();
  chrome.storage.local.get(['chromeEvents'], function(result) {
    if (chrome.runtime.lastError) {
      console.log('Storage get error:', chrome.runtime.lastError.message);
      return;
    }
    
    const events = result.chromeEvents || [];
    events.push({
      ...data,
      timestamp: timestamp / 1000,
      source: 'fallback'
    });
    
    // Keep only last 100 events
    if (events.length > 100) {
      events.splice(0, events.length - 100);
    }
    
    chrome.storage.local.set({
      chromeEvents: events,
      currentDomain: data.type === 'current_domain' ? data.domain : currentDomain,
      lastUpdate: timestamp
    });
  });
}

// Save to storage for debugging (even when native messaging works)
function saveToStorageForDebug(data) {
  if (!chrome.storage || !chrome.storage.local) {
    return;
  }
  
  const timestamp = Date.now();
  chrome.storage.local.get(['chromeEvents'], function(result) {
    if (chrome.runtime.lastError) {
      return;
    }
    
    const events = result.chromeEvents || [];
    events.push({
      ...data,
      timestamp: timestamp / 1000,
      source: 'native_messaging'
    });
    
    // Keep only last 100 events
    if (events.length > 100) {
      events.splice(0, events.length - 100);
    }
    
    chrome.storage.local.set({
      chromeEvents: events,
      currentDomain: data.type === 'current_domain' ? data.domain : currentDomain,
      lastUpdate: timestamp,
      eventsSent: eventsSentCount,
      nativeStatus: lastNativeMessagingStatus,
      storageStatus: lastStorageStatus
    });
  });
}

// Write current domain to a predictable file location
function writeToDownloadFile(data) {
  try {
    // Create simple domain update structure
    const domainData = {
      domain: data.type === 'current_domain' ? data.domain : currentDomain,
      timestamp: Date.now() / 1000,
      url: data.url || '',
      type: data.type || 'unknown'
    };
    
    // Create JSONL format for events (append-style)
    const jsonLine = JSON.stringify(domainData) + '\n';
    
    // Convert to blob
    const blob = new Blob([jsonLine], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    
    // Generate timestamp-based filename to avoid conflicts
    const timestamp = Math.floor(Date.now() / 1000);
    const filename = `keystroke_tracker_chrome_${timestamp}.json`;
    
    // Use Downloads API (this will go to ~/Downloads/)
    if (chrome.downloads && chrome.downloads.download) {
      chrome.downloads.download({
        url: url,
        filename: filename,
        conflictAction: 'overwrite',
        saveAs: false
      }, function(downloadId) {
        URL.revokeObjectURL(url);
        if (chrome.runtime.lastError) {
          console.log('Download API error:', chrome.runtime.lastError.message);
          lastStorageStatus = 'error: ' + chrome.runtime.lastError.message;
        } else {
          console.log('Successfully downloaded:', filename, 'ID:', downloadId, 'Domain:', domainData.domain);
          lastStorageStatus = 'working (downloads API)';
        }
        notifyDebugUpdate();
      });
    } else {
      console.log('Downloads API not available');
      lastStorageStatus = 'error: downloads API not available';
      URL.revokeObjectURL(url);
      notifyDebugUpdate();
    }
  } catch (error) {
    console.log('Download file error:', error);
    lastStorageStatus = 'error: ' + error.message;
    notifyDebugUpdate();
  }
}

// Notify popup of debug updates
function notifyDebugUpdate() {
  try {
    chrome.runtime.sendMessage({
      type: 'debug_update',
      currentDomain: currentDomain,
      eventsSent: eventsSentCount,
      nativeStatus: lastNativeMessagingStatus,
      storageStatus: lastStorageStatus
    }, function(response) {
      // Handle response or error silently
      if (chrome.runtime.lastError) {
        // Popup not open, ignore the error
      }
    });
  } catch (error) {
    // Popup might not be open, ignore errors
  }
}

// Handle tab activation (switching between existing tabs)
chrome.tabs.onActivated.addListener(async (activeInfo) => {
  try {
    const tab = await chrome.tabs.get(activeInfo.tabId);
    const newDomain = extractDomain(tab.url);
    
    if (newDomain !== currentDomain && currentDomain !== '') {
      const switchEvent = {
        type: 'domain_switch',
        from_domain: currentDomain,
        to_domain: newDomain,
        timestamp: Date.now() / 1000
      };
      
      sendToNativeHost(switchEvent);
      console.log('Tab switch:', currentDomain, '→', newDomain);
    }
    
    // Update current domain
    const domainEvent = {
      type: 'current_domain',
      domain: newDomain,
      url: tab.url,
      title: tab.title,
      timestamp: Date.now() / 1000
    };
    
    sendToNativeHost(domainEvent);
    currentDomain = newDomain;
    lastSwitchTime = Date.now();
    
  } catch (error) {
    console.log('Error handling tab activation:', error);
  }
});

// Handle tab updates (URL changes, page loads)
chrome.tabs.onUpdated.addListener(async (_, changeInfo, tab) => {
  // Only process when URL actually changes and tab is active
  if (changeInfo.url && tab.active) {
    try {
      const newDomain = extractDomain(changeInfo.url);
      
      if (newDomain !== currentDomain) {
        if (currentDomain !== '') {
          const switchEvent = {
            type: 'domain_switch',
            from_domain: currentDomain,
            to_domain: newDomain,
            timestamp: Date.now() / 1000
          };
          
          sendToNativeHost(switchEvent);
          console.log('URL change:', currentDomain, '→', newDomain);
        }
        
        // Update current domain
        const domainEvent = {
          type: 'current_domain',
          domain: newDomain,
          url: changeInfo.url,
          title: tab.title,
          timestamp: Date.now() / 1000
        };
        
        sendToNativeHost(domainEvent);
        currentDomain = newDomain;
        lastSwitchTime = Date.now();
      }
    } catch (error) {
      console.log('Error handling tab update:', error);
    }
  }
});

// Initialize with current active tab
chrome.tabs.query({active: true, currentWindow: true}, function(tabs) {
  if (tabs[0]) {
    currentDomain = extractDomain(tabs[0].url);
    const initEvent = {
      type: 'current_domain',
      domain: currentDomain,
      url: tabs[0].url,
      title: tabs[0].title,
      timestamp: Date.now() / 1000
    };
    
    sendToNativeHost(initEvent);
    console.log('Initialized with domain:', currentDomain);
  }
});