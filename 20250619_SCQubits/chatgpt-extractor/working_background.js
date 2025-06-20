// Working background script - combined functionality

console.log('Background script loaded');

// Message handlers
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  console.log('Background received:', request.action);
  
  if (request.action === 'testConnection') {
    console.log('Test connection successful');
    sendResponse({ success: true, message: 'Background script is working' });
    return true;
  }
  
  if (request.action === 'startSPADownload') {
    console.log('Starting SPA download...');
    
    // Get active tab
    chrome.tabs.query({ active: true, currentWindow: true }, async (tabs) => {
      const tab = tabs[0];
      console.log('Active tab:', tab.url);
      
      try {
        // Inject the SPA controller
        await chrome.scripting.executeScript({
          target: { tabId: tab.id },
          files: ['spa_controller.js']
        });
        
        console.log('SPA controller injected successfully');
        sendResponse({ success: true });
      } catch (error) {
        console.error('Failed to inject script:', error);
        sendResponse({ error: error.message });
      }
    });
    
    return true; // Keep channel open for async response
  }
  
  if (request.action === 'downloadChat') {
    console.log('Downloading chat:', request.index, request.chatData.title);
    
    const { chatData, index } = request;
    const safeTitle = chatData.title.replace(/[^a-z0-9]/gi, '_').substring(0, 40);
    const filename = index === 0 ? 
      'chatgpt_export/index.html' : 
      `chatgpt_export/${index.toString().padStart(3, '0')}_${safeTitle}.html`;
    
    // Use data URL to download
    const dataUrl = 'data:text/html;charset=utf-8,' + encodeURIComponent(chatData.html);
    
    chrome.downloads.download({
      url: dataUrl,
      filename: filename,
      saveAs: false
    }, (downloadId) => {
      if (chrome.runtime.lastError) {
        console.error('Download error:', chrome.runtime.lastError);
        sendResponse({ error: chrome.runtime.lastError.message });
      } else {
        console.log('Downloaded:', filename);
        sendResponse({ success: true });
      }
    });
    
    return true; // Keep channel open for async response
  }
});