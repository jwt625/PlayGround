// SPA-aware downloader that works with ChatGPT's React structure

let isDownloading = false;
let chatLinks = [];
let currentIndex = 0;
let downloadedChats = [];

console.log('SPA downloader background script loaded');

// Message handlers
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  console.log('Background received:', request.action);
  
  switch (request.action) {
    case 'startSPADownload':
      startSPADownload(sender.tab.id).then(sendResponse);
      return true;
      
    case 'downloadChat':
      downloadChat(sender.tab.id, request.chatData, request.index).then(sendResponse);
      return true;
      
    case 'getStatus':
      sendResponse({
        isDownloading,
        current: currentIndex,
        total: chatLinks.length,
        downloaded: downloadedChats.length
      });
      break;
  }
});

async function startSPADownload(tabId) {
  console.log('Starting SPA download...');
  
  if (isDownloading) {
    return { error: 'Download already in progress' };
  }
  
  isDownloading = true;
  currentIndex = 0;
  downloadedChats = [];
  
  try {
    // Inject the SPA controller script
    await chrome.scripting.executeScript({
      target: { tabId },
      files: ['spa_controller.js']
    });
    
    console.log('SPA controller injected');
    return { success: true };
    
  } catch (error) {
    console.error('Failed to start SPA download:', error);
    isDownloading = false;
    return { error: error.message };
  }
}

async function downloadChat(tabId, chatData, index) {
  try {
    // Create safe filename
    const safeTitle = chatData.title.replace(/[^a-z0-9]/gi, '_').substring(0, 40);
    const filename = `chatgpt_export/${index.toString().padStart(3, '0')}_${safeTitle}.html`;
    
    // Download using data URL
    const dataUrl = 'data:text/html;charset=utf-8,' + encodeURIComponent(chatData.html);
    
    await chrome.downloads.download({
      url: dataUrl,
      filename: filename,
      saveAs: false
    });
    
    downloadedChats.push({ ...chatData, filename });
    currentIndex = index;
    
    console.log(`Downloaded ${index}/${chatLinks.length}: ${chatData.title}`);
    return { success: true };
    
  } catch (error) {
    console.error('Download error:', error);
    return { error: error.message };
  }
}