// Background script to handle bulk downloads
// This persists even when popup closes or pages navigate

let isDownloading = false;
let downloadQueue = [];
let currentIndex = 0;
let allChats = [];
let downloadedFiles = [];

// Listen for messages from popup
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  console.log('Background received message:', request.action);
  
  if (request.action === 'startBulkDownload') {
    if (isDownloading) {
      sendResponse({ error: 'Download already in progress' });
      return;
    }
    
    startBulkDownload();
    sendResponse({ success: true });
  } else if (request.action === 'getDownloadStatus') {
    sendResponse({
      isDownloading: isDownloading,
      currentIndex: currentIndex,
      total: allChats.length,
      currentChat: allChats[currentIndex] || null,
      completed: downloadedFiles.length
    });
  }
  
  return true;
});

async function startBulkDownload() {
  console.log('Starting bulk download...');
  isDownloading = true;
  currentIndex = 0;
  downloadedFiles = [];
  
  try {
    // Get active tab
    const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
    
    // First, extract all chat URLs while we're on the main page
    console.log('Extracting chat list...');
    allChats = await extractChatList(tab.id);
    
    if (allChats.length === 0) {
      throw new Error('No chats found');
    }
    
    console.log(`Found ${allChats.length} chats to download`);
    
    // Process each chat
    for (let i = 0; i < allChats.length; i++) {
      currentIndex = i;
      const chat = allChats[i];
      
      console.log(`Downloading chat ${i + 1}/${allChats.length}: ${chat.title}`);
      
      try {
        // Navigate to the chat
        await navigateToChat(tab.id, chat.url);
        
        // Wait for page to fully load
        console.log('Waiting 15 seconds for page to load...');
        await new Promise(resolve => setTimeout(resolve, 15000));
        
        // Download the HTML
        const filename = await downloadChatHTML(tab.id, chat, i + 1, allChats);
        
        downloadedFiles.push({
          ...chat,
          filename: filename,
          index: i + 1
        });
        
      } catch (error) {
        console.error(`Failed to download ${chat.title}:`, error);
        downloadedFiles.push({
          ...chat,
          filename: null,
          index: i + 1,
          error: error.message
        });
      }
    }
    
    // Create index file
    await createIndexFile(downloadedFiles);
    
    console.log('Bulk download complete!');
    
  } catch (error) {
    console.error('Bulk download error:', error);
  } finally {
    isDownloading = false;
  }
}

async function extractChatList(tabId) {
  return new Promise((resolve) => {
    chrome.scripting.executeScript({
      target: { tabId: tabId },
      func: async () => {
        const chats = [];
        const historyDiv = document.getElementById('history');
        
        if (!historyDiv) {
          return { error: 'No history div found' };
        }
        
        // First, scroll to load all chats
        console.log('Scrolling to load all chats...');
        const scrollContainer = historyDiv.closest('[class*="overflow-y-auto"]') || historyDiv.parentElement;
        
        if (scrollContainer) {
          let lastHeight = scrollContainer.scrollHeight;
          let attempts = 0;
          
          while (attempts < 10) {
            scrollContainer.scrollTop = scrollContainer.scrollHeight;
            await new Promise(r => setTimeout(r, 1000));
            
            if (scrollContainer.scrollHeight === lastHeight) {
              break;
            }
            lastHeight = scrollContainer.scrollHeight;
            attempts++;
          }
        }
        
        // Extract all chat links
        const chatLinks = historyDiv.querySelectorAll('a[href^="/c/"], a[href^="/chat/"]');
        
        chatLinks.forEach(link => {
          const titleSpan = link.querySelector('span[dir="auto"]');
          const title = titleSpan ? titleSpan.textContent.trim() : 'Untitled Chat';
          const href = link.getAttribute('href');
          
          if (href) {
            chats.push({
              title: title,
              href: href,
              url: new URL(href, window.location.origin).toString()
            });
          }
        });
        
        return { chats: chats };
      }
    }, (results) => {
      if (results && results[0] && results[0].result) {
        resolve(results[0].result.chats || []);
      } else {
        resolve([]);
      }
    });
  });
}

async function navigateToChat(tabId, url) {
  return new Promise((resolve) => {
    chrome.tabs.update(tabId, { url: url }, () => {
      chrome.tabs.onUpdated.addListener(function listener(updatedTabId, changeInfo) {
        if (updatedTabId === tabId && changeInfo.status === 'complete') {
          chrome.tabs.onUpdated.removeListener(listener);
          resolve();
        }
      });
    });
  });
}

async function downloadChatHTML(tabId, chat, index, allChatsList) {
  return new Promise((resolve) => {
    chrome.scripting.executeScript({
      target: { tabId: tabId },
      func: (allChatsData, currentIndex) => {
        let html = document.documentElement.outerHTML;
        
        // Update all chat links in the sidebar to point to local files
        allChatsData.forEach((chatData, idx) => {
          const safeTitle = chatData.title.replace(/[^a-z0-9]/gi, '_').toLowerCase().substring(0, 50);
          const localFilename = `${(idx + 1).toString().padStart(3, '0')}_${safeTitle}.html`;
          
          const oldHref = chatData.href;
          const escapedOldHref = oldHref.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
          
          html = html.replace(
            new RegExp(`href="${escapedOldHref}"`, 'g'),
            `href="${localFilename}"`
          );
        });
        
        return html;
      },
      args: [allChatsList, index]
    }, (results) => {
      if (!results || !results[0]) {
        resolve(null);
        return;
      }
      
      const modifiedHtml = results[0].result;
      const blob = new Blob([modifiedHtml], { type: 'text/html' });
      const url = URL.createObjectURL(blob);
      
      const safeTitle = chat.title.replace(/[^a-z0-9]/gi, '_').toLowerCase().substring(0, 50);
      const filenameOnly = `${index.toString().padStart(3, '0')}_${safeTitle}.html`;
      const filename = `chatgpt_export/${filenameOnly}`;
      
      chrome.downloads.download({
        url: url,
        filename: filename,
        saveAs: false,
        conflictAction: 'uniquify'
      }, () => {
        URL.revokeObjectURL(url);
        resolve(filenameOnly);
      });
    });
  });
}

async function createIndexFile(downloadedFiles) {
  const indexHTML = `
<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8">
  <title>ChatGPT Export Index</title>
  <style>
    body {
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', sans-serif;
      max-width: 1200px;
      margin: 0 auto;
      padding: 20px;
    }
    .chat-item {
      padding: 10px;
      border-bottom: 1px solid #e5e5e5;
    }
    .success { color: #10a37f; }
    .error { color: #ef4444; }
  </style>
</head>
<body>
  <h1>ChatGPT Export - ${new Date().toLocaleDateString()}</h1>
  <p>Total: ${downloadedFiles.length} | Success: ${downloadedFiles.filter(f => f.filename).length}</p>
  <div>
    ${downloadedFiles.map(file => `
      <div class="chat-item">
        ${file.filename ? 
          `<a href="${file.filename}">${file.index}. ${file.title}</a> <span class="success">✓</span>` :
          `${file.index}. ${file.title} <span class="error">✗ ${file.error || 'Failed'}</span>`
        }
      </div>
    `).join('')}
  </div>
</body>
</html>`;
  
  const blob = new Blob([indexHTML], { type: 'text/html' });
  const url = URL.createObjectURL(blob);
  
  chrome.downloads.download({
    url: url,
    filename: 'chatgpt_export/index.html',
    saveAs: false
  }, () => {
    URL.revokeObjectURL(url);
  });
}