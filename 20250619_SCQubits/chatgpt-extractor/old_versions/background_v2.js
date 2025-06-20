// Background script v2 - with persistent state management

let downloadState = {
  isActive: false,
  allChats: [],
  currentIndex: 0,
  downloadedFiles: [],
  lastUrl: null,
  startTime: null
};

// Load state on startup
chrome.runtime.onInstalled.addListener(() => {
  loadState();
});

chrome.runtime.onStartup.addListener(() => {
  loadState();
});

// Save state to storage
async function saveState() {
  await chrome.storage.local.set({ downloadState });
  
  // Also save to a file for backup
  const stateJson = JSON.stringify(downloadState, null, 2);
  const blob = new Blob([stateJson], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  
  chrome.downloads.download({
    url: url,
    filename: 'chatgpt_export/download_state.json',
    saveAs: false,
    conflictAction: 'overwrite'
  }, () => {
    URL.revokeObjectURL(url);
  });
}

// Load state from storage
async function loadState() {
  const result = await chrome.storage.local.get('downloadState');
  if (result.downloadState) {
    downloadState = result.downloadState;
    console.log('Loaded state:', downloadState);
    
    // If download was active, check if we should resume
    if (downloadState.isActive) {
      console.log('Previous download was interrupted. Ready to resume.');
    }
  }
}

// Listen for tab updates to auto-continue download
chrome.tabs.onUpdated.addListener(async (tabId, changeInfo, tab) => {
  if (changeInfo.status === 'complete' && 
      tab.url && 
      (tab.url.includes('chatgpt.com') || tab.url.includes('chat.openai.com'))) {
    
    // If we're in the middle of a download, continue
    if (downloadState.isActive && downloadState.currentIndex < downloadState.allChats.length) {
      console.log('Page loaded, checking if we should continue download...');
      
      const currentChat = downloadState.allChats[downloadState.currentIndex];
      if (tab.url === currentChat.url) {
        console.log('Correct chat loaded, waiting then downloading...');
        
        // Wait for page to fully render
        setTimeout(async () => {
          await downloadCurrentChat(tabId);
          await moveToNextChat(tabId);
        }, 15000); // 15 second wait
      }
    }
  }
});

// Message handlers
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  console.log('Background received:', request.action);
  
  switch (request.action) {
    case 'startBulkDownload':
      startBulkDownload().then(sendResponse);
      return true;
      
    case 'resumeDownload':
      resumeDownload().then(sendResponse);
      return true;
      
    case 'getStatus':
      sendResponse(downloadState);
      break;
      
    case 'cancelDownload':
      downloadState.isActive = false;
      saveState();
      sendResponse({ success: true });
      break;
  }
});

async function startBulkDownload() {
  try {
    const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
    
    // Reset state
    downloadState = {
      isActive: true,
      allChats: [],
      currentIndex: 0,
      downloadedFiles: [],
      lastUrl: tab.url,
      startTime: new Date().toISOString()
    };
    
    // Extract all chats first
    console.log('Extracting chat list...');
    const chats = await extractAllChats(tab.id);
    
    if (!chats || chats.length === 0) {
      throw new Error('No chats found. Make sure sidebar is open.');
    }
    
    downloadState.allChats = chats;
    await saveState();
    
    console.log(`Found ${chats.length} chats. Starting download...`);
    
    // Start the download process
    await moveToNextChat(tab.id);
    
    return { success: true, chatCount: chats.length };
    
  } catch (error) {
    console.error('Error starting download:', error);
    downloadState.isActive = false;
    await saveState();
    return { error: error.message };
  }
}

async function resumeDownload() {
  if (!downloadState.isActive || downloadState.currentIndex >= downloadState.allChats.length) {
    return { error: 'No active download to resume' };
  }
  
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  await moveToNextChat(tab.id);
  
  return { success: true, resumedAt: downloadState.currentIndex };
}

async function downloadCurrentChat(tabId) {
  const chat = downloadState.allChats[downloadState.currentIndex];
  console.log(`Downloading chat ${downloadState.currentIndex + 1}/${downloadState.allChats.length}: ${chat.title}`);
  
  try {
    const filename = await downloadChatHTML(tabId, chat, downloadState.currentIndex + 1, downloadState.allChats);
    
    downloadState.downloadedFiles.push({
      ...chat,
      filename: filename,
      index: downloadState.currentIndex + 1,
      downloadedAt: new Date().toISOString()
    });
    
  } catch (error) {
    console.error(`Failed to download ${chat.title}:`, error);
    downloadState.downloadedFiles.push({
      ...chat,
      filename: null,
      index: downloadState.currentIndex + 1,
      error: error.message
    });
  }
  
  downloadState.currentIndex++;
  await saveState();
}

async function moveToNextChat(tabId) {
  if (downloadState.currentIndex >= downloadState.allChats.length) {
    // All done!
    console.log('All downloads complete!');
    await createIndexFile(downloadState.downloadedFiles);
    downloadState.isActive = false;
    await saveState();
    return;
  }
  
  const nextChat = downloadState.allChats[downloadState.currentIndex];
  console.log(`Navigating to chat ${downloadState.currentIndex + 1}: ${nextChat.title}`);
  
  // Navigate to the next chat
  chrome.tabs.update(tabId, { url: nextChat.url });
  // The tab update listener will handle the rest
}

async function extractAllChats(tabId) {
  return new Promise((resolve) => {
    chrome.scripting.executeScript({
      target: { tabId: tabId },
      func: async () => {
        const chats = [];
        const historyDiv = document.getElementById('history');
        
        if (!historyDiv) {
          throw new Error('History sidebar not found');
        }
        
        // Scroll to load all chats
        console.log('Scrolling to load all chats...');
        const scrollContainer = historyDiv.closest('[class*="overflow-y-auto"]') || historyDiv.parentElement;
        
        if (scrollContainer) {
          let lastHeight = scrollContainer.scrollHeight;
          let attempts = 0;
          
          while (attempts < 20) { // More attempts for longer lists
            scrollContainer.scrollTop = scrollContainer.scrollHeight;
            await new Promise(r => setTimeout(r, 1000));
            
            if (scrollContainer.scrollHeight === lastHeight) {
              // Wait one more time to be sure
              await new Promise(r => setTimeout(r, 1000));
              if (scrollContainer.scrollHeight === lastHeight) {
                break;
              }
            }
            lastHeight = scrollContainer.scrollHeight;
            attempts++;
          }
          
          console.log(`Scrolled ${attempts} times`);
        }
        
        // Extract all chat links
        const chatLinks = historyDiv.querySelectorAll('a[href^="/c/"], a[href^="/chat/"]');
        console.log(`Found ${chatLinks.length} chat links`);
        
        chatLinks.forEach((link, index) => {
          const titleSpan = link.querySelector('span[dir="auto"]');
          const title = titleSpan ? titleSpan.textContent.trim() : `Untitled Chat ${index + 1}`;
          const href = link.getAttribute('href');
          
          if (href) {
            chats.push({
              title: title,
              href: href,
              url: new URL(href, window.location.origin).toString()
            });
          }
        });
        
        return chats;
      }
    }, (results) => {
      if (chrome.runtime.lastError) {
        console.error('Script error:', chrome.runtime.lastError);
        resolve([]);
      } else if (results && results[0] && results[0].result) {
        resolve(results[0].result);
      } else {
        resolve([]);
      }
    });
  });
}

async function downloadChatHTML(tabId, chat, index, allChatsList) {
  return new Promise((resolve) => {
    chrome.scripting.executeScript({
      target: { tabId: tabId },
      func: (allChatsData, currentIndex) => {
        let html = document.documentElement.outerHTML;
        
        // Update all chat links to point to local files
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
        conflictAction: 'overwrite'
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
  <title>ChatGPT Export - ${new Date().toLocaleDateString()}</title>
  <style>
    body { font-family: -apple-system, sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px; }
    .stats { background: #f5f5f5; padding: 20px; border-radius: 8px; margin-bottom: 20px; }
    .chat-item { padding: 12px; border-bottom: 1px solid #e5e5e5; display: flex; justify-content: space-between; }
    .chat-item:hover { background: #f9f9f9; }
    a { color: #0066cc; text-decoration: none; }
    a:hover { text-decoration: underline; }
    .success { color: #10a37f; }
    .error { color: #ef4444; }
    .timestamp { color: #666; font-size: 0.9em; }
  </style>
</head>
<body>
  <h1>ChatGPT Export</h1>
  
  <div class="stats">
    <h2>Export Summary</h2>
    <p>Export started: ${downloadState.startTime ? new Date(downloadState.startTime).toLocaleString() : 'Unknown'}</p>
    <p>Total conversations: ${downloadedFiles.length}</p>
    <p>Successfully downloaded: ${downloadedFiles.filter(f => f.filename).length}</p>
    <p>Failed: ${downloadedFiles.filter(f => !f.filename).length}</p>
  </div>
  
  <h2>Conversations</h2>
  <div>
    ${downloadedFiles.map(file => `
      <div class="chat-item">
        <div>
          ${file.filename ? 
            `<a href="${file.filename}">${file.index}. ${file.title}</a>` :
            `<span style="color: #999;">${file.index}. ${file.title}</span>`
          }
        </div>
        <div>
          ${file.filename ? 
            `<span class="success">✓ Downloaded</span>` :
            `<span class="error">✗ ${file.error || 'Failed'}</span>`
          }
          ${file.downloadedAt ? 
            `<span class="timestamp"> at ${new Date(file.downloadedAt).toLocaleTimeString()}</span>` : 
            ''
          }
        </div>
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
    saveAs: false,
    conflictAction: 'overwrite'
  }, () => {
    URL.revokeObjectURL(url);
  });
}