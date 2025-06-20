// Simple background script - just download all chats

let isDownloading = false;
let allChats = [];
let currentIndex = 0;

// Message handlers
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  console.log('Background received:', request.action);
  
  if (request.action === 'startDownload') {
    if (isDownloading) {
      sendResponse({ error: 'Already downloading' });
    } else {
      startDownload();
      sendResponse({ success: true });
    }
  } else if (request.action === 'getStatus') {
    sendResponse({
      isDownloading,
      current: currentIndex,
      total: allChats.length
    });
  }
  
  return true;
});

async function startDownload() {
  console.log('Starting download...');
  isDownloading = true;
  currentIndex = 0;
  
  try {
    // Get the active tab
    const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
    
    // Extract all chat URLs first
    console.log('Getting chat list...');
    allChats = await getChatList(tab.id);
    console.log(`Found ${allChats.length} chats`);
    
    if (allChats.length === 0) {
      throw new Error('No chats found');
    }
    
    // Download each one
    for (let i = 0; i < allChats.length; i++) {
      currentIndex = i;
      const chat = allChats[i];
      console.log(`Downloading ${i + 1}/${allChats.length}: ${chat.title}`);
      
      // Navigate to the chat
      await chrome.tabs.update(tab.id, { url: chat.url });
      
      // Wait for page to load
      await waitForTabLoad(tab.id);
      
      // Extra wait for content to render
      await new Promise(resolve => setTimeout(resolve, 15000));
      
      // Download the page
      await downloadPage(tab.id, chat, i + 1);
    }
    
    console.log('All downloads complete!');
    
  } catch (error) {
    console.error('Download error:', error);
  } finally {
    isDownloading = false;
  }
}

function getChatList(tabId) {
  return new Promise((resolve) => {
    chrome.scripting.executeScript({
      target: { tabId },
      func: () => {
        const chats = [];
        const links = document.querySelectorAll('#history a[href^="/c/"], #history a[href^="/chat/"]');
        
        links.forEach(link => {
          const titleEl = link.querySelector('span[dir="auto"]');
          const title = titleEl ? titleEl.textContent.trim() : 'Untitled';
          const href = link.getAttribute('href');
          
          if (href) {
            chats.push({
              title,
              href,
              url: new URL(href, window.location.origin).toString()
            });
          }
        });
        
        return chats;
      }
    }, (results) => {
      resolve(results?.[0]?.result || []);
    });
  });
}

function waitForTabLoad(tabId) {
  return new Promise((resolve) => {
    chrome.tabs.onUpdated.addListener(function listener(id, info) {
      if (id === tabId && info.status === 'complete') {
        chrome.tabs.onUpdated.removeListener(listener);
        resolve();
      }
    });
  });
}

function downloadPage(tabId, chat, index) {
  return new Promise((resolve) => {
    chrome.scripting.executeScript({
      target: { tabId },
      func: () => document.documentElement.outerHTML
    }, (results) => {
      if (results?.[0]?.result) {
        const html = results[0].result;
        
        // Create safe filename
        const safeTitle = chat.title.replace(/[^a-z0-9]/gi, '_').substring(0, 40);
        const filename = `chatgpt_export/${index.toString().padStart(3, '0')}_${safeTitle}.html`;
        
        // Convert to data URL (works in service worker)
        const dataUrl = 'data:text/html;charset=utf-8,' + encodeURIComponent(html);
        
        // Download
        chrome.downloads.download({
          url: dataUrl,
          filename: filename,
          saveAs: false
        }, () => {
          console.log(`Downloaded: ${filename}`);
          resolve();
        });
      } else {
        resolve();
      }
    });
  });
}