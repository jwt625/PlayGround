// Bulk downloader for ChatGPT conversations

console.log('Bulk downloader popup loaded');

document.addEventListener('DOMContentLoaded', () => {
  const startBtn = document.getElementById('startBtn');
  const statusDiv = document.getElementById('status');
  const progressDiv = document.getElementById('progress');
  const progressText = document.getElementById('progressText');
  const progressBar = document.getElementById('progressBar');
  
  console.log('Bulk downloader initialized');
  console.log('Button found:', !!startBtn);
  
  let allChats = [];
  let downloadedFiles = [];
  
  startBtn.addEventListener('click', async () => {
    console.log('Start button clicked');
    startBtn.disabled = true;
    statusDiv.textContent = 'Starting bulk download...';
    progressDiv.style.display = 'block';
    
    try {
      const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
      
      // Step 1: Extract chat history
      updateProgress(0, 1, 'Extracting chat list...');
      
      const chatList = await extractChatList(tab.id);
      if (!chatList || chatList.length === 0) {
        throw new Error('No chats found. Make sure the sidebar is open.');
      }
      
      allChats = chatList;
      statusDiv.textContent = `Found ${chatList.length} chats. Downloading...`;
      
      // Step 2: Download each chat
      for (let i = 0; i < chatList.length; i++) {
        const chat = chatList[i];
        updateProgress(i + 1, chatList.length, `Downloading: ${chat.title}`);
        
        try {
          // Navigate to the chat
          await navigateToChat(tab.id, chat.url);
          
          // Wait for page to fully load (15 seconds as requested)
          await new Promise(resolve => setTimeout(resolve, 15000));
          
          // Download the HTML
          const filename = await downloadChatHTML(tab.id, chat, i + 1);
          
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
      
      // Step 3: Create an index file
      await createIndexFile(downloadedFiles);
      
      statusDiv.textContent = `Download complete! ${downloadedFiles.filter(f => f.filename).length}/${chatList.length} chats saved.`;
      progressDiv.style.display = 'none';
      
    } catch (error) {
      console.error('Error:', error);
      statusDiv.textContent = `Error: ${error.message}`;
      progressDiv.style.display = 'none';
    }
    
    startBtn.disabled = false;
  });
  
  function updateProgress(current, total, message) {
    const percentage = Math.round((current / total) * 100);
    progressText.textContent = `${message} (${current}/${total})`;
    progressBar.style.width = percentage + '%';
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
            // Scroll to bottom to load all chats
            let lastHeight = scrollContainer.scrollHeight;
            let attempts = 0;
            
            while (attempts < 10) {
              scrollContainer.scrollTop = scrollContainer.scrollHeight;
              await new Promise(r => setTimeout(r, 1000)); // Wait 1 second
              
              if (scrollContainer.scrollHeight === lastHeight) {
                break; // No more content loaded
              }
              lastHeight = scrollContainer.scrollHeight;
              attempts++;
            }
            
            console.log(`Scrolled ${attempts} times to load all chats`);
          }
          
          // Now extract all chat links
          const chatLinks = historyDiv.querySelectorAll('a[href^="/c/"], a[href^="/chat/"]');
          console.log(`Found ${chatLinks.length} chat links`);
          
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
          
          return { chats: chats, count: chats.length };
        }
      }, (results) => {
        if (chrome.runtime.lastError || !results || !results[0]) {
          console.error('Error extracting chats:', chrome.runtime.lastError);
          resolve([]);
        } else {
          const result = results[0].result;
          if (result.error) {
            console.error(result.error);
            resolve([]);
          } else {
            console.log(`Successfully extracted ${result.count} chats`);
            resolve(result.chats);
          }
        }
      });
    });
  }
  
  async function navigateToChat(tabId, url) {
    return new Promise((resolve) => {
      chrome.tabs.update(tabId, { url: url }, () => {
        // Wait for navigation to complete
        chrome.tabs.onUpdated.addListener(function listener(updatedTabId, changeInfo) {
          if (updatedTabId === tabId && changeInfo.status === 'complete') {
            chrome.tabs.onUpdated.removeListener(listener);
            resolve();
          }
        });
      });
    });
  }
  
  async function downloadChatHTML(tabId, chat, index) {
    return new Promise((resolve) => {
      chrome.scripting.executeScript({
        target: { tabId: tabId },
        func: (allChatsData, currentIndex) => {
          let html = document.documentElement.outerHTML;
          
          // Update all chat links in the sidebar to point to local files
          allChatsData.forEach((chatData, idx) => {
            const safeTitle = chatData.title.replace(/[^a-z0-9]/gi, '_').toLowerCase().substring(0, 50);
            const localFilename = `${(idx + 1).toString().padStart(3, '0')}_${safeTitle}.html`;
            
            // Replace the href in the HTML
            // Looking for patterns like href="/c/xxx" or href="/chat/xxx"
            const oldHref = chatData.href;
            const escapedOldHref = oldHref.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
            
            // Replace in links
            html = html.replace(
              new RegExp(`href="${escapedOldHref}"`, 'g'),
              `href="${localFilename}"`
            );
            
            // Also update any active states or current page indicators
            if (idx === currentIndex - 1) {
              // This is the current page, might need special handling
              html = html.replace(
                new RegExp(`href="${localFilename}"([^>]*?)>`, 'g'),
                `href="${localFilename}" style="background-color: #f3f4f6;"$1>`
              );
            }
          });
          
          return html;
        },
        args: [allChats, index]
      }, (results) => {
        if (chrome.runtime.lastError || !results || !results[0]) {
          resolve(null);
          return;
        }
        
        const modifiedHtml = results[0].result;
        const blob = new Blob([modifiedHtml], { type: 'text/html' });
        const url = URL.createObjectURL(blob);
        
        // Create filename without timestamp for cleaner names
        const safeTitle = chat.title.replace(/[^a-z0-9]/gi, '_').toLowerCase().substring(0, 50);
        const filenameOnly = `${index.toString().padStart(3, '0')}_${safeTitle}.html`;
        const filename = `chatgpt_export/${filenameOnly}`;
        
        chrome.downloads.download({
          url: url,
          filename: filename,
          saveAs: false,
          conflictAction: 'uniquify'
        }, (downloadId) => {
          URL.revokeObjectURL(url);
          resolve(filenameOnly); // Return just the filename, not the full path
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
      background: #f5f5f5;
    }
    h1 {
      color: #202123;
      margin-bottom: 30px;
    }
    .chat-list {
      background: white;
      border-radius: 8px;
      padding: 20px;
      box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    .chat-item {
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding: 15px;
      border-bottom: 1px solid #e5e5e5;
    }
    .chat-item:last-child {
      border-bottom: none;
    }
    .chat-title {
      font-weight: 500;
      color: #202123;
      text-decoration: none;
      flex: 1;
    }
    .chat-title:hover {
      color: #10a37f;
    }
    .chat-status {
      font-size: 14px;
      color: #6e6e80;
      margin-left: 20px;
    }
    .success {
      color: #10a37f;
    }
    .error {
      color: #ef4444;
    }
    .metadata {
      margin-bottom: 20px;
      color: #6e6e80;
    }
  </style>
</head>
<body>
  <h1>ChatGPT Conversation Export</h1>
  
  <div class="metadata">
    <p>Exported on: ${new Date().toLocaleString()}</p>
    <p>Total conversations: ${downloadedFiles.length}</p>
    <p>Successfully downloaded: ${downloadedFiles.filter(f => f.filename).length}</p>
  </div>
  
  <div class="chat-list">
    ${downloadedFiles.map(file => `
      <div class="chat-item">
        ${file.filename ? 
          `<a href="${file.filename}" class="chat-title">${file.index}. ${file.title}</a>
           <span class="chat-status success">✓ Downloaded</span>` :
          `<span class="chat-title">${file.index}. ${file.title}</span>
           <span class="chat-status error">✗ Failed: ${file.error || 'Unknown error'}</span>`
        }
      </div>
    `).join('')}
  </div>
  
  <script>
    console.log('Index page loaded. All links have been updated to local files.');
  </script>
</body>
</html>
    `;
    
    const indexBlob = new Blob([indexHTML], { type: 'text/html' });
    const indexUrl = URL.createObjectURL(indexBlob);
    
    chrome.downloads.download({
      url: indexUrl,
      filename: 'chatgpt_export/index.html',
      saveAs: false
    }, () => {
      URL.revokeObjectURL(indexUrl);
    });
  }
});