// Simple script that just grabs and downloads HTML

document.getElementById('grabBtn').addEventListener('click', async () => {
  const status = document.getElementById('status');
  status.textContent = 'Grabbing HTML...';
  
  try {
    const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
    
    chrome.scripting.executeScript({
      target: { tabId: tab.id },
      func: () => document.documentElement.outerHTML
    }, (results) => {
      if (chrome.runtime.lastError) {
        status.textContent = 'Error: ' + chrome.runtime.lastError.message;
        return;
      }
      
      const html = results[0].result;
      const blob = new Blob([html], { type: 'text/html' });
      const url = URL.createObjectURL(blob);
      const timestamp = new Date().getTime();
      
      chrome.downloads.download({
        url: url,
        filename: `chatgpt_page_${timestamp}.html`,
        saveAs: true
      }, () => {
        URL.revokeObjectURL(url);
        status.textContent = 'Downloaded!';
      });
    });
  } catch (error) {
    status.textContent = 'Error: ' + error.message;
  }
});

document.getElementById('grabAllBtn').addEventListener('click', async () => {
  const status = document.getElementById('status');
  status.textContent = 'Starting bulk grab...';
  
  try {
    const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
    
    // First get all chat links
    chrome.scripting.executeScript({
      target: { tabId: tab.id },
      func: () => {
        const links = document.querySelectorAll('#history a[href^="/c/"], #history a[href^="/chat/"]');
        const chats = [];
        links.forEach((link, index) => {
          const titleEl = link.querySelector('span[dir="auto"]');
          chats.push({
            title: titleEl ? titleEl.textContent : `Chat ${index + 1}`,
            url: link.href
          });
        });
        return chats;
      }
    }, async (results) => {
      if (chrome.runtime.lastError || !results[0]) {
        status.textContent = 'Error getting chat list';
        return;
      }
      
      const chats = results[0].result;
      status.textContent = `Found ${chats.length} chats. Downloading...`;
      
      // Download each one
      for (let i = 0; i < chats.length; i++) {
        const chat = chats[i];
        status.textContent = `Downloading ${i + 1}/${chats.length}...`;
        
        // Navigate to the chat
        await chrome.tabs.update(tab.id, { url: chat.url });
        
        // Wait for page to load
        await new Promise(resolve => {
          chrome.tabs.onUpdated.addListener(function listener(tabId, info) {
            if (tabId === tab.id && info.status === 'complete') {
              chrome.tabs.onUpdated.removeListener(listener);
              resolve();
            }
          });
        });
        
        // Wait extra for content
        await new Promise(resolve => setTimeout(resolve, 5000));
        
        // Grab and download
        await new Promise(resolve => {
          chrome.scripting.executeScript({
            target: { tabId: tab.id },
            func: () => document.documentElement.outerHTML
          }, (results) => {
            if (results && results[0]) {
              const html = results[0].result;
              const safeTitle = chat.title.replace(/[^a-z0-9]/gi, '_').substring(0, 40);
              const blob = new Blob([html], { type: 'text/html' });
              const url = URL.createObjectURL(blob);
              
              chrome.downloads.download({
                url: url,
                filename: `chatgpt_export/${(i + 1).toString().padStart(3, '0')}_${safeTitle}.html`,
                saveAs: false
              }, () => {
                URL.revokeObjectURL(url);
                resolve();
              });
            } else {
              resolve();
            }
          });
        });
      }
      
      status.textContent = `Done! Downloaded ${chats.length} chats.`;
    });
  } catch (error) {
    status.textContent = 'Error: ' + error.message;
  }
});