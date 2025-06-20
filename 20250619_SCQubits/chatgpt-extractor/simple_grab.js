// Simple script that just grabs and downloads HTML

let chatList = [];
let currentIndex = -1;

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
      const timestamp = new Date().getTime();
      
      // Get the title from the page
      chrome.scripting.executeScript({
        target: { tabId: tab.id },
        func: () => {
          const titleEl = document.querySelector('h1');
          return titleEl ? titleEl.textContent : 'Untitled';
        }
      }, (titleResults) => {
        const title = titleResults[0]?.result || 'Untitled';
        const safeTitle = title.replace(/[^a-z0-9]/gi, '_').substring(0, 50);
        
        // Create blob and URL
        const blob = new Blob([html], { type: 'text/html' });
        const url = URL.createObjectURL(blob);
        
        const intendedFilename = `chatgpt_${safeTitle}_${timestamp}.html`;
        
        chrome.downloads.download({
          url: url,
          filename: intendedFilename,
          saveAs: false
        }, (downloadId) => {
          if (chrome.runtime.lastError) {
            console.error('Download error:', chrome.runtime.lastError);
            status.textContent = 'Error: ' + chrome.runtime.lastError.message;
          } else {
            // Query the actual filename
            chrome.downloads.search({id: downloadId}, (downloads) => {
              if (downloads && downloads[0]) {
                const actualFilename = downloads[0].filename.split('/').pop();
                console.log(`Download mapping: ${actualFilename} -> ${intendedFilename}`);
                
                // Store mapping
                chrome.storage.local.get(['downloadMappings'], (result) => {
                  const mappings = result.downloadMappings || {};
                  mappings[actualFilename] = intendedFilename;
                  chrome.storage.local.set({downloadMappings: mappings});
                });
                
                status.textContent = `Downloaded as: ${actualFilename}`;
              }
            });
          }
          setTimeout(() => URL.revokeObjectURL(url), 1000);
        });
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
      
      // Reverse to start with oldest chat (bottom of list)
      chats.reverse();
      
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
        
        // Wait extra for content - need more time for full render
        await new Promise(resolve => setTimeout(resolve, 10000));
        
        // Grab and download with updated links
        await new Promise(resolve => {
          chrome.scripting.executeScript({
            target: { tabId: tab.id },
            func: (allChats) => {
              // Get the HTML
              let html = document.documentElement.outerHTML;
              
              // Update all chat links in the sidebar to point to local files
              allChats.forEach((chat, idx) => {
                const timestamp = new Date().toISOString().replace(/[:.]/g, '-').substring(0, 19);
                const safeTitle = chat.title.replace(/[^a-z0-9\s]/gi, '_').trim().replace(/\s+/g, '_').substring(0, 60);
                const localFile = `${timestamp}_${safeTitle}.html`;
                // Replace the href in the sidebar
                const pattern = new RegExp(`href="${chat.url.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}"`, 'g');
                html = html.replace(pattern, `href="${localFile}"`);
              });
              
              return html;
            },
            args: [chats]
          }, (results) => {
            if (results && results[0]) {
              const html = results[0].result;
              const timestamp = new Date().toISOString().replace(/[:.]/g, '-').substring(0, 19);
              const safeTitle = chat.title.replace(/[^a-z0-9\s]/gi, '_').trim().replace(/\s+/g, '_').substring(0, 60);
              const filename = `${timestamp}_${safeTitle}.html`;
              
              // Download and track mapping
              const blob = new Blob([html], { type: 'text/html' });
              const url = URL.createObjectURL(blob);
              
              chrome.downloads.download({
                url: url,
                filename: filename,
                saveAs: false
              }, (downloadId) => {
                if (downloadId) {
                  // Query actual filename after short delay
                  setTimeout(() => {
                    chrome.downloads.search({id: downloadId}, (downloads) => {
                      if (downloads && downloads[0]) {
                        const actualFilename = downloads[0].filename.split('/').pop();
                        console.log(`Mapping: ${actualFilename} -> ${filename}`);
                        
                        // Store mapping
                        chrome.storage.local.get(['bulkMappings'], (result) => {
                          const mappings = result.bulkMappings || {};
                          mappings[actualFilename] = {
                            intended: filename,
                            title: chat.title,
                            index: i
                          };
                          chrome.storage.local.set({bulkMappings: mappings});
                        });
                      }
                    });
                  }, 500);
                }
                setTimeout(() => URL.revokeObjectURL(url), 2000);
                resolve();
              });
            } else {
              resolve();
            }
          });
        });
      }
      
      status.textContent = `Done! Downloaded ${chats.length} chats. Creating mapping file...`;
      
      // Wait a bit then create mapping file
      setTimeout(() => {
        chrome.storage.local.get(['bulkMappings'], (result) => {
          const mappings = result.bulkMappings || {};
          
          // Create CSV mapping file
          let csv = 'Temp Filename,Intended Filename,Title\n';
          Object.entries(mappings).forEach(([tempName, info]) => {
            csv += `"${tempName}","${info.intended}","${info.title}"\n`;
          });
          
          const blob = new Blob([csv], { type: 'text/csv' });
          const url = URL.createObjectURL(blob);
          
          chrome.downloads.download({
            url: url,
            filename: 'chatgpt_download_mapping.csv',
            saveAs: false
          }, () => {
            URL.revokeObjectURL(url);
            status.textContent = `Done! Downloaded ${chats.length} chats + mapping file.`;
            
            // Clear the mappings
            chrome.storage.local.remove(['bulkMappings']);
          });
        });
      }, 2000);
    });
  } catch (error) {
    status.textContent = 'Error: ' + error.message;
  }
});

// Load chat list
document.getElementById('loadChatsBtn').addEventListener('click', async () => {
  const status = document.getElementById('status');
  status.textContent = 'Loading chat list...';
  
  try {
    const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
    
    chrome.scripting.executeScript({
      target: { tabId: tab.id },
      func: () => {
        const links = document.querySelectorAll('#history a[href^="/c/"], #history a[href^="/chat/"]');
        const chats = [];
        links.forEach((link, index) => {
          const titleEl = link.querySelector('span[dir="auto"]');
          chats.push({
            title: titleEl ? titleEl.textContent : `Chat ${index + 1}`,
            url: link.href,
            href: link.getAttribute('href')
          });
        });
        return chats;
      }
    }, (results) => {
      if (chrome.runtime.lastError || !results[0]) {
        status.textContent = 'Error getting chat list';
        return;
      }
      
      chatList = results[0].result.reverse(); // Start with oldest
      currentIndex = -1;
      
      // Save chat list to file
      const timestamp = new Date().toISOString().replace(/[:.]/g, '-').substring(0, 19);
      let csvContent = 'Index,Title,URL,Filename\n';
      chatList.forEach((chat, idx) => {
        const safeTitle = chat.title.replace(/[^a-z0-9\s]/gi, '_').trim().replace(/\s+/g, '_').substring(0, 60);
        const filename = `${(idx + 1).toString().padStart(3, '0')}_${timestamp}_${safeTitle}.html`;
        csvContent += `${idx + 1},"${chat.title.replace(/"/g, '""')}","${chat.url}","${filename}"\n`;
      });
      
      const blob = new Blob([csvContent], { type: 'text/csv' });
      const url = URL.createObjectURL(blob);
      
      chrome.downloads.download({
        url: url,
        filename: `chatgpt_chat_list_${timestamp}.csv`,
        saveAs: false
      }, () => {
        URL.revokeObjectURL(url);
      });
      
      document.getElementById('navigation').style.display = 'block';
      updateNavigation();
      status.textContent = `Loaded ${chatList.length} chats. Chat list saved to downloads. Use arrows to navigate.`;
    });
  } catch (error) {
    status.textContent = 'Error: ' + error.message;
  }
});

// Navigation functions
function updateNavigation() {
  const prevBtn = document.getElementById('prevBtn');
  const nextBtn = document.getElementById('nextBtn');
  const chatInfo = document.getElementById('chatInfo');
  
  prevBtn.disabled = currentIndex <= 0;
  nextBtn.disabled = currentIndex >= chatList.length - 1;
  
  if (currentIndex >= 0 && currentIndex < chatList.length) {
    const chat = chatList[currentIndex];
    chatInfo.textContent = `${currentIndex + 1}/${chatList.length}: ${chat.title}`;
    
    // Save current index to storage for keyboard shortcut
    chrome.storage.local.set({ currentIndex: currentIndex, chatList: chatList });
    
    // Check if we're on the right page
    chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
      if (tabs[0] && tabs[0].url.includes(chat.href)) {
        document.getElementById('status').textContent = 'On correct page. Wait for render, then Download.';
        document.getElementById('downloadCurrentBtn').style.background = '#10a37f';
      } else {
        document.getElementById('downloadCurrentBtn').style.background = '#999';
      }
    });
  } else {
    chatInfo.textContent = 'Click Next to start';
  }
}

// Previous button
document.getElementById('prevBtn').addEventListener('click', async () => {
  if (currentIndex > 0) {
    currentIndex--;
    await navigateToChat(currentIndex);
  }
});

// Next button
document.getElementById('nextBtn').addEventListener('click', async () => {
  if (currentIndex < chatList.length - 1) {
    currentIndex++;
    await navigateToChat(currentIndex);
  }
});

// Show which chat to navigate to
async function navigateToChat(index) {
  const status = document.getElementById('status');
  const chat = chatList[index];
  
  // Copy URL to clipboard
  try {
    await navigator.clipboard.writeText(chat.url);
    status.textContent = `Please manually navigate to: "${chat.title}" (URL copied to clipboard)`;
  } catch (err) {
    status.textContent = `Please manually navigate to: "${chat.title}"`;
  }
  
  updateNavigation();
}

// Download current button
document.getElementById('downloadCurrentBtn').addEventListener('click', async () => {
  if (currentIndex < 0 || currentIndex >= chatList.length) {
    document.getElementById('status').textContent = 'No chat selected';
    return;
  }
  
  const status = document.getElementById('status');
  const chat = chatList[currentIndex];
  status.textContent = 'Downloading...';
  
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
      const timestamp = new Date().toISOString().replace(/[:.]/g, '-').substring(0, 19);
      const safeTitle = chat.title.replace(/[^a-z0-9\s]/gi, '_').trim().replace(/\s+/g, '_').substring(0, 60);
      const filename = `${(currentIndex + 1).toString().padStart(3, '0')}_${timestamp}_${safeTitle}.html`;
      
      // Create blob and download
      const blob = new Blob([html], { type: 'text/html' });
      const url = URL.createObjectURL(blob);
      
      chrome.downloads.download({
        url: url,
        filename: filename,
        saveAs: false
      }, (downloadId) => {
        if (chrome.runtime.lastError) {
          status.textContent = 'Error: ' + chrome.runtime.lastError.message;
        } else {
          status.textContent = `Downloaded: ${filename}`;
        }
        setTimeout(() => URL.revokeObjectURL(url), 1000);
      });
    });
  } catch (error) {
    status.textContent = 'Error: ' + error.message;
  }
});