// Popup script for ChatGPT Conversation Extractor
console.log('Popup script loading...');

document.addEventListener('DOMContentLoaded', () => {
  console.log('DOM Content Loaded');
  const extractBtn = document.getElementById('extractBtn');
  const extractAllBtn = document.getElementById('extractAllBtn');
  const statusDiv = document.getElementById('status');
  const progressDiv = document.getElementById('progress');
  const progressText = document.getElementById('progressText');
  const progressBar = document.getElementById('progressBar');
  const includeCodeCheckbox = document.getElementById('includeCode');
  const includeTimestampsCheckbox = document.getElementById('includeTimestamps');
  const darkModeCheckbox = document.getElementById('darkMode');
  
  console.log('Elements found:', {
    extractBtn: !!extractBtn,
    extractAllBtn: !!extractAllBtn,
    statusDiv: !!statusDiv,
    progressDiv: !!progressDiv
  });
  
  // Load saved preferences
  chrome.storage.local.get(['includeCode', 'includeTimestamps', 'darkMode'], (result) => {
    includeCodeCheckbox.checked = result.includeCode !== false;
    includeTimestampsCheckbox.checked = result.includeTimestamps !== false;
    darkModeCheckbox.checked = result.darkMode === true;
  });
  
  // Save preferences when changed
  includeCodeCheckbox.addEventListener('change', () => {
    chrome.storage.local.set({ includeCode: includeCodeCheckbox.checked });
  });
  
  includeTimestampsCheckbox.addEventListener('change', () => {
    chrome.storage.local.set({ includeTimestamps: includeTimestampsCheckbox.checked });
  });
  
  darkModeCheckbox.addEventListener('change', () => {
    chrome.storage.local.set({ darkMode: darkModeCheckbox.checked });
  });
  
  extractBtn.addEventListener('click', async () => {
    console.log('Extract button clicked');
    extractBtn.disabled = true;
    showStatus('Extracting conversation...', 'info');
    
    try {
      // Get the active tab
      const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
      console.log('Active tab:', tab);
      
      // Check if we're on a ChatGPT page
      if (!tab.url.includes('chatgpt.com') && !tab.url.includes('chat.openai.com')) {
        showStatus('Please navigate to a ChatGPT conversation first.', 'error');
        extractBtn.disabled = false;
        return;
      }
      
      // First try to inject the content script manually
      console.log('Injecting content script...');
      chrome.scripting.executeScript({
        target: { tabId: tab.id },
        files: ['content.js']
      }, () => {
        if (chrome.runtime.lastError) {
          console.error('Failed to inject content script:', chrome.runtime.lastError);
        }
        
        // Wait a bit for the script to initialize
        setTimeout(() => {
          // Send message to content script
          console.log('Sending message to content script...');
          chrome.tabs.sendMessage(tab.id, { action: 'extractConversation' }, (response) => {
            console.log('Response from content script:', response);
            console.log('Chrome runtime last error:', chrome.runtime.lastError);
            
            if (chrome.runtime.lastError) {
              console.error('Chrome runtime error:', chrome.runtime.lastError);
              showStatus('Error: ' + chrome.runtime.lastError.message + '. Try refreshing the page.', 'error');
              extractBtn.disabled = false;
              return;
            }
        
        if (!response || !response.success) {
          showStatus('Failed to extract conversation. Please refresh the page and try again.', 'error');
          extractBtn.disabled = false;
          return;
        }
        
        // Generate HTML with the extracted data
        const html = generateHTML(response.data, {
          includeCode: includeCodeCheckbox.checked,
          includeTimestamps: includeTimestampsCheckbox.checked,
          darkMode: darkModeCheckbox.checked
        });
        
        // Create blob and download
        const blob = new Blob([html], { type: 'text/html' });
        const url = URL.createObjectURL(blob);
        
        // Generate filename
        const date = new Date().toISOString().split('T')[0];
        const cleanTitle = response.data.title.replace(/[^a-z0-9]/gi, '_').toLowerCase();
        const filename = `chatgpt_${cleanTitle}_${date}.html`;
        
        // Download the file
        chrome.downloads.download({
          url: url,
          filename: filename,
          saveAs: true
        }, (downloadId) => {
          if (chrome.runtime.lastError) {
            showStatus('Download failed: ' + chrome.runtime.lastError.message, 'error');
          } else {
            showStatus('Conversation extracted successfully!', 'success');
          }
          extractBtn.disabled = false;
          URL.revokeObjectURL(url);
        });
      });
        }, 500); // Wait 500ms for content script to initialize
      });
    } catch (error) {
      showStatus('Error: ' + error.message, 'error');
      extractBtn.disabled = false;
    }
  });
  
  function showStatus(message, type) {
    statusDiv.textContent = message;
    statusDiv.className = type;
    
    if (type === 'success') {
      setTimeout(() => {
        statusDiv.textContent = '';
        statusDiv.className = '';
      }, 3000);
    }
  }
  
  // Export All functionality
  extractAllBtn.addEventListener('click', async () => {
    console.log('Export All button clicked');
    extractBtn.disabled = true;
    extractAllBtn.disabled = true;
    progressDiv.style.display = 'block';
    showStatus('Starting bulk export...', 'info');
    
    try {
      const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
      
      if (!tab.url.includes('chatgpt.com') && !tab.url.includes('chat.openai.com')) {
        showStatus('Please navigate to ChatGPT first.', 'error');
        extractBtn.disabled = false;
        extractAllBtn.disabled = false;
        progressDiv.style.display = 'none';
        return;
      }
      
      // First, get the chat history
      chrome.tabs.sendMessage(tab.id, { action: 'extractChatHistory' }, async (response) => {
        if (chrome.runtime.lastError || !response || !response.success) {
          showStatus('Failed to get chat history. Make sure the sidebar is open.', 'error');
          extractBtn.disabled = false;
          extractAllBtn.disabled = false;
          progressDiv.style.display = 'none';
          return;
        }
        
        const chatHistory = response.data;
        if (chatHistory.length === 0) {
          showStatus('No chats found in history.', 'error');
          extractBtn.disabled = false;
          extractAllBtn.disabled = false;
          progressDiv.style.display = 'none';
          return;
        }
        
        showStatus(`Found ${chatHistory.length} chats. Starting extraction...`, 'info');
        
        const allConversations = [];
        const failedChats = [];
        
        // Process each chat
        for (let i = 0; i < chatHistory.length; i++) {
          const chat = chatHistory[i];
          updateProgress(i, chatHistory.length, `Extracting: ${chat.title}`);
          
          try {
            // Navigate to the chat
            await navigateToChat(tab.id, chat.url);
            
            // Wait for page to load
            await new Promise(resolve => setTimeout(resolve, 2000));
            
            // Extract the conversation
            const conversation = await extractConversationFromTab(tab.id);
            if (conversation) {
              allConversations.push({
                ...conversation,
                chatInfo: chat
              });
            } else {
              failedChats.push(chat.title);
            }
          } catch (error) {
            console.error(`Failed to extract ${chat.title}:`, error);
            failedChats.push(chat.title);
          }
        }
        
        // Create index and download all conversations
        if (allConversations.length > 0) {
          await downloadAllConversations(allConversations, chatHistory, {
            includeCode: includeCodeCheckbox.checked,
            includeTimestamps: includeTimestampsCheckbox.checked,
            darkMode: darkModeCheckbox.checked
          });
          
          if (failedChats.length > 0) {
            showStatus(`Exported ${allConversations.length} chats. Failed: ${failedChats.length}`, 'info');
          } else {
            showStatus(`Successfully exported all ${allConversations.length} chats!`, 'success');
          }
        } else {
          showStatus('No conversations could be extracted.', 'error');
        }
        
        extractBtn.disabled = false;
        extractAllBtn.disabled = false;
        progressDiv.style.display = 'none';
      });
    } catch (error) {
      showStatus('Error: ' + error.message, 'error');
      extractBtn.disabled = false;
      extractAllBtn.disabled = false;
      progressDiv.style.display = 'none';
    }
  });
  
  function updateProgress(current, total, message) {
    const percentage = Math.round((current / total) * 100);
    progressText.textContent = `${message} (${current}/${total})`;
    progressBar.style.width = percentage + '%';
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
  
  async function extractConversationFromTab(tabId) {
    return new Promise((resolve) => {
      chrome.tabs.sendMessage(tabId, { action: 'extractConversation' }, (response) => {
        if (chrome.runtime.lastError || !response || !response.success) {
          resolve(null);
        } else {
          resolve(response.data);
        }
      });
    });
  }
  
  async function downloadAllConversations(conversations, chatHistory, options) {
    const date = new Date().toISOString().split('T')[0];
    const timestamp = new Date().getTime();
    const folderName = `chatgpt_export_${date}_${timestamp}`;
    
    // Create index HTML
    const indexHtml = generateIndexHTML(chatHistory, conversations, options);
    
    // Download index
    const indexBlob = new Blob([indexHtml], { type: 'text/html' });
    const indexUrl = URL.createObjectURL(indexBlob);
    
    await new Promise((resolve) => {
      chrome.downloads.download({
        url: indexUrl,
        filename: `${folderName}/index.html`,
        saveAs: false
      }, () => {
        URL.revokeObjectURL(indexUrl);
        resolve();
      });
    });
    
    // Download each conversation
    for (let i = 0; i < conversations.length; i++) {
      const conv = conversations[i];
      const html = generateHTML(conv, options);
      const blob = new Blob([html], { type: 'text/html' });
      const url = URL.createObjectURL(blob);
      
      const safeTitle = conv.chatInfo.title.replace(/[^a-z0-9]/gi, '_').toLowerCase();
      const filename = `${folderName}/conversations/${i + 1}_${safeTitle}.html`;
      
      await new Promise((resolve) => {
        chrome.downloads.download({
          url: url,
          filename: filename,
          saveAs: false
        }, () => {
          URL.revokeObjectURL(url);
          resolve();
        });
      });
    }
  }
});

function generateHTML(conversationData, options) {
  const { title, messages, extractedAt, url } = conversationData;
  const { includeCode, includeTimestamps, darkMode } = options;
  
  const styles = `
    <style>
      body {
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 'Ubuntu', sans-serif;
        line-height: 1.6;
        max-width: 800px;
        margin: 0 auto;
        padding: 20px;
        ${darkMode ? `
          background-color: #202123;
          color: #ececf1;
        ` : `
          background-color: #ffffff;
          color: #202123;
        `}
      }
      
      .header {
        border-bottom: 2px solid ${darkMode ? '#444654' : '#e5e5e5'};
        padding-bottom: 20px;
        margin-bottom: 30px;
      }
      
      h1 {
        margin: 0 0 10px 0;
        color: ${darkMode ? '#ececf1' : '#202123'};
      }
      
      .meta {
        font-size: 14px;
        color: ${darkMode ? '#c5c5d2' : '#6e6e80'};
      }
      
      .message {
        margin-bottom: 30px;
        padding: 20px;
        border-radius: 8px;
        ${darkMode ? `
          background-color: #2a2b2d;
          border: 1px solid #444654;
        ` : `
          background-color: #f7f7f8;
          border: 1px solid #e5e5e5;
        `}
      }
      
      .message.user {
        ${darkMode ? `
          background-color: #343541;
        ` : `
          background-color: #ffffff;
        `}
      }
      
      .message-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 10px;
        font-weight: bold;
      }
      
      .role {
        color: ${darkMode ? '#ececf1' : '#202123'};
      }
      
      .timestamp {
        font-size: 12px;
        color: ${darkMode ? '#8e8ea0' : '#6e6e80'};
        font-weight: normal;
      }
      
      .message-content {
        white-space: pre-wrap;
        word-wrap: break-word;
      }
      
      pre {
        background-color: ${darkMode ? '#1e1e1e' : '#f6f8fa'};
        padding: 16px;
        border-radius: 6px;
        overflow-x: auto;
        margin: 10px 0;
        border: 1px solid ${darkMode ? '#444654' : '#e5e5e5'};
      }
      
      code {
        font-family: 'Consolas', 'Monaco', 'Courier New', monospace;
        font-size: 14px;
        color: ${darkMode ? '#ececf1' : '#202123'};
      }
      
      .code-header {
        font-size: 12px;
        color: ${darkMode ? '#8e8ea0' : '#6e6e80'};
        margin-bottom: 8px;
      }
      
      .footer {
        margin-top: 50px;
        padding-top: 20px;
        border-top: 2px solid ${darkMode ? '#444654' : '#e5e5e5'};
        text-align: center;
        font-size: 12px;
        color: ${darkMode ? '#8e8ea0' : '#6e6e80'};
      }
    </style>
  `;
  
  let messagesHTML = '';
  
  messages.forEach(message => {
    const roleDisplay = message.role === 'user' ? 'You' : 'ChatGPT';
    const timestamp = includeTimestamps && message.timestamp ? 
      new Date(message.timestamp).toLocaleString() : '';
    
    let messageContent = escapeHTML(message.content);
    
    // Add code blocks if enabled
    if (includeCode && message.codeBlocks && message.codeBlocks.length > 0) {
      message.codeBlocks.forEach(block => {
        const codeHTML = `
          <div class="code-block">
            <div class="code-header">Language: ${escapeHTML(block.language)}</div>
            <pre><code>${escapeHTML(block.code)}</code></pre>
          </div>
        `;
        messageContent += '\n\n' + codeHTML;
      });
    }
    
    messagesHTML += `
      <div class="message ${message.role}">
        <div class="message-header">
          <span class="role">${roleDisplay}</span>
          ${timestamp ? `<span class="timestamp">${timestamp}</span>` : ''}
        </div>
        <div class="message-content">${messageContent}</div>
      </div>
    `;
  });
  
  return `
    <!DOCTYPE html>
    <html>
    <head>
      <meta charset="UTF-8">
      <meta name="viewport" content="width=device-width, initial-scale=1.0">
      <title>${escapeHTML(title)}</title>
      ${styles}
    </head>
    <body>
      <div class="header">
        <h1>${escapeHTML(title)}</h1>
        <div class="meta">
          <div>Extracted on: ${new Date(extractedAt).toLocaleString()}</div>
          <div>Source: <a href="${escapeHTML(url)}" target="_blank">${escapeHTML(url)}</a></div>
        </div>
      </div>
      
      <div class="messages">
        ${messagesHTML}
      </div>
      
      <div class="footer">
        Extracted by ChatGPT Conversation Extractor
      </div>
    </body>
    </html>
  `;
}

function escapeHTML(str) {
  if (!str) return '';
  const div = document.createElement('div');
  div.textContent = str;
  return div.innerHTML;
}

function generateIndexHTML(chatHistory, conversations, options) {
  const { darkMode } = options;
  const date = new Date().toISOString();
  
  const styles = `
    <style>
      body {
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 'Ubuntu', sans-serif;
        line-height: 1.6;
        max-width: 1200px;
        margin: 0 auto;
        padding: 20px;
        ${darkMode ? `
          background-color: #202123;
          color: #ececf1;
        ` : `
          background-color: #ffffff;
          color: #202123;
        `}
      }
      
      h1 {
        margin-bottom: 10px;
        color: ${darkMode ? '#ececf1' : '#202123'};
      }
      
      .meta {
        font-size: 14px;
        color: ${darkMode ? '#c5c5d2' : '#6e6e80'};
        margin-bottom: 30px;
      }
      
      .stats {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 20px;
        margin-bottom: 40px;
      }
      
      .stat-card {
        padding: 20px;
        border-radius: 8px;
        text-align: center;
        ${darkMode ? `
          background-color: #2a2b2d;
          border: 1px solid #444654;
        ` : `
          background-color: #f7f7f8;
          border: 1px solid #e5e5e5;
        `}
      }
      
      .stat-number {
        font-size: 32px;
        font-weight: bold;
        color: #10a37f;
        margin-bottom: 5px;
      }
      
      .stat-label {
        font-size: 14px;
        color: ${darkMode ? '#c5c5d2' : '#6e6e80'};
      }
      
      .conversations-list {
        margin-top: 40px;
      }
      
      .conversation-item {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 15px 20px;
        margin-bottom: 10px;
        border-radius: 8px;
        transition: all 0.2s;
        ${darkMode ? `
          background-color: #2a2b2d;
          border: 1px solid #444654;
        ` : `
          background-color: #f7f7f8;
          border: 1px solid #e5e5e5;
        `}
      }
      
      .conversation-item:hover {
        ${darkMode ? `
          background-color: #343541;
          border-color: #565869;
        ` : `
          background-color: #ececec;
          border-color: #d3d3d3;
        `}
      }
      
      .conversation-title {
        font-weight: 500;
        flex: 1;
        margin-right: 20px;
      }
      
      .conversation-meta {
        display: flex;
        gap: 20px;
        font-size: 14px;
        color: ${darkMode ? '#c5c5d2' : '#6e6e80'};
      }
      
      .conversation-link {
        color: #10a37f;
        text-decoration: none;
        font-weight: 500;
      }
      
      .conversation-link:hover {
        text-decoration: underline;
      }
      
      .status-badge {
        display: inline-block;
        padding: 4px 8px;
        border-radius: 4px;
        font-size: 12px;
        font-weight: 500;
      }
      
      .status-success {
        background-color: #d4edda;
        color: #155724;
      }
      
      .status-failed {
        background-color: #f8d7da;
        color: #721c24;
      }
      
      .footer {
        margin-top: 50px;
        padding-top: 20px;
        border-top: 2px solid ${darkMode ? '#444654' : '#e5e5e5'};
        text-align: center;
        font-size: 12px;
        color: ${darkMode ? '#8e8ea0' : '#6e6e80'};
      }
    </style>
  `;
  
  // Count statistics
  const successCount = conversations.length;
  const failedCount = chatHistory.length - successCount;
  const totalMessages = conversations.reduce((sum, conv) => sum + (conv.messages?.length || 0), 0);
  
  // Create conversation list HTML
  let conversationsListHTML = '';
  chatHistory.forEach((chat, index) => {
    const conv = conversations.find(c => c.chatInfo.href === chat.href);
    const isSuccess = !!conv;
    const messageCount = conv ? conv.messages?.length || 0 : 0;
    const filename = isSuccess ? `conversations/${index + 1}_${chat.title.replace(/[^a-z0-9]/gi, '_').toLowerCase()}.html` : '';
    
    conversationsListHTML += `
      <div class="conversation-item">
        <div class="conversation-title">${escapeHTML(chat.title)}</div>
        <div class="conversation-meta">
          ${isSuccess ? `<span>${messageCount} messages</span>` : ''}
          <span class="status-badge ${isSuccess ? 'status-success' : 'status-failed'}">
            ${isSuccess ? 'Exported' : 'Failed'}
          </span>
          ${isSuccess ? `<a href="${filename}" class="conversation-link">View</a>` : ''}
        </div>
      </div>
    `;
  });
  
  return `
    <!DOCTYPE html>
    <html>
    <head>
      <meta charset="UTF-8">
      <meta name="viewport" content="width=device-width, initial-scale=1.0">
      <title>ChatGPT Export - Index</title>
      ${styles}
    </head>
    <body>
      <h1>ChatGPT Conversation Export</h1>
      <div class="meta">
        <div>Exported on: ${new Date(date).toLocaleString()}</div>
        <div>Total conversations: ${chatHistory.length}</div>
      </div>
      
      <div class="stats">
        <div class="stat-card">
          <div class="stat-number">${chatHistory.length}</div>
          <div class="stat-label">Total Chats</div>
        </div>
        <div class="stat-card">
          <div class="stat-number">${successCount}</div>
          <div class="stat-label">Successfully Exported</div>
        </div>
        <div class="stat-card">
          <div class="stat-number">${failedCount}</div>
          <div class="stat-label">Failed</div>
        </div>
        <div class="stat-card">
          <div class="stat-number">${totalMessages}</div>
          <div class="stat-label">Total Messages</div>
        </div>
      </div>
      
      <div class="conversations-list">
        <h2>Conversations</h2>
        ${conversationsListHTML}
      </div>
      
      <div class="footer">
        Extracted by ChatGPT Conversation Extractor
      </div>
    </body>
    </html>
  `;
}