// SPA controller - runs in the ChatGPT page context
// Clicks through chats and extracts content without page reloads

console.log('SPA Controller loaded');

(async function() {
  // Find all chat links in the sidebar
  const chatLinks = Array.from(document.querySelectorAll('#history a[href^="/c/"], #history a[href^="/chat/"]'));
  console.log(`Found ${chatLinks.length} chats to download`);
  
  if (chatLinks.length === 0) {
    alert('No chats found. Make sure the sidebar is open!');
    return;
  }
  
  // Build chat list with updated local links
  const allChats = chatLinks.map((link, index) => {
    const titleEl = link.querySelector('span[dir="auto"]');
    const title = titleEl ? titleEl.textContent.trim() : `Chat ${index + 1}`;
    const href = link.getAttribute('href');
    const safeTitle = title.replace(/[^a-z0-9]/gi, '_').substring(0, 40);
    const localFile = `${(index + 1).toString().padStart(3, '0')}_${safeTitle}.html`;
    
    return { title, href, localFile, element: link };
  });
  
  // Create index HTML
  const indexHTML = createIndexHTML(allChats);
  await downloadHTML(indexHTML, 'index.html', 0);
  
  // Wait a bit before starting
  await new Promise(resolve => setTimeout(resolve, 1000));
  
  // Process each chat
  for (let i = 0; i < allChats.length; i++) {
    const chat = allChats[i];
    console.log(`Processing ${i + 1}/${allChats.length}: ${chat.title}`);
    
    // Click the chat link
    chat.element.click();
    
    // Wait for content to load
    await waitForContent();
    
    // Extract and build HTML
    const html = await extractAndBuildHTML(chat.title, allChats);
    
    // Send to background for download
    await chrome.runtime.sendMessage({
      action: 'downloadChat',
      chatData: {
        title: chat.title,
        html: html
      },
      index: i + 1
    });
    
    // Small delay between downloads
    await new Promise(resolve => setTimeout(resolve, 500));
  }
  
  console.log('All chats processed!');
  alert(`Download complete! Downloaded ${allChats.length} chats to your Downloads/chatgpt_export folder.`);
})();

async function waitForContent() {
  // Wait for the conversation to load
  return new Promise((resolve) => {
    let attempts = 0;
    const checkInterval = setInterval(() => {
      // Look for message containers
      const messages = document.querySelectorAll('[data-message-author-role]');
      attempts++;
      
      if (messages.length > 0 || attempts > 20) {
        clearInterval(checkInterval);
        // Extra wait for any animations
        setTimeout(resolve, 1000);
      }
    }, 500);
  });
}

async function extractAndBuildHTML(title, allChats) {
  // Extract messages
  const messages = [];
  const messageElements = document.querySelectorAll('[data-message-author-role]');
  
  messageElements.forEach(element => {
    const role = element.getAttribute('data-message-author-role');
    
    // Find content within this message
    const contentEl = element.querySelector('.text-token-text-primary.w-full') || element;
    const textElements = contentEl.querySelectorAll('.whitespace-pre-wrap, .markdown');
    
    let content = '';
    textElements.forEach(el => {
      content += el.innerHTML + '\n';
    });
    
    if (content.trim()) {
      messages.push({ role, content });
    }
  });
  
  // Build HTML with sidebar
  return `<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8">
  <title>${escapeHtml(title)}</title>
  <style>
    body { margin: 0; font-family: -apple-system, sans-serif; display: flex; height: 100vh; }
    .sidebar { width: 260px; background: #f3f4f6; padding: 20px; overflow-y: auto; flex-shrink: 0; }
    .sidebar h3 { margin: 0 0 15px 0; font-size: 16px; }
    .chat-link { display: block; padding: 8px 12px; margin: 2px 0; text-decoration: none; color: #374151; border-radius: 6px; }
    .chat-link:hover { background: #e5e7eb; }
    .chat-link.current { background: #ddd6fe; color: #5b21b6; }
    .content { flex: 1; overflow-y: auto; padding: 20px 40px; }
    .message { margin: 20px 0; padding: 20px; border-radius: 8px; }
    .message.user { background: #f3f4f6; }
    .message.assistant { background: #fff; border: 1px solid #e5e7eb; }
    .role { font-weight: bold; margin-bottom: 10px; color: #374151; }
    .message-content { line-height: 1.6; }
    pre { background: #1f2937; color: #f3f4f6; padding: 16px; border-radius: 6px; overflow-x: auto; }
    code { background: #f3f4f6; padding: 2px 4px; border-radius: 3px; }
    pre code { background: none; padding: 0; }
  </style>
</head>
<body>
  <div class="sidebar">
    <h3>Chats</h3>
    ${allChats.map((chat, index) => `
      <a href="${chat.localFile}" class="chat-link ${chat.title === title ? 'current' : ''}">
        ${escapeHtml(chat.title)}
      </a>
    `).join('')}
  </div>
  
  <div class="content">
    <h1>${escapeHtml(title)}</h1>
    ${messages.map(msg => `
      <div class="message ${msg.role}">
        <div class="role">${msg.role === 'user' ? 'You' : 'ChatGPT'}</div>
        <div class="message-content">${msg.content}</div>
      </div>
    `).join('')}
  </div>
</body>
</html>`;
}

function createIndexHTML(allChats) {
  return `<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8">
  <title>ChatGPT Export - Index</title>
  <style>
    body { font-family: -apple-system, sans-serif; max-width: 800px; margin: 0 auto; padding: 40px 20px; }
    h1 { color: #111827; }
    .info { background: #f3f4f6; padding: 20px; border-radius: 8px; margin: 20px 0; }
    .chat-list { margin: 20px 0; }
    .chat-item { padding: 12px 0; border-bottom: 1px solid #e5e7eb; }
    .chat-item:last-child { border-bottom: none; }
    a { color: #2563eb; text-decoration: none; }
    a:hover { text-decoration: underline; }
  </style>
</head>
<body>
  <h1>ChatGPT Conversation Export</h1>
  <div class="info">
    <p>Exported: ${new Date().toLocaleString()}</p>
    <p>Total conversations: ${allChats.length}</p>
  </div>
  <div class="chat-list">
    <h2>All Conversations</h2>
    ${allChats.map((chat, index) => `
      <div class="chat-item">
        <a href="${chat.localFile}">${index + 1}. ${escapeHtml(chat.title)}</a>
      </div>
    `).join('')}
  </div>
</body>
</html>`;
}

async function downloadHTML(html, filename, index) {
  const dataUrl = 'data:text/html;charset=utf-8,' + encodeURIComponent(html);
  
  await chrome.runtime.sendMessage({
    action: 'downloadChat',
    chatData: {
      title: filename,
      html: html
    },
    index: index
  });
}

function escapeHtml(text) {
  const div = document.createElement('div');
  div.textContent = text;
  return div.innerHTML;
}