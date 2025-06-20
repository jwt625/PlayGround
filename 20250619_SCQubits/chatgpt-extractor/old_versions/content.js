// Content script for extracting ChatGPT conversations

function extractConversation() {
  console.log('Starting conversation extraction...');
  
  // Find all message containers using the class selector
  const messages = [];
  
  // Look for all message containers with the specific class
  const messageContainers = document.querySelectorAll('.text-token-text-primary.w-full');
  console.log(`Found ${messageContainers.length} message containers`);
  
  messageContainers.forEach((container, index) => {
    console.log(`Processing message container ${index + 1}`);
    // Find the parent element with data-message-author-role
    const messageElement = container.closest('[data-message-author-role]');
    
    if (!messageElement) return;
    
    const role = messageElement.getAttribute('data-message-author-role');
    const messageId = messageElement.getAttribute('data-message-id');
    
    // Get the text content from various possible content containers
    // ChatGPT uses different classes for different types of content
    const contentSelectors = [
      '.whitespace-pre-wrap',
      '.markdown.prose',
      '.markdown.prose.dark\\:prose-invert',
      '[class*="markdown"][class*="prose"]'
    ];
    
    let content = '';
    
    // Try each selector to find content
    for (const selector of contentSelectors) {
      const contentElements = container.querySelectorAll(selector);
      contentElements.forEach(contentEl => {
        // Avoid duplicating content if it's already captured
        if (!content.includes(contentEl.textContent.trim())) {
          content += contentEl.textContent + '\n';
        }
      });
    }
    
    // If still no content, try getting all text content from the container
    if (!content.trim()) {
      content = container.textContent;
    }
    
    // Also look for code blocks within the container
    const codeBlocks = container.querySelectorAll('pre');
    const codeContent = [];
    codeBlocks.forEach(block => {
      // Get the language if available
      const langElement = block.previousElementSibling?.querySelector('span');
      const language = langElement ? langElement.textContent : 'plaintext';
      
      // Get the code content
      const codeElement = block.querySelector('code');
      if (codeElement) {
        codeContent.push({
          language: language,
          code: codeElement.textContent
        });
      }
    });
    
    // Get timestamp if available
    const timeElement = messageElement.querySelector('time');
    const timestamp = timeElement ? timeElement.getAttribute('datetime') : null;
    
    // Get model info for assistant messages
    const modelSlug = messageElement.getAttribute('data-message-model-slug');
    
    if (content.trim() || codeContent.length > 0) {
      messages.push({
        role: role,
        id: messageId,
        content: content.trim(),
        codeBlocks: codeContent,
        timestamp: timestamp,
        model: modelSlug
      });
    }
  });
  
  // Get conversation title
  const titleElement = document.querySelector('title');
  const title = titleElement ? titleElement.textContent : 'ChatGPT Conversation';
  
  return {
    title: title,
    messages: messages,
    extractedAt: new Date().toISOString(),
    url: window.location.href
  };
}

// Function to extract chat history list
function extractChatHistory() {
  console.log('Extracting chat history...');
  
  const chats = [];
  const historyDiv = document.getElementById('history');
  
  if (!historyDiv) {
    console.log('History div not found');
    return chats;
  }
  
  // Find all chat links in the history
  const chatLinks = historyDiv.querySelectorAll('a[href^="/c/"], a[href^="/chat/"]');
  
  chatLinks.forEach(link => {
    // Find the title span
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
  
  console.log(`Found ${chats.length} chats in history`);
  return chats;
}

// Listen for messages from the popup
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  console.log('Content script received message:', request);
  
  if (request.action === 'extractConversation') {
    console.log('Extracting conversation...');
    try {
      const conversation = extractConversation();
      console.log('Extracted conversation:', conversation);
      sendResponse({ success: true, data: conversation });
    } catch (error) {
      console.error('Error extracting conversation:', error);
      sendResponse({ success: false, error: error.message });
    }
  } else if (request.action === 'extractChatHistory') {
    console.log('Extracting chat history...');
    try {
      const chatHistory = extractChatHistory();
      console.log('Extracted chat history:', chatHistory);
      sendResponse({ success: true, data: chatHistory });
    } catch (error) {
      console.error('Error extracting chat history:', error);
      sendResponse({ success: false, error: error.message });
    }
  }
  return true; // Keep the message channel open for async response
});

console.log('ChatGPT Extractor content script loaded');