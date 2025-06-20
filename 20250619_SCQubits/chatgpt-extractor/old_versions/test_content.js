// Test content script to examine ChatGPT HTML structure

console.log('=== ChatGPT HTML Test Script Loaded ===');

// Function to test various selectors and log what we find
function testHTMLStructure() {
  console.log('\n--- Testing HTML Structure ---');
  
  // Test 1: Look for message containers
  console.log('\n1. Testing message containers (.text-token-text-primary.w-full):');
  const messageContainers = document.querySelectorAll('.text-token-text-primary.w-full');
  console.log(`Found ${messageContainers.length} message containers`);
  if (messageContainers.length > 0) {
    console.log('First container HTML:', messageContainers[0].outerHTML.substring(0, 200) + '...');
  }
  
  // Test 2: Look for data-message attributes
  console.log('\n2. Testing data-message-author-role elements:');
  const messageElements = document.querySelectorAll('[data-message-author-role]');
  console.log(`Found ${messageElements.length} elements with data-message-author-role`);
  messageElements.forEach((el, i) => {
    if (i < 3) { // Show first 3
      console.log(`Message ${i + 1}: role="${el.getAttribute('data-message-author-role')}", id="${el.getAttribute('data-message-id')}"`);
    }
  });
  
  // Test 3: Look for whitespace-pre-wrap content
  console.log('\n3. Testing whitespace-pre-wrap elements:');
  const whitespacePre = document.querySelectorAll('.whitespace-pre-wrap');
  console.log(`Found ${whitespacePre.length} whitespace-pre-wrap elements`);
  if (whitespacePre.length > 0) {
    console.log('First element text:', whitespacePre[0].textContent.substring(0, 100) + '...');
  }
  
  // Test 4: Look for markdown prose
  console.log('\n4. Testing markdown prose elements:');
  const markdownProse = document.querySelectorAll('.markdown.prose');
  console.log(`Found ${markdownProse.length} markdown prose elements`);
  if (markdownProse.length > 0) {
    console.log('First element HTML:', markdownProse[0].outerHTML.substring(0, 200) + '...');
  }
  
  // Test 5: Look for chat history
  console.log('\n5. Testing chat history (#history):');
  const historyDiv = document.getElementById('history');
  if (historyDiv) {
    console.log('History div found!');
    const chatLinks = historyDiv.querySelectorAll('a[href^="/c/"], a[href^="/chat/"]');
    console.log(`Found ${chatLinks.length} chat links`);
    if (chatLinks.length > 0) {
      console.log('First 3 chats:');
      for (let i = 0; i < Math.min(3, chatLinks.length); i++) {
        const titleSpan = chatLinks[i].querySelector('span[dir="auto"]');
        console.log(`${i + 1}. "${titleSpan?.textContent}" - ${chatLinks[i].getAttribute('href')}`);
      }
    }
  } else {
    console.log('History div NOT found');
  }
  
  // Test 6: Try alternative selectors
  console.log('\n6. Testing alternative selectors:');
  
  // Look for any article tags
  const articles = document.querySelectorAll('article');
  console.log(`Found ${articles.length} article elements`);
  
  // Look for main content area
  const main = document.querySelector('main');
  console.log('Main element found:', !!main);
  
  // Look for any elements with "message" in class name
  const messageClasses = document.querySelectorAll('[class*="message"]');
  console.log(`Found ${messageClasses.length} elements with "message" in class name`);
  
  // Test 7: Log the body structure
  console.log('\n7. Body structure:');
  console.log('Body classes:', document.body.className);
  console.log('Body children count:', document.body.children.length);
  
  // Test 8: Look for specific conversation elements
  console.log('\n8. Looking for conversation-specific elements:');
  
  // Try different query patterns
  const patterns = [
    'div[class*="conversation"]',
    'div[class*="thread"]',
    'div[class*="chat"]',
    '[role="main"]',
    '[data-testid*="conversation"]'
  ];
  
  patterns.forEach(pattern => {
    const elements = document.querySelectorAll(pattern);
    if (elements.length > 0) {
      console.log(`Pattern "${pattern}": found ${elements.length} elements`);
    }
  });
}

// Run the test immediately
testHTMLStructure();

// Also run it after a delay in case content loads dynamically
setTimeout(() => {
  console.log('\n=== Running test again after 2 seconds ===');
  testHTMLStructure();
}, 2000);

// Listen for messages to run test on demand
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  if (request.action === 'runTest') {
    console.log('\n=== Running test on demand ===');
    testHTMLStructure();
    sendResponse({ success: true, message: 'Test completed, check console' });
  }
  return true;
});