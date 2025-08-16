// IMMEDIATE LOG - This should appear as soon as the file loads
console.log('🚨 CONTENT SCRIPT FILE LOADING NOW!');
console.log('🚨 URL:', window.location.href);
console.log('🚨 Time:', new Date().toISOString());

try {
  console.log('🧪 TEST: Minimal content script loaded successfully!');
  console.log('🧪 TEST: Current URL:', window.location.href);
  console.log('🧪 TEST: DOM ready state:', document.readyState);
  console.log('🧪 TEST: Chrome runtime available:', !!chrome.runtime);

  // Set up basic message listener
  if (chrome && chrome.runtime && chrome.runtime.onMessage) {
    chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
      console.log('🧪 TEST: Received message:', message);

      if (message.type === 'test' || message.type === 'getStatus') {
        console.log('🧪 TEST: Responding to message');
        sendResponse({ success: true, message: 'Test content script is working!' });
      }

      return true;
    });
    console.log('🧪 TEST: Message listener set up successfully');
  } else {
    console.error('🧪 TEST: Chrome runtime not available!');
  }

} catch (error) {
  console.error('🚨 ERROR in test content script:', error);
}
