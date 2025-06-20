// Test popup script

document.addEventListener('DOMContentLoaded', () => {
  const injectBtn = document.getElementById('injectBtn');
  const runTestBtn = document.getElementById('runTestBtn');
  const grabHtmlBtn = document.getElementById('grabHtmlBtn');
  const statusDiv = document.getElementById('status');
  
  // Inject test script
  injectBtn.addEventListener('click', async () => {
    statusDiv.textContent = 'Injecting test script...';
    
    try {
      const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
      
      chrome.scripting.executeScript({
        target: { tabId: tab.id },
        files: ['test_content.js']
      }, () => {
        if (chrome.runtime.lastError) {
          statusDiv.innerHTML = `<strong>Error:</strong> ${chrome.runtime.lastError.message}`;
        } else {
          statusDiv.innerHTML = '<strong>Success!</strong> Check the page console (F12) for results.';
        }
      });
    } catch (error) {
      statusDiv.innerHTML = `<strong>Error:</strong> ${error.message}`;
    }
  });
  
  // Run test
  runTestBtn.addEventListener('click', async () => {
    statusDiv.textContent = 'Running test...';
    
    try {
      const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
      
      chrome.tabs.sendMessage(tab.id, { action: 'runTest' }, (response) => {
        if (chrome.runtime.lastError) {
          statusDiv.innerHTML = `<strong>Error:</strong> ${chrome.runtime.lastError.message}<br>Try injecting the script first.`;
        } else {
          statusDiv.innerHTML = '<strong>Test completed!</strong> Check the page console for results.';
        }
      });
    } catch (error) {
      statusDiv.innerHTML = `<strong>Error:</strong> ${error.message}`;
    }
  });
  
  // Grab full HTML
  grabHtmlBtn.addEventListener('click', async () => {
    statusDiv.textContent = 'Grabbing HTML...';
    
    try {
      const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
      
      chrome.scripting.executeScript({
        target: { tabId: tab.id },
        func: () => {
          // Get the full rendered HTML
          const html = document.documentElement.outerHTML;
          return html;
        }
      }, (results) => {
        if (chrome.runtime.lastError) {
          statusDiv.innerHTML = `<strong>Error:</strong> ${chrome.runtime.lastError.message}`;
        } else if (results && results[0] && results[0].result) {
          const html = results[0].result;
          
          // Create blob and download
          const blob = new Blob([html], { type: 'text/html' });
          const url = URL.createObjectURL(blob);
          const timestamp = new Date().getTime();
          
          chrome.downloads.download({
            url: url,
            filename: `chatgpt_raw_html_${timestamp}.html`,
            saveAs: true
          }, () => {
            URL.revokeObjectURL(url);
            statusDiv.innerHTML = '<strong>HTML downloaded!</strong>';
          });
        }
      });
    } catch (error) {
      statusDiv.innerHTML = `<strong>Error:</strong> ${error.message}`;
    }
  });
});