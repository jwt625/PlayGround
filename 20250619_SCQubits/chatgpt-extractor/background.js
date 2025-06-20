// Handle keyboard shortcuts
chrome.commands.onCommand.addListener((command) => {
  console.log('Command received:', command);
  if (command === 'grab-current-page') {
    grabCurrentPage();
  }
});

async function grabCurrentPage() {
  console.log('grabCurrentPage called');
  try {
    const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
    console.log('Current tab:', tab.url);
    
    // Check if we're on ChatGPT
    if (!tab.url.includes('chatgpt.com') && !tab.url.includes('chat.openai.com')) {
      console.log('Not on ChatGPT');
      return;
    }
    
    console.log('Executing script...');
    const results = await chrome.scripting.executeScript({
      target: { tabId: tab.id },
      func: () => {
        // Get the HTML
        const html = document.documentElement.outerHTML;
        
        // Get title from the page
        const titleEl = document.querySelector('h1');
        const title = titleEl ? titleEl.textContent : 'Untitled';
        
        return { html, title };
      }
    });
    
    if (!results || !results[0]) {
      console.log('No results from script');
      return;
    }
    
    const { html, title } = results[0].result;
    console.log('Got HTML, title:', title);
    
    // Get current chat index from storage
    const data = await chrome.storage.local.get(['currentIndex', 'chatList']);
    const currentIndex = data.currentIndex || 0;
    const timestamp = new Date().toISOString().replace(/[:.]/g, '-').substring(0, 19);
    const safeTitle = title.replace(/[^a-z0-9\s]/gi, '_').trim().replace(/\s+/g, '_').substring(0, 60);
    
    let filename;
    if (data.chatList && currentIndex >= 0) {
      filename = `${(currentIndex + 1).toString().padStart(3, '0')}_${timestamp}_${safeTitle}.html`;
    } else {
      filename = `chatgpt_${safeTitle}_${timestamp}.html`;
    }
    
    console.log('Downloading as:', filename);
    
    // Use data URL instead of blob URL (service workers don't support createObjectURL)
    const dataUrl = 'data:text/html;charset=utf-8,' + encodeURIComponent(html);
    
    chrome.downloads.download({
      url: dataUrl,
      filename: filename,
      saveAs: false
    }, (downloadId) => {
      console.log('Download started:', downloadId);
      if (chrome.runtime.lastError) {
        console.error('Download error:', chrome.runtime.lastError);
      }
    });
  } catch (error) {
    console.error('Error:', error);
  }
}