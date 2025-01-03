// Function to trigger content scraping
async function scrapeTweet(tweetUrl) {
  // Get all tabs that match the tweet URL
  const tabs = await chrome.tabs.query({
    url: [
      "https://twitter.com/*",
      "https://x.com/*"
    ]
  });

  // Find the tab with our tweet or create a new one
  let tweetTab;
  if (tabs.length > 0) {
    tweetTab = tabs[0];
    await chrome.tabs.update(tweetTab.id, { url: tweetUrl });
  } else {
    tweetTab = await chrome.tabs.create({ url: tweetUrl, active: false });
  }

  // Wait for the page to load and inject our content script
  await new Promise(resolve => setTimeout(resolve, 2000));
  
  try {
    await chrome.scripting.executeScript({
      target: { tabId: tweetTab.id },
      function: () => {
        // This will trigger our content.js to scrape the tweet
        window.dispatchEvent(new CustomEvent('SCRAPE_TWEET'));
      }
    });
  } catch (error) {
    console.error('Error executing script:', error);
  }
}

// Function to save settings
async function saveSettings() {
  const tweetUrl = document.getElementById('tweetUrl').value;
  
  // Basic URL validation
  if (!tweetUrl.match(/^https?:\/\/(twitter\.com|x\.com)\/\w+\/status\/\d+/)) {
    alert('Please enter a valid tweet URL\nExample: https://twitter.com/username/status/123456789');
    return;
  }
  
  // Save to Chrome storage
  await chrome.storage.sync.set({ tweetUrl });
  
  // Try to scrape the tweet immediately
  await scrapeTweet(tweetUrl);
  
  // Show success message
  alert('Settings saved! The tweet will appear at the top of your home feed.');
}

// Function to save HTML to file
function saveHtmlToFile(type) {
  const content = type === 'raw' ? 
    document.getElementById('rawHtml').textContent :
    document.getElementById('scrapedContent').textContent;
    
  if (!content || content === 'No content scraped yet...' || content === 'No raw HTML yet...') {
    alert('No content to save yet. Please fetch the tweet first.');
    return;
  }

  // Create blob and download link
  const blob = new Blob([content], { type: 'text/html' });
  const url = URL.createObjectURL(blob);
  const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
  const prefix = type === 'raw' ? 'raw' : 'parsed';
  
  const a = document.createElement('a');
  a.href = url;
  a.download = `tweet-${prefix}-${timestamp}.html`;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

// Event Listeners
document.getElementById('saveButton').addEventListener('click', saveSettings);
document.getElementById('refreshButton').addEventListener('click', async () => {
  const { tweetUrl } = await chrome.storage.sync.get(['tweetUrl']);
  if (tweetUrl) {
    await scrapeTweet(tweetUrl);
  } else {
    alert('Please save a tweet URL first');
  }
});
document.getElementById('saveHtmlButton').addEventListener('click', () => saveHtmlToFile('parsed'));
document.getElementById('saveRawHtmlButton').addEventListener('click', () => saveHtmlToFile('raw'));

// Listen for scraped content from content script
chrome.runtime.onMessage.addListener((message) => {
  if (message.type === 'SCRAPED_CONTENT') {
    document.getElementById('scrapedContent').textContent = message.parsedContent || 'No tweet element found';
    document.getElementById('rawHtml').textContent = message.rawHtml || 'No raw HTML available';
    document.getElementById('timestamp').textContent = 
      `Last updated: ${new Date(message.timestamp).toLocaleString()}`;
  }
});
document.getElementById('refreshButton').addEventListener('click', async () => {
  const { tweetUrl } = await chrome.storage.sync.get(['tweetUrl']);
  if (tweetUrl) {
    await scrapeTweet(tweetUrl);
  } else {
    alert('Please save a tweet URL first');
  }
});

// Listen for scraped content from content script
chrome.runtime.onMessage.addListener((message) => {
  if (message.type === 'SCRAPED_CONTENT') {
    document.getElementById('scrapedContent').textContent = message.content;
    document.getElementById('timestamp').textContent = 
      `Last updated: ${new Date(message.timestamp).toLocaleString()}`;
  }
});

// Load saved settings
chrome.storage.sync.get(['tweetUrl'], (result) => {
  if (result.tweetUrl) {
    document.getElementById('tweetUrl').value = result.tweetUrl;
    scrapeTweet(result.tweetUrl);  // Try to get initial preview
  }
});