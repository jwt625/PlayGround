// Simple popup that communicates with background script

document.addEventListener('DOMContentLoaded', () => {
  const startBtn = document.getElementById('startBtn');
  const statusDiv = document.getElementById('status');
  const progressDiv = document.getElementById('progress');
  
  // Check initial status
  updateStatus();
  
  startBtn.addEventListener('click', async () => {
    console.log('Starting bulk download via background script');
    
    chrome.runtime.sendMessage({ action: 'startBulkDownload' }, (response) => {
      if (response.error) {
        statusDiv.textContent = `Error: ${response.error}`;
      } else {
        statusDiv.textContent = 'Download started! Check progress in console.';
        updateStatus();
      }
    });
  });
  
  // Poll for status updates
  function updateStatus() {
    chrome.runtime.sendMessage({ action: 'getDownloadStatus' }, (response) => {
      if (response.isDownloading) {
        startBtn.disabled = true;
        progressDiv.style.display = 'block';
        statusDiv.textContent = `Downloading: ${response.currentChat?.title || 'Loading...'}`;
        document.getElementById('progressText').textContent = 
          `Progress: ${response.currentIndex + 1}/${response.total}`;
        const percentage = ((response.currentIndex + 1) / response.total) * 100;
        document.getElementById('progressBar').style.width = percentage + '%';
        
        // Poll again in 2 seconds
        setTimeout(updateStatus, 2000);
      } else {
        startBtn.disabled = false;
        progressDiv.style.display = 'none';
        if (response.completed > 0) {
          statusDiv.textContent = `Completed! Downloaded ${response.completed} chats.`;
        }
      }
    });
  }
});