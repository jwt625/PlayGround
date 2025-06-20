// Smart popup that works with background script v2

document.addEventListener('DOMContentLoaded', () => {
  // Elements
  const statusCard = document.getElementById('statusCard');
  const statusText = document.getElementById('statusText');
  const progressInfo = document.getElementById('progressInfo');
  const progressFill = document.getElementById('progressFill');
  const progressText = document.getElementById('progressText');
  const errorMsg = document.getElementById('errorMsg');
  
  // Sections
  const notReady = document.getElementById('notReady');
  const inProgress = document.getElementById('inProgress');
  const resumeOption = document.getElementById('resumeOption');
  const complete = document.getElementById('complete');
  
  // Buttons
  const startBtn = document.getElementById('startBtn');
  const cancelBtn = document.getElementById('cancelBtn');
  const resumeBtn = document.getElementById('resumeBtn');
  const restartBtn = document.getElementById('restartBtn');
  const newDownloadBtn = document.getElementById('newDownloadBtn');
  
  // Check status on load
  checkStatus();
  
  // Poll for updates while popup is open
  const statusInterval = setInterval(checkStatus, 2000);
  
  // Clean up interval when popup closes
  window.addEventListener('unload', () => {
    clearInterval(statusInterval);
  });
  
  // Button handlers
  startBtn.addEventListener('click', async () => {
    const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
    
    if (!tab.url || (!tab.url.includes('chatgpt.com') && !tab.url.includes('chat.openai.com'))) {
      showError('Please navigate to ChatGPT first');
      return;
    }
    
    chrome.runtime.sendMessage({ action: 'startBulkDownload' }, (response) => {
      if (response.error) {
        showError(response.error);
      } else {
        checkStatus();
      }
    });
  });
  
  resumeBtn.addEventListener('click', () => {
    chrome.runtime.sendMessage({ action: 'resumeDownload' }, (response) => {
      if (response.error) {
        showError(response.error);
      } else {
        checkStatus();
      }
    });
  });
  
  restartBtn.addEventListener('click', () => {
    chrome.runtime.sendMessage({ action: 'startBulkDownload' }, (response) => {
      if (response.error) {
        showError(response.error);
      } else {
        checkStatus();
      }
    });
  });
  
  newDownloadBtn.addEventListener('click', () => {
    chrome.runtime.sendMessage({ action: 'startBulkDownload' }, (response) => {
      if (response.error) {
        showError(response.error);
      } else {
        checkStatus();
      }
    });
  });
  
  cancelBtn.addEventListener('click', () => {
    if (confirm('Are you sure you want to cancel the download?')) {
      chrome.runtime.sendMessage({ action: 'cancelDownload' }, () => {
        checkStatus();
      });
    }
  });
  
  function checkStatus() {
    chrome.runtime.sendMessage({ action: 'getStatus' }, (state) => {
      if (!state) return;
      
      hideAll();
      errorMsg.classList.add('hidden');
      
      if (state.isActive) {
        // Download in progress
        statusCard.className = 'status-card active';
        statusText.textContent = `Downloading: ${state.allChats[state.currentIndex]?.title || 'Loading...'}`;
        
        progressInfo.classList.remove('hidden');
        const progress = ((state.currentIndex + 1) / state.allChats.length) * 100;
        progressFill.style.width = progress + '%';
        progressText.textContent = `Chat ${state.currentIndex + 1} of ${state.allChats.length}`;
        
        inProgress.classList.remove('hidden');
        
      } else if (state.allChats.length > 0 && state.currentIndex < state.allChats.length) {
        // Download was interrupted
        statusCard.className = 'status-card';
        statusText.textContent = 'Download interrupted';
        
        document.getElementById('resumeInfo').textContent = 
          `Downloaded ${state.currentIndex} of ${state.allChats.length} chats`;
        
        resumeOption.classList.remove('hidden');
        
      } else if (state.downloadedFiles.length > 0) {
        // Download complete
        statusCard.className = 'status-card complete';
        statusText.textContent = 'Download complete!';
        
        progressInfo.classList.remove('hidden');
        progressFill.style.width = '100%';
        progressText.textContent = `Successfully downloaded ${state.downloadedFiles.filter(f => f.filename).length} of ${state.downloadedFiles.length} chats`;
        
        complete.classList.remove('hidden');
        
      } else {
        // Ready to start
        statusCard.className = 'status-card';
        statusText.textContent = 'Ready to download';
        notReady.classList.remove('hidden');
      }
    });
  }
  
  function hideAll() {
    notReady.classList.add('hidden');
    inProgress.classList.add('hidden');
    resumeOption.classList.add('hidden');
    complete.classList.add('hidden');
  }
  
  function showError(message) {
    errorMsg.textContent = message;
    errorMsg.classList.remove('hidden');
  }
});