document.addEventListener('DOMContentLoaded', function() {
  const grabButton = document.getElementById('grabTitles');
  const statusDiv = document.getElementById('status');
  const videoListDiv = document.getElementById('videoList');
  
  grabButton.addEventListener('click', async function() {
    try {
      grabButton.disabled = true;
      grabButton.textContent = 'Grabbing...';
      
      showStatus('Getting current tab...', 'info');
      
      // Get current tab
      const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
      
      if (!tab.url.includes('youtube.com')) {
        throw new Error('Please navigate to YouTube first');
      }
      
      showStatus('Sending message to content script...', 'info');
      
      // Send message to content script
      const response = await chrome.tabs.sendMessage(tab.id, {
        type: 'getVideoTitles'
      });
      
      if (response.success) {
        showStatus(`Found ${response.videos.length} videos`, 'success');
        displayVideos(response.videos);
      } else {
        throw new Error(response.error || 'Failed to get video titles');
      }
      
    } catch (error) {
      console.error('Error:', error);
      showStatus(`Error: ${error.message}`, 'error');
    } finally {
      grabButton.disabled = false;
      grabButton.textContent = 'Grab Video Titles';
    }
  });
  
  function showStatus(message, type) {
    statusDiv.textContent = message;
    statusDiv.className = type;
    statusDiv.style.display = 'block';
  }
  
  function displayVideos(videos) {
    videoListDiv.innerHTML = '';
    
    if (videos.length === 0) {
      videoListDiv.innerHTML = '<div class="video-item">No videos found</div>';
      return;
    }
    
    videos.forEach((video, index) => {
      const videoItem = document.createElement('div');
      videoItem.className = 'video-item';
      
      videoItem.innerHTML = `
        <div class="video-title">${index + 1}. ${video.title || 'No title'}</div>
        <div class="video-channel">Channel: ${video.channel || 'Unknown'}</div>
        ${video.videoId ? `<div style="color: #999; font-size: 11px;">ID: ${video.videoId}</div>` : ''}
      `;
      
      videoListDiv.appendChild(videoItem);
    });
  }
});
