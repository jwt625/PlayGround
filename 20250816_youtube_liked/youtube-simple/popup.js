document.addEventListener('DOMContentLoaded', function() {
  const grabButton = document.getElementById('grabTitles');
  const scrollButton = document.getElementById('scrollDown');
  const removeButton = document.getElementById('removeTop');
  const exportButton = document.getElementById('exportJson');
  const statusDiv = document.getElementById('status');
  const videoListDiv = document.getElementById('videoList');

  let currentVideos = []; // Store the current video data

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
        currentVideos = response.videos; // Store the videos
        showStatus(`Found ${response.videos.length} videos`, 'success');
        displayVideos(response.videos);
        exportButton.disabled = false; // Enable export button
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

  scrollButton.addEventListener('click', async function() {
    try {
      scrollButton.disabled = true;
      scrollButton.textContent = 'Scrolling...';

      showStatus('Getting current tab...', 'info');

      // Get current tab
      const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });

      if (!tab.url.includes('youtube.com')) {
        throw new Error('Please navigate to YouTube first');
      }

      showStatus('Scrolling to bottom...', 'info');

      // Send message to content script to scroll
      const response = await chrome.tabs.sendMessage(tab.id, {
        type: 'scrollToBottom'
      });

      if (response && response.success) {
        showStatus('Scrolled to bottom successfully', 'success');
      } else {
        showStatus('Scroll completed', 'info');
      }

    } catch (error) {
      console.error('Error:', error);
      showStatus(`Error: ${error.message}`, 'error');
    } finally {
      scrollButton.disabled = false;
      scrollButton.textContent = 'Scroll to Bottom';
    }
  });

  removeButton.addEventListener('click', async function() {
    try {
      removeButton.disabled = true;
      removeButton.textContent = 'Removing...';

      showStatus('Getting current tab...', 'info');

      // Get current tab
      const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });

      if (!tab.url.includes('youtube.com')) {
        throw new Error('Please navigate to YouTube first');
      }

      showStatus('Removing top video...', 'info');

      // Send message to content script to remove top video
      const response = await chrome.tabs.sendMessage(tab.id, {
        type: 'removeTopVideo'
      });

      if (response && response.success) {
        showStatus(`Removed video: ${response.videoTitle || 'Unknown'}`, 'success');
      } else {
        throw new Error(response?.error || 'Failed to remove video');
      }

    } catch (error) {
      console.error('Error:', error);
      showStatus(`Error: ${error.message}`, 'error');
    } finally {
      removeButton.disabled = false;
      removeButton.textContent = 'Remove Top Video';
    }
  });

  exportButton.addEventListener('click', function() {
    try {
      if (currentVideos.length === 0) {
        showStatus('No videos to export. Grab some videos first!', 'error');
        return;
      }

      exportButton.disabled = true;
      exportButton.textContent = 'Exporting...';

      // Create export data with metadata
      const exportData = {
        exportDate: new Date().toISOString(),
        totalVideos: currentVideos.length,
        source: 'YouTube Video Titles Extension',
        url: window.location?.href || 'Unknown',
        videos: currentVideos
      };

      // Create JSON blob
      const jsonString = JSON.stringify(exportData, null, 2);
      const blob = new Blob([jsonString], { type: 'application/json' });

      // Create download link
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `youtube-videos-${new Date().toISOString().split('T')[0]}.json`;

      // Trigger download
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);

      // Clean up
      URL.revokeObjectURL(url);

      showStatus(`Exported ${currentVideos.length} videos to JSON`, 'success');

    } catch (error) {
      console.error('Export error:', error);
      showStatus(`Export failed: ${error.message}`, 'error');
    } finally {
      exportButton.disabled = false;
      exportButton.textContent = 'Export JSON';
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
