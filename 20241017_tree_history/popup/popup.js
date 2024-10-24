document.addEventListener('DOMContentLoaded', function() {
  // Get initial tracking status
  chrome.runtime.sendMessage({action: "getTrackingStatus"}, function(response) {
    updateTrackingUI(response.isTracking);
  });

  // Set up tracking toggle
  const toggleButton = document.getElementById('toggleTracking');
  toggleButton.addEventListener('click', function() {
    chrome.runtime.sendMessage({action: "toggleTracking"}, function(response) {
      updateTrackingUI(response.isTracking);
    });
  });

  function updateTrackingUI(isTracking) {
    const toggleButton = document.getElementById('toggleTracking');
    const statusElement = document.getElementById('trackingStatus');
    
    toggleButton.textContent = isTracking ? 'Stop Tracking' : 'Start Tracking';
    statusElement.textContent = isTracking ? 'Tracking Active' : 'Not Tracking';
    statusElement.className = `status ${isTracking ? 'active' : 'inactive'}`;
  }

  // Get and display tab tree
  chrome.runtime.sendMessage({action: "getTabTree"}, function(response) {
    if (chrome.runtime.lastError) {
      console.error(chrome.runtime.lastError);
      alert('An error occurred while fetching the tab tree. Please try again.');
      return;
    }

    const tabTree = response.tabTree || {};
    const tabTreeElement = document.getElementById('tabTree');
    
    function createTreeView(tree, element) {
      const ul = document.createElement('ul');
      Object.values(tree).forEach(node => {
        const li = document.createElement('li');
        
        // Create main node text
        const nodeText = document.createElement('div');
        nodeText.textContent = `${node.title} (${node.url})`;
        if (node.closedAt) {
          nodeText.textContent += ` [Closed]`;
        }
        
        // Add word frequency if available
        if (node.topWords && node.topWords.length > 0) {
          const wordFreq = document.createElement('div');
          wordFreq.style.fontSize = '0.8em';
          wordFreq.style.color = '#666';
          wordFreq.textContent = 'Top words: ' + 
            node.topWords.map(w => `${w.word}(${w.count})`).join(', ');
          nodeText.appendChild(wordFreq);
        }
        
        li.appendChild(nodeText);
        
        if (node.children && node.children.length > 0) {
          createTreeView(node.children, li);
        }
        ul.appendChild(li);
      });
      element.appendChild(ul);
    }

    createTreeView(tabTree, tabTreeElement);

    // Save button functionality
    document.getElementById('saveButton').addEventListener('click', function() {
      const jsonString = JSON.stringify(tabTree, null, 2);
      const blob = new Blob([jsonString], {type: "application/json"});
      const url = URL.createObjectURL(blob);
      
      const a = document.createElement('a');
      a.href = url;
      a.download = 'tabTree.json';
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    });

    // Clear button functionality
    document.getElementById('clearButton').addEventListener('click', function() {
      if (confirm('Are you sure you want to clear the tab tree?')) {
        sendMessageWithRetry({action: "clearTabTree"}, 3)
          .then(response => {
            if (response && response.success) {
              location.reload();  // Reload the popup to show the updated tree
            } else {
              alert('Failed to clear the tab tree. Please try again.');
            }
          })
          .catch(error => {
            console.error(error);
            alert('An error occurred while clearing the tab tree. Please try again.');
          });
      }
    });
  });
});

function sendMessageWithRetry(message, maxRetries) {
  return new Promise((resolve, reject) => {
    function attemptSend(retriesLeft) {
      chrome.runtime.sendMessage(message, response => {
        if (chrome.runtime.lastError) {
          console.log(chrome.runtime.lastError);
          if (retriesLeft > 0) {
            console.log(`Retrying... ${retriesLeft} attempts left.`);
            setTimeout(() => attemptSend(retriesLeft - 1), 1000);
          } else {
            reject(new Error('Max retries reached'));
          }
        } else {
          resolve(response);
        }
      });
    }
    attemptSend(maxRetries);
  });
}

// Add this to your existing popup.js event listeners
document.getElementById('openViewer')?.addEventListener('click', () => {
  chrome.tabs.create({
    url: chrome.runtime.getURL('viewer/viewer.html')
  });
});