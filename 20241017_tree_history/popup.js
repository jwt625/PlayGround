document.addEventListener('DOMContentLoaded', function() {
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
        li.textContent = `${node.title} (${node.url})`;
        if (node.closedAt) {
          li.textContent += ` [Closed]`;
        }
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