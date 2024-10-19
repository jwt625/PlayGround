// Saves options to chrome.storage
function save_options() {
  var excludedDomains = document.getElementById('excludedDomains').value.split('\n').map(s => s.trim()).filter(Boolean);
  chrome.storage.local.set({
    config: { excludedDomains: excludedDomains }
  }, function() {
    // Update status to let user know options were saved.
    var status = document.getElementById('status');
    status.textContent = 'Options saved.';
    setTimeout(function() {
      status.textContent = '';
    }, 750);

    // Send message to background script to update config
    chrome.runtime.sendMessage({action: "updateConfig", config: { excludedDomains: excludedDomains }});
  });
}

// Restores select box and checkbox state using the preferences
// stored in chrome.storage.
function restore_options() {
  chrome.storage.local.get({
    config: { excludedDomains: [] }
  }, function(items) {
    document.getElementById('excludedDomains').value = items.config.excludedDomains.join('\n');
  });
}

document.addEventListener('DOMContentLoaded', restore_options);
document.getElementById('save').addEventListener('click', save_options);