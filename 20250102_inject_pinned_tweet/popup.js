document.getElementById('saveButton').addEventListener('click', async () => {
  const profileUrl = document.getElementById('profileUrl').value;
  
  // Save to Chrome storage
  await chrome.storage.sync.set({
    profileUrl: profileUrl
  });
  
  // Show success message
  alert('Settings saved!');
});

// Load saved settings
chrome.storage.sync.get(['profileUrl'], (result) => {
  if (result.profileUrl) {
    document.getElementById('profileUrl').value = result.profileUrl;
  }
});