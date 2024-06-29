chrome.webNavigation.onCompleted.addListener(function(details) {
    if (details.url.includes("onshape.com")) {
      chrome.scripting.executeScript({
        target: { tabId: details.tabId },
        files: ['content.js']
      });
    }
  }, { url: [{ urlMatches: 'https://.*\\.onshape\\.com/.*' }] });
  