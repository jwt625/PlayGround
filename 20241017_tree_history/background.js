let tabTree = {};
let excludedDomains = [];
let extensionInitialized = false;

// Load configuration
chrome.storage.local.get(['config'], (result) => {
  if (result.config) {
    excludedDomains = result.config.excludedDomains || [];
  }
});

chrome.runtime.onInstalled.addListener(() => {
  chrome.storage.local.get(['tabTree', 'config'], (result) => {
    if (result.config) {
      excludedDomains = result.config.excludedDomains || [];
    }
    // Initialize with an empty tabTree
    tabTree = {};
    saveTabTree();
    extensionInitialized = true;
  });
});

function isExcluded(url) {
  if (!url) return true; // Exclude empty tabs
  return excludedDomains.some(domain => url.includes(domain));
}

function addTabToTree(tab) {
  if (isExcluded(tab.url)) return;

  const newNode = {
    id: tab.id,
    url: tab.url,
    title: tab.title,
    createdAt: Date.now(),
    closedAt: null,
    children: []
  };

  if (tab.openerTabId && tabTree[tab.openerTabId]) {
    // This is a child tab
    tabTree[tab.openerTabId].children.push(newNode);
  } else {
    // This is a root tab
    tabTree[tab.id] = newNode;
  }
}

chrome.tabs.onCreated.addListener((tab) => {
  if (!extensionInitialized) return;

  // Don't add the tab immediately, wait for it to load
  chrome.tabs.onUpdated.addListener(function listener(tabId, info) {
    if (tabId === tab.id && info.status === 'complete') {
      chrome.tabs.get(tabId, (updatedTab) => {
        if (!isExcluded(updatedTab.url)) {
          addTabToTree(updatedTab);
          saveTabTree();
        }
      });
      chrome.tabs.onUpdated.removeListener(listener);
    }
  });
});

chrome.tabs.onActivated.addListener((activeInfo) => {
  if (!extensionInitialized) return;

  if (!tabTree[activeInfo.tabId]) {
    // If the activated tab is not in our tree, add it
    chrome.tabs.get(activeInfo.tabId, (tab) => {
      if (!isExcluded(tab.url)) {
        addTabToTree(tab);
        saveTabTree();
      }
    });
  }
});

chrome.tabs.onUpdated.addListener((tabId, changeInfo, tab) => {
  if (!extensionInitialized) return;

  if (changeInfo.status === 'complete' && !isExcluded(tab.url)) {
    updateTabInfo(tabId, tab.url, tab.title);
  }
});

chrome.tabs.onRemoved.addListener((tabId) => {
  if (!extensionInitialized) return;

  updateTabClosedTime(tabId);
});

function updateTabInfo(tabId, url, title) {
  function updateNode(node) {
    if (node.id === tabId) {
      node.url = url;
      node.title = title;
      return true;
    }
    for (let child of node.children) {
      if (updateNode(child)) {
        return true;
      }
    }
    return false;
  }

  for (let rootId in tabTree) {
    if (updateNode(tabTree[rootId])) {
      saveTabTree();
      return;
    }
  }
}

function updateTabClosedTime(tabId) {
  function closeNode(node) {
    if (node.id === tabId) {
      node.closedAt = Date.now();
      return true;
    }
    for (let child of node.children) {
      if (closeNode(child)) {
        return true;
      }
    }
    return false;
  }

  for (let rootId in tabTree) {
    if (closeNode(tabTree[rootId])) {
      saveTabTree();
      return;
    }
  }
}

function saveTabTree() {
  chrome.storage.local.set({ tabTree: tabTree });
}

chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  if (request.action === "clearTabTree") {
    tabTree = {};
    saveTabTree();
    sendResponse({success: true});
  } else if (request.action === "getTabTree") {
    sendResponse({tabTree: tabTree});
  } else if (request.action === "updateConfig") {
    chrome.storage.local.set({ config: request.config }, () => {
      excludedDomains = request.config.excludedDomains || [];
      sendResponse({success: true});
    });
  }
  return true;  // Indicates that the response is sent asynchronously
});