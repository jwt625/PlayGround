let tabTree = {};

chrome.runtime.onInstalled.addListener(() => {
  chrome.storage.local.get(['tabTree'], (result) => {
    if (result.tabTree) {
      tabTree = result.tabTree;
    }
  });
});

chrome.tabs.onCreated.addListener((tab) => {
  chrome.tabs.get(tab.id, (newTab) => {
    const newNode = {
      id: newTab.id,
      url: newTab.url,
      title: newTab.title,
      createdAt: Date.now(),
      closedAt: null,
      children: []
    };

    if (newTab.openerTabId && tabTree[newTab.openerTabId]) {
      // This is a child tab
      tabTree[newTab.openerTabId].children.push(newTab.id);
      tabTree[newTab.id] = newNode;
    } else {
      // This is a root tab
      tabTree[newTab.id] = newNode;
    }

    saveTabTree();
  });
});

chrome.tabs.onUpdated.addListener((tabId, changeInfo, tab) => {
  if (changeInfo.status === 'complete' && tabTree[tabId]) {
    tabTree[tabId].url = tab.url;
    tabTree[tabId].title = tab.title;
    saveTabTree();
  }
});

chrome.tabs.onRemoved.addListener((tabId) => {
  if (tabTree[tabId]) {
    tabTree[tabId].closedAt = Date.now();
    saveTabTree();
  }
});

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
  }
  return true;  // Indicates that the response is sent asynchronously
});