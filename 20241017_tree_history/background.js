let tabTree = {};

chrome.tabs.onCreated.addListener((tab) => {
  chrome.tabs.get(tab.id, (newTab) => {
    if (newTab.openerTabId) {
      if (!tabTree[newTab.openerTabId]) {
        tabTree[newTab.openerTabId] = [];
      }
      tabTree[newTab.openerTabId].push({
        id: newTab.id,
        url: newTab.url,
        title: newTab.title,
        timestamp: Date.now()
      });
    } else {
      tabTree[newTab.id] = [];
    }
    saveTabTree();
  });
});

chrome.tabs.onUpdated.addListener((tabId, changeInfo, tab) => {
  if (changeInfo.status === 'complete') {
    updateTabInfo(tabId, tab.url, tab.title);
  }
});

chrome.tabs.onRemoved.addListener((tabId) => {
  delete tabTree[tabId];
  for (let parentId in tabTree) {
    tabTree[parentId] = tabTree[parentId].filter(child => child.id !== tabId);
  }
  saveTabTree();
});

function updateTabInfo(tabId, url, title) {
  for (let parentId in tabTree) {
    let childIndex = tabTree[parentId].findIndex(child => child.id === tabId);
    if (childIndex !== -1) {
      tabTree[parentId][childIndex].url = url;
      tabTree[parentId][childIndex].title = title;
      saveTabTree();
      break;
    }
  }
}

function saveTabTree() {
  chrome.storage.local.set({ tabTree: tabTree });
}

chrome.storage.local.get(['tabTree'], (result) => {
  if (result.tabTree) {
    tabTree = result.tabTree;
  }
});