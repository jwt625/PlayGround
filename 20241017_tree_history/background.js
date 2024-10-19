let tabTree = {};
let excludedDomains = [];
let extensionInitialized = false;
let tabHistory = {};
let userTimeZone = 'UTC'; // Default to UTC

// Load configuration and existing tabTree
chrome.storage.local.get(['config', 'tabTree', 'userTimeZone'], (result) => {
  if (result.config) {
    excludedDomains = result.config.excludedDomains || [];
  }
  if (result.tabTree) {
    tabTree = result.tabTree;
  }
  if (result.userTimeZone) {
    userTimeZone = result.userTimeZone;
  }
  extensionInitialized = true;
});

function isExcluded(url) {
  if (!url) return true; // Exclude empty tabs
  return excludedDomains.some(domain => url.includes(domain));
}

function getHumanReadableTime(timestamp) {
  const date = new Date(timestamp);
  return date.toLocaleString('en-US', { timeZone: userTimeZone, 
    year: 'numeric', month: '2-digit', day: '2-digit', 
    hour: '2-digit', minute: '2-digit', second: '2-digit', 
    hour12: false 
  }).replace(/[/,:]/g, '-');
}

function findNodeById(tree, id) {
  if (tree.id === id) return tree;
  if (tree.children) {
    for (let child of tree.children) {
      const found = findNodeById(child, id);
      if (found) return found;
    }
  }
  return null;
}

function addNodeToTree(node, parentId = null) {
  if (parentId) {
    for (let rootId in tabTree) {
      const parentNode = findNodeById(tabTree[rootId], parentId);
      if (parentNode) {
        if (!parentNode.children) parentNode.children = [];
        parentNode.children.push(node);
        return;
      }
    }
  }
  // If no parent found or parentId is null, add as root node
  tabTree[node.id] = node;
}

function createNode(tab) {
  const timestamp = Date.now();
  return {
    id: `${tab.id}-${timestamp}`, // Unique ID for each node
    tabId: tab.id,
    url: tab.url,
    title: tab.title,
    createdAt: timestamp,
    createdAtHuman: getHumanReadableTime(timestamp),
    closedAt: null,
    closedAtHuman: null,
    children: []
  };
}

function addTabToTree(tab, parentId = null) {
  if (isExcluded(tab.url)) return;

  const newNode = createNode(tab);
  addNodeToTree(newNode, parentId);

  if (!tabHistory[tab.id]) {
    tabHistory[tab.id] = [];
  }
  tabHistory[tab.id].push(newNode);

  saveTabTree();
}

chrome.tabs.onCreated.addListener((tab) => {
  if (!extensionInitialized) return;

  // If openerTabId is set, use it as the parent
  if (tab.openerTabId) {
    const parentHistory = tabHistory[tab.openerTabId];
    if (parentHistory && parentHistory.length > 0) {
      addTabToTree(tab, parentHistory[parentHistory.length - 1].id);
    } else {
      addTabToTree(tab);
    }
  } else {
    addTabToTree(tab);
  }
});

chrome.tabs.onUpdated.addListener((tabId, changeInfo, tab) => {
  if (!extensionInitialized || changeInfo.status !== 'complete' || isExcluded(tab.url)) return;

  const currentTabHistory = tabHistory[tabId];
  if (currentTabHistory && currentTabHistory.length > 0) {
    const lastNode = currentTabHistory[currentTabHistory.length - 1];
    if (lastNode.url !== tab.url) {
      // This is navigation within the same tab
      addTabToTree(tab, lastNode.id);
    }
  } else {
    // This is a new tab that we haven't seen before
    addTabToTree(tab);
  }
});

chrome.webNavigation.onCommitted.addListener((details) => {
  if (!extensionInitialized || details.frameId !== 0 || isExcluded(details.url)) return;

  if (details.transitionType === 'reload' || details.transitionQualifiers.includes('forward_back')) {
    const history = tabHistory[details.tabId];
    if (history) {
      const index = history.findIndex(node => node.url === details.url);
      if (index !== -1) {
        // We've navigated back to a previous page
        tabHistory[details.tabId] = history.slice(0, index + 1);
        saveTabTree();
      }
    }
  }
});

chrome.tabs.onRemoved.addListener((tabId) => {
  if (!extensionInitialized) return;

  updateTabClosedTime(tabId);
  delete tabHistory[tabId];
});

function updateTabClosedTime(tabId) {
  function closeNode(node) {
    if (node.tabId === tabId) {
      const timestamp = Date.now();
      node.closedAt = timestamp;
      node.closedAtHuman = getHumanReadableTime(timestamp);
      return true;
    }
    if (node.children) {
      for (let child of node.children) {
        if (closeNode(child)) {
          return true;
        }
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
    tabHistory = {};
    saveTabTree();
    sendResponse({success: true});
  } else if (request.action === "getTabTree") {
    sendResponse({tabTree: tabTree});
  } else if (request.action === "updateConfig") {
    chrome.storage.local.set({ config: request.config }, () => {
      excludedDomains = request.config.excludedDomains || [];
      sendResponse({success: true});
    });
  } else if (request.action === "updateTimeZone") {
    userTimeZone = request.timeZone;
    chrome.storage.local.set({ userTimeZone: userTimeZone }, () => {
      sendResponse({success: true});
    });
  }
  return true;  // Indicates that the response is sent asynchronously
});