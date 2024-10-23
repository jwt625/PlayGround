
// State variables
let tabTree = {};
let excludedDomains = [];
let extensionInitialized = false;
let tabHistory = {};
let userTimeZone = 'UTC'; // Default to UTC
let isTracking = false; // New tracking state
let trackingCheckInterval = null;

// Initialize tracking check interval
function initTrackingCheck() {
  if (trackingCheckInterval) {
    clearInterval(trackingCheckInterval);
  }
  
  trackingCheckInterval = setInterval(() => {
    if (isTracking) {
      // Verify tracking state and reinitialize if needed
      chrome.storage.local.get(['isTracking'], (result) => {
        if (result.isTracking !== isTracking) {
          console.log('Tracking state mismatch detected, reinitializing...');
          initializeExtension();
        }
      });
    }
  }, 60000); // Check every minute
}

// Function to update extension icon
function updateIcon(tracking) {
  const iconPath = tracking ? {
    16: 'icons/active_16.png',
    32: 'icons/active_32.png',
    48: 'icons/active_48.png',
    128: 'icons/active_128.png'
  } : {
    16: 'icons/inactive_16.png',
    32: 'icons/inactive_32.png',
    48: 'icons/inactive_48.png',
    128: 'icons/inactive_128.png'
  };
  
  chrome.action.setIcon({ path: iconPath });
}

// Single initialization function
async function initializeExtension() {
  try {
    const result = await chrome.storage.local.get(['config', 'tabTree', 'userTimeZone', 'isTracking']);
    
    if (result.config) {
      excludedDomains = result.config.excludedDomains || [];
    }
    if (result.tabTree) {
      tabTree = result.tabTree;
    }
    if (result.userTimeZone) {
      userTimeZone = result.userTimeZone;
    }
    isTracking = result.isTracking || false;
    
    updateIcon(isTracking);
    initTrackingCheck();
    extensionInitialized = true;
    
    // Log initialization status
    console.log('Extension initialized:', {
      isTracking,
      domainsCount: excludedDomains.length,
      treeSize: Object.keys(tabTree).length
    });
  } catch (error) {
    console.error('Initialization error:', error);
  }
}

// Initialize event listeners
function initializeEventListeners() {
  // Tab creation listener
  chrome.tabs.onCreated.addListener((tab) => {
    if (!extensionInitialized || !isTracking) return;
    
    try {
      chrome.storage.local.set({ [`opener_${tab.id}`]: tab.openerTabId });
      
      function updateListener(tabId, info, updatedTab) {
        if (tabId === tab.id && info.status === 'complete') {
          chrome.storage.local.get(`opener_${tab.id}`, async (result) => {
            const openerTabId = result[`opener_${tab.id}`];
            if (openerTabId) {
              const parentHistory = tabHistory[openerTabId];
              if (parentHistory && parentHistory.length > 0) {
                await addTabToTree(updatedTab, parentHistory[parentHistory.length - 1].id);
              } else {
                await addTabToTree(updatedTab);
              }
              chrome.storage.local.remove(`opener_${tab.id}`);
            } else {
              await addTabToTree(updatedTab);
            }
          });
          chrome.tabs.onUpdated.removeListener(updateListener);
        }
      }
      
      chrome.tabs.onUpdated.addListener(updateListener);
    } catch (error) {
      console.error('Error in onCreated listener:', error);
    }
  });

  // Tab update listener
  chrome.tabs.onUpdated.addListener((tabId, changeInfo, tab) => {
    if (!extensionInitialized || !isTracking || changeInfo.status !== 'complete' || isExcluded(tab.url)) return;
    
    const currentTabHistory = tabHistory[tabId];
    if (currentTabHistory?.length > 0) {
      const lastNode = currentTabHistory[currentTabHistory.length - 1];
      if (lastNode.url !== tab.url) {
        const existingIndex = currentTabHistory.findIndex(node => node.url === tab.url);
        if (existingIndex !== -1) {
          handleExistingTab(currentTabHistory, existingIndex, tab, tabId);
        } else {
          addTabToTree(tab, lastNode.id);
        }
      } else if (lastNode.title !== tab.title) {
        updateNodeTitle(lastNode, tab.title);
      }
    } else {
      addTabToTree(tab);
    }
  });

  // Navigation listener
  chrome.webNavigation.onCommitted.addListener((details) => {
    if (!extensionInitialized || !isTracking || details.frameId !== 0 || isExcluded(details.url)) return;
    
    if (details.transitionType === 'link' && details.transitionQualifiers.includes('forward_back')) {
      handleNavigationChange(details);
    }
  });

  // Tab removal listener
  chrome.tabs.onRemoved.addListener((tabId) => {
    if (!extensionInitialized || !isTracking) return;
    updateTabClosedTime(tabId);
    delete tabHistory[tabId];
  });

  // Message listener
  chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
    if (request.action === "toggleTracking") {
      toggleTracking()
        .then(response => sendResponse(response))
        .catch(error => sendResponse({ error: error.message }));
      return true;
    }
    
    // Handle other messages
    handleOtherMessages(request, sendResponse);
    return true;
  });

  // Alarm for state verification
  chrome.alarms.create('verifyTrackingState', { periodInMinutes: 1 });
  chrome.alarms.onAlarm.addListener((alarm) => {
    if (alarm.name === 'verifyTrackingState') {
      chrome.storage.local.get(['isTracking'], (result) => {
        if (result.isTracking !== isTracking) {
          console.log('Tracking state mismatch detected, reinitializing...');
          initializeExtension();
        }
      });
    }
  });
}

// Start initialization
initializeExtension().then(() => {
  initializeEventListeners();
});

// Initialize on install or update
chrome.runtime.onInstalled.addListener(() => {
  initializeExtension().then(() => {
    initializeEventListeners();
  });
});

// [Rest of your existing helper functions remain the same]
// (isExcluded, getHumanReadableTime, findNodeById, addNodeToTree, createNode,
// addTabToTree, analyzePageContent, getWordFrequency, updateTabClosedTime,
// saveTabTree, etc.)


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

// Add this function to check for duplicate nodes
function isDuplicateNode(newNode, existingNode) {
  return newNode.url === existingNode.url && 
         Math.abs(newNode.createdAt - existingNode.createdAt) < 1000; // Within 1 second
}


function addNodeToTree(node, parentId = null) {
  if (parentId) {
    for (let rootId in tabTree) {
      const parentNode = findNodeById(tabTree[rootId], parentId);
      if (parentNode) {
        if (!parentNode.children) parentNode.children = [];
          // Check for duplicates in children
          const isDuplicate = parentNode.children.some(child => 
            isDuplicateNode(node, child)
          );
          
          if (!isDuplicate) {
            parentNode.children.push(node);
        }
        return;
      }
    }
  }
  // If no parent found or parentId is null, add as root node
  // Check for duplicates in root nodes
  const isDuplicate = Object.values(tabTree).some(existingNode => 
    isDuplicateNode(node, existingNode)
  );
  
  if (!isDuplicate) {
    tabTree[node.id] = node;
  }
}

// Update createNode function to include word frequency
function createNode(tab) {
  const timestamp = Date.now();
  return {
    id: `${tab.id}-${timestamp}`,
    tabId: tab.id,
    url: tab.url,
    title: tab.title,
    createdAt: timestamp,
    createdAtHuman: getHumanReadableTime(timestamp),
    closedAt: null,
    closedAtHuman: null,
    children: [],
    topWords: null // Will be populated after analysis
  };
}

// Modified addTabToTree function with debouncing
let addTabDebounceTimers = {};
// Update addTabToTree to include word frequency analysis
async function addTabToTree(tab, parentId = null) {
  if (isExcluded(tab.url)) return;

  // Clear any existing timer for this tab
  if (addTabDebounceTimers[tab.id]) {
    clearTimeout(addTabDebounceTimers[tab.id]);
  }

  // Set a new timer
  addTabDebounceTimers[tab.id] = setTimeout(async () => {
    const newNode = createNode(tab);
    
    // Analyze page content
    const wordFrequency = await analyzePageContent(tab.id);
    if (wordFrequency) {
      newNode.topWords = wordFrequency;
    }

    addNodeToTree(newNode, parentId);

    if (!tabHistory[tab.id]) {
      tabHistory[tab.id] = [];
    }
    
    // Check for duplicates in tab history
    const isDuplicate = tabHistory[tab.id].some(historyNode => 
      isDuplicateNode(newNode, historyNode)
    );
    
    if (!isDuplicate) {
      tabHistory[tab.id].push(newNode);
      saveTabTree();
    }
    
    delete addTabDebounceTimers[tab.id];
  }, 100); // 100ms debounce time
}

chrome.tabs.onCreated.addListener((tab) => {
  if (!extensionInitialized || !isTracking) return; // Add tracking check

  // Store the opener tab ID for later use
  chrome.storage.local.set({ [`opener_${tab.id}`]: tab.openerTabId });

  // Don't add the tab immediately, wait for it to load
  chrome.tabs.onUpdated.addListener(function listener(tabId, info, updatedTab) {
    if (tabId === tab.id && info.status === 'complete') {
      chrome.storage.local.get(`opener_${tab.id}`, (result) => {
        const openerTabId = result[`opener_${tab.id}`];
        if (openerTabId) {
          const parentHistory = tabHistory[openerTabId];
          if (parentHistory && parentHistory.length > 0) {
            addTabToTree(updatedTab, parentHistory[parentHistory.length - 1].id);
          } else {
            addTabToTree(updatedTab);
          }
          chrome.storage.local.remove(`opener_${tab.id}`);
        } else {
          addTabToTree(updatedTab);
        }
      });
      chrome.tabs.onUpdated.removeListener(listener);
    }
  });
});

chrome.tabs.onUpdated.addListener((tabId, changeInfo, tab) => {
  if (!extensionInitialized || !isTracking || changeInfo.status !== 'complete' || isExcluded(tab.url)) return;

  const currentTabHistory = tabHistory[tabId];
  if (currentTabHistory && currentTabHistory.length > 0) {
    const lastNode = currentTabHistory[currentTabHistory.length - 1];
    if (lastNode.url !== tab.url) {
      // Check if we're navigating back to a previous page
      const existingIndex = currentTabHistory.findIndex(node => node.url === tab.url);
      if (existingIndex !== -1) {
        // We're navigating back, truncate the history and update close times
        const timestamp = Date.now();
        for (let i = existingIndex + 1; i < currentTabHistory.length; i++) {
          currentTabHistory[i].closedAt = timestamp;
          currentTabHistory[i].closedAtHuman = getHumanReadableTime(timestamp);
        }
        tabHistory[tabId] = currentTabHistory.slice(0, existingIndex + 1);
        updateNodeTitle(tabHistory[tabId][existingIndex], tab.title);
      } else {
        // This is navigation to a new page
        addTabToTree(tab, lastNode.id);
      }
    } else if (lastNode.title !== tab.title) {
      // Update the title if it has changed
      updateNodeTitle(lastNode, tab.title);
    }
  } else {
    // This is a new tab that we haven't seen before
    addTabToTree(tab);
  }
});

chrome.webNavigation.onCommitted.addListener((details) => {
  if (!extensionInitialized || !isTracking || details.frameId !== 0 || isExcluded(details.url)) return;

  if (details.transitionType === 'link' && details.transitionQualifiers.includes('forward_back')) {
    // This is likely a "Go back" or "Go forward" action
    const history = tabHistory[details.tabId];
    if (history) {
      const index = history.findIndex(node => node.url === details.url);
      if (index !== -1) {
        const timestamp = Date.now();
        // Update close times for nodes we're moving past
        if (index < history.length - 1) {
          for (let i = index + 1; i < history.length; i++) {
            history[i].closedAt = timestamp;
            history[i].closedAtHuman = getHumanReadableTime(timestamp);
          }
        }
        tabHistory[details.tabId] = history.slice(0, index + 1);
        saveTabTree();
      }
    }
  }
});

chrome.tabs.onRemoved.addListener((tabId) => {
  if (!extensionInitialized || !isTracking) return;
  
  if (addTabDebounceTimers[tabId]) {
    clearTimeout(addTabDebounceTimers[tabId]);
    delete addTabDebounceTimers[tabId];
  }

  updateTabClosedTime(tabId);
  delete tabHistory[tabId];
});


// Add functions for word frequency analysis
async function analyzePageContent(tabId) {
  if (!isTracking) return null;
  
  // Inject content script to analyze the page
  try {
    const [{ result }] = await chrome.scripting.executeScript({
      target: { tabId: tabId },
      func: getWordFrequency,
    });
    return result;
  } catch (error) {
    console.error('Error analyzing page content:', error);
    return null;
  }
}



// Function to be injected into the page
function getWordFrequency() {
  // Comprehensive list of English stop words
  const stopWords = new Set([
    // Articles and basic prepositions
    'a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'by', 'for', 'with', 'about',
    'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below',
    'to', 'from', 'up', 'down', 'of', 'off',
    
    // Pronouns and their variants
    'i', 'me', 'my', 'mine', 'myself',
    'you', 'your', 'yours', 'yourself', 'yourselves',
    'he', 'him', 'his', 'himself',
    'she', 'her', 'hers', 'herself',
    'it', 'its', 'itself',
    'we', 'us', 'our', 'ours', 'ourselves',
    'they', 'them', 'their', 'theirs', 'themselves',
    'this', 'that', 'these', 'those',
    'who', 'whom', 'whose', 'which', 'what',
    
    // Verbs and verb forms
    'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
    'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing',
    'would', 'should', 'could', 'might', 'must', 'can', 'will',
    'shall', 'may', 'ought',
    
    // Common contractions
    "i'm", "i've", "i'll", "i'd",
    "you're", "you've", "you'll", "you'd",
    "he's", "he'll", "he'd",
    "she's", "she'll", "she'd",
    "it's", "it'll", "it'd",
    "we're", "we've", "we'll", "we'd",
    "they're", "they've", "they'll", "they'd",
    "that's", "that'll", "that'd",
    "who's", "who'll", "who'd",
    "what's", "what're", "what'll", "what'd",
    "where's", "where'll", "where'd",
    "when's", "when'll", "when'd",
    "why's", "why'll", "why'd",
    "how's", "how'll", "how'd",
    "ain't", "isn't", "aren't", "wasn't", "weren't",
    "hasn't", "haven't", "hadn't",
    "doesn't", "don't", "didn't",
    "won't", "wouldn't", "shan't", "shouldn't",
    "can't", "cannot", "couldn't",
    "mustn't", "mightn't",
    
    // Common adverbs and adjectives
    'just', 'very', 'quite', 'rather', 'somewhat',
    'more', 'most', 'much', 'many', 'some', 'few', 'all', 'any', 'enough',
    'such', 'same', 'different', 'other', 'another', 'each', 'every', 'either',
    'neither', 'several', 'both', 'else',
    'here', 'there', 'where', 'when', 'why', 'how',
    'again', 'ever', 'never', 'always', 'sometimes', 'often', 'usually',
    'already', 'still', 'now', 'then', 'once', 'twice',
    'only', 'even', 'also', 'too', 'instead', 'rather',
    
    // Miscellaneous common words
    'like', 'well', 'back', 'there', 'still', 'yet', 'else', 'further',
    'since', 'while', 'whether', 'though', 'although', 'unless',
    'however', 'moreover', 'therefore', 'hence', 'furthermore',
    'otherwise', 'nevertheless', 'meanwhile', 'afterward', 'afterwards',
    'yes', 'no', 'not', 'nor', 'none', 'nothing', 'nobody',
    'anywhere', 'everywhere', 'somewhere', 'nowhere',
    'among', 'beside', 'besides', 'beyond', 'within', 'without'
  ]);

  // Get all text content from the page
  const text = document.body.innerText;
  
  // Split into words, convert to lowercase, and filter
  const words = text.toLowerCase()
    .replace(/[^a-z0-9\s]/g, '') // Remove punctuation and special characters
    .split(/\s+/) // Split on whitespace
    .filter(word => 
      word.length > 2 && // Skip very short words
      !stopWords.has(word) && // Skip stop words
      !/^\d+$/.test(word) // Skip pure numbers
    );

  // Count word frequencies
  const frequencyMap = {};
  words.forEach(word => {
    frequencyMap[word] = (frequencyMap[word] || 0) + 1;
  });

  // Get top 5 words by frequency
  return Object.entries(frequencyMap)
    .sort(([,a], [,b]) => b - a)
    .slice(0, 5)
    .map(([word, count]) => ({ word, count }));
}



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
  if (request.action === "toggleTracking") {
    isTracking = !isTracking;
    updateIcon(isTracking); // update icon
    chrome.storage.local.set({ isTracking: isTracking });
    sendResponse({ isTracking: isTracking });
  } else if (request.action === "getTrackingStatus") {
    sendResponse({ isTracking: isTracking });
  } 

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