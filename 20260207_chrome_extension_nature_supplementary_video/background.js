// Background service worker - handles transcoding via offscreen document

let creatingOffscreen = null;
let offscreenPort = null;
let pendingTranscode = null;
let offscreenReady = false;
let offscreenReadyResolve = null;

async function ensureOffscreenDocument() {
  const offscreenUrl = 'offscreen.html';

  if (offscreenPort) return;

  const existingContexts = await chrome.runtime.getContexts({
    contextTypes: ['OFFSCREEN_DOCUMENT'],
    documentUrls: [chrome.runtime.getURL(offscreenUrl)]
  });

  if (existingContexts.length === 0) {
    if (creatingOffscreen) {
      await creatingOffscreen;
    } else {
      creatingOffscreen = chrome.offscreen.createDocument({
        url: offscreenUrl,
        reasons: ['WORKERS'],
        justification: 'FFmpeg WASM transcoding requires workers'
      });
      await creatingOffscreen;
      creatingOffscreen = null;
    }
  }

  let attempts = 0;
  while (!offscreenPort && attempts < 50) {
    await new Promise(r => setTimeout(r, 100));
    attempts++;
  }

  if (!offscreenPort) {
    throw new Error('Offscreen document did not connect');
  }

  if (!offscreenReady) {
    await new Promise((resolve) => {
      offscreenReadyResolve = resolve;
      setTimeout(() => {
        if (offscreenReadyResolve) {
          offscreenReadyResolve();
          offscreenReadyResolve = null;
        }
      }, 5000);
    });
  }
}

chrome.runtime.onConnect.addListener((port) => {
  if (port.name === 'offscreen') {
    offscreenPort = port;

    port.onMessage.addListener((message) => {
      if (message.type === 'READY') {
        offscreenReady = true;
        if (offscreenReadyResolve) {
          offscreenReadyResolve();
          offscreenReadyResolve = null;
        }
      }

      if (message.type === 'PROGRESS' && message.tabId) {
        chrome.tabs.sendMessage(message.tabId, {
          type: 'TRANSCODE_PROGRESS',
          progress: message.progress,
          status: message.status
        }).catch(() => {});
      }

      if (message.type === 'RESULT' && pendingTranscode) {
        if (message.error) {
          pendingTranscode.reject(new Error(message.error));
        } else {
          pendingTranscode.resolve(message.result);
        }
        pendingTranscode = null;
      }
    });

    port.onDisconnect.addListener(() => {
      offscreenPort = null;
    });
  }
});

chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message.type === 'TRANSCODE_VIDEO') {
    handleTranscode(message.videoUrl, sender.tab.id)
      .then(result => sendResponse(result))
      .catch(error => sendResponse({ error: error.message }));
    return true;
  }
});

async function handleTranscode(videoUrl, tabId) {
  await ensureOffscreenDocument();

  const resultPromise = new Promise((resolve, reject) => {
    pendingTranscode = { resolve, reject };
    setTimeout(() => {
      if (pendingTranscode) {
        pendingTranscode.reject(new Error('Transcoding timed out'));
        pendingTranscode = null;
      }
    }, 5 * 60 * 1000);
  });

  offscreenPort.postMessage({
    type: 'TRANSCODE',
    videoUrl: videoUrl,
    tabId: tabId
  });

  return await resultPromise;
}

