// Background service worker - handles transcoding via offscreen document

let creatingOffscreen = null;
let offscreenPort = null;
let pendingTranscode = null;
let offscreenReady = false;
let offscreenReadyResolve = null;

async function ensureOffscreenDocument() {
  const offscreenUrl = 'offscreen.html';

  // If we already have a port, we're good
  if (offscreenPort) {
    console.log('[Background] Already have offscreen port');
    return;
  }

  try {
    // Check if offscreen document already exists
    const existingContexts = await chrome.runtime.getContexts({
      contextTypes: ['OFFSCREEN_DOCUMENT'],
      documentUrls: [chrome.runtime.getURL(offscreenUrl)]
    });

    console.log('[Background] Existing offscreen contexts:', existingContexts.length);

    if (existingContexts.length === 0) {
      // Avoid race condition
      if (creatingOffscreen) {
        await creatingOffscreen;
      } else {
        console.log('[Background] Creating offscreen document...');
        creatingOffscreen = chrome.offscreen.createDocument({
          url: offscreenUrl,
          reasons: ['WORKERS'],
          justification: 'FFmpeg WASM transcoding requires workers'
        });

        await creatingOffscreen;
        creatingOffscreen = null;
        console.log('[Background] Offscreen document created successfully');
      }
    }

    // Wait for port connection
    let attempts = 0;
    while (!offscreenPort && attempts < 50) {
      await new Promise(r => setTimeout(r, 100));
      attempts++;
    }

    if (!offscreenPort) {
      throw new Error('Offscreen document did not connect');
    }

    // Wait for READY signal from offscreen
    if (!offscreenReady) {
      console.log('[Background] Waiting for offscreen READY signal...');
      await new Promise((resolve) => {
        offscreenReadyResolve = resolve;
        // Timeout after 5 seconds
        setTimeout(() => {
          if (offscreenReadyResolve) {
            console.warn('[Background] Timeout waiting for READY, proceeding anyway');
            offscreenReadyResolve();
            offscreenReadyResolve = null;
          }
        }, 5000);
      });
    }
    console.log('[Background] Offscreen ready, proceeding');
  } catch (error) {
    console.error('[Background] Error creating offscreen document:', error);
    creatingOffscreen = null;
    throw error;
  }
}

// Listen for port connections from offscreen document
chrome.runtime.onConnect.addListener((port) => {
  console.log('[Background] Port connected:', port.name);

  if (port.name === 'offscreen') {
    offscreenPort = port;
    console.log('[Background] Offscreen port connected!');

    port.onMessage.addListener((message) => {
      console.log('[Background] Port message:', message.type);

      if (message.type === 'READY') {
        console.log('[Background] Offscreen is ready!');
        offscreenReady = true;
        if (offscreenReadyResolve) {
          offscreenReadyResolve();
          offscreenReadyResolve = null;
        }
      }

      if (message.type === 'DEBUG') {
        console.log('[Background] DEBUG from offscreen:', message.msg);
        if (message.stack) {
          console.log('[Background] Stack:', message.stack);
        }
      }

      if (message.type === 'PROGRESS') {
        console.log('[Background] Progress:', message.status, message.progress + '%');
        // Forward progress to content script
        if (message.tabId) {
          chrome.tabs.sendMessage(message.tabId, {
            type: 'TRANSCODE_PROGRESS',
            progress: message.progress,
            status: message.status
          }).catch(() => {});
        }
      }

      if (message.type === 'RESULT') {
        console.log('[Background] Got transcode result via port');
        if (pendingTranscode) {
          if (message.error) {
            pendingTranscode.reject(new Error(message.error));
          } else {
            pendingTranscode.resolve(message.result);
          }
          pendingTranscode = null;
        }
      }
    });

    port.onDisconnect.addListener(() => {
      console.log('[Background] Offscreen port disconnected');
      offscreenPort = null;
    });
  }
});

// Handle messages from content script only
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message.type === 'TRANSCODE_VIDEO') {
    console.log('[Background] Received TRANSCODE_VIDEO from content script');
    handleTranscode(message.videoUrl, sender.tab.id)
      .then(result => {
        console.log('[Background] Sending result back to content script');
        sendResponse(result);
      })
      .catch(error => {
        console.error('[Background] Error:', error);
        sendResponse({ error: error.message });
      });
    return true; // Keep channel open for async response
  }
});

async function handleTranscode(videoUrl, tabId) {
  console.log('[Background] Starting transcode for:', videoUrl);

  try {
    await ensureOffscreenDocument();

    // Create a promise that will be resolved when offscreen sends result
    const resultPromise = new Promise((resolve, reject) => {
      pendingTranscode = { resolve, reject };

      // Timeout after 5 minutes
      setTimeout(() => {
        if (pendingTranscode) {
          pendingTranscode.reject(new Error('Transcoding timed out'));
          pendingTranscode = null;
        }
      }, 5 * 60 * 1000);
    });

    // Send to offscreen document via port
    console.log('[Background] Sending TRANSCODE to offscreen via port...');
    offscreenPort.postMessage({
      type: 'TRANSCODE',
      videoUrl: videoUrl,
      tabId: tabId
    });

    return await resultPromise;
  } catch (error) {
    console.error('[Background] Transcode error:', error);
    throw error;
  }
}

