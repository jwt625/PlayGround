// Background service worker - handles transcoding via offscreen document

let creatingOffscreen = null;
let offscreenPort = null;
let pendingTranscode = null;
let offscreenReady = false;
let offscreenReadyResolve = null;

// Storage for chunked transfers
const chunkStorage = new Map(); // transferId -> { chunks: [], totalChunks: number }
// Storage for results that need chunked retrieval
const resultStorage = new Map(); // transferId -> { base64, mimeType }

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

// Storage for chunked results from offscreen
let chunkedResult = null;

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

      // Handle chunked results from offscreen
      if (message.type === 'RESULT_CHUNKED_START') {
        chunkedResult = {
          mimeType: message.mimeType,
          isMultipleVideos: message.isMultipleVideos || false,
          totalChunks: message.totalChunks,
          chunks: [],
          received: 0
        };
      }

      if (message.type === 'RESULT_CHUNK' && chunkedResult) {
        chunkedResult.chunks[message.chunkIndex] = message.chunk;
        chunkedResult.received++;
      }

      if (message.type === 'RESULT_CHUNKED_END' && chunkedResult && pendingTranscode) {
        const combined = chunkedResult.chunks.join('');

        let result;
        if (chunkedResult.isMultipleVideos) {
          // Parse the JSON for multiple videos
          result = JSON.parse(combined);
        } else {
          // Single video base64 data
          result = {
            base64: combined,
            mimeType: chunkedResult.mimeType
          };
        }
        chunkedResult = null;
        pendingTranscode.resolve(result);
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

  if (message.type === 'EXTRACT_ZIP') {
    handleExtractZip(message.zipUrl, sender.tab.id)
      .then(result => sendResponse(result))
      .catch(error => sendResponse({ error: error.message }));
    return true;
  }

  if (message.type === 'EXTRACT_ZIP_DATA') {
    handleExtractZipData(message.zipBase64, sender.tab.id)
      .then(result => sendResponse(result))
      .catch(error => sendResponse({ error: error.message }));
    return true;
  }

  if (message.type === 'EXTRACT_ZIP_CHUNK') {
    handleExtractZipChunk(message.transferId, message.chunkIndex, message.totalChunks, message.chunk)
      .then(result => sendResponse(result))
      .catch(error => sendResponse({ error: error.message }));
    return true;
  }

  if (message.type === 'EXTRACT_ZIP_PROCESS') {
    handleExtractZipProcess(message.transferId, sender.tab.id)
      .then(result => sendResponse(result))
      .catch(error => sendResponse({ error: error.message }));
    return true;
  }

  if (message.type === 'GET_RESULT_CHUNK') {
    handleGetResultChunk(message.resultId, message.chunkIndex)
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

async function handleExtractZip(zipUrl, tabId) {
  await ensureOffscreenDocument();

  const resultPromise = new Promise((resolve, reject) => {
    pendingTranscode = { resolve, reject };
    setTimeout(() => {
      if (pendingTranscode) {
        pendingTranscode.reject(new Error('ZIP extraction timed out'));
        pendingTranscode = null;
      }
    }, 5 * 60 * 1000);
  });

  offscreenPort.postMessage({
    type: 'EXTRACT_ZIP',
    zipUrl: zipUrl,
    tabId: tabId
  });

  return await resultPromise;
}

async function handleExtractZipData(zipBase64, tabId) {
  await ensureOffscreenDocument();

  const resultPromise = new Promise((resolve, reject) => {
    pendingTranscode = { resolve, reject };
    setTimeout(() => {
      if (pendingTranscode) {
        pendingTranscode.reject(new Error('ZIP extraction timed out'));
        pendingTranscode = null;
      }
    }, 5 * 60 * 1000);
  });

  offscreenPort.postMessage({
    type: 'EXTRACT_ZIP_DATA',
    zipBase64: zipBase64,
    tabId: tabId
  });

  const result = await resultPromise;

  // Check if result is too large to send in one message (>32MB base64)
  const MAX_RESPONSE_SIZE = 32 * 1024 * 1024;
  if (result.base64 && result.base64.length > MAX_RESPONSE_SIZE) {
    // Store result for chunked retrieval
    const resultId = Date.now().toString() + '_result';
    resultStorage.set(resultId, result);

    const totalChunks = Math.ceil(result.base64.length / MAX_RESPONSE_SIZE);
    return {
      chunked: true,
      resultId: resultId,
      totalChunks: totalChunks,
      totalLength: result.base64.length,
      mimeType: result.mimeType
    };
  }

  return result;
}

async function handleExtractZipChunk(transferId, chunkIndex, totalChunks, chunk) {
  if (!chunkStorage.has(transferId)) {
    chunkStorage.set(transferId, { chunks: new Array(totalChunks), totalChunks, received: 0 });
  }

  const storage = chunkStorage.get(transferId);
  storage.chunks[chunkIndex] = chunk;
  storage.received++;


  return { success: true, received: storage.received, total: totalChunks };
}

async function handleExtractZipProcess(transferId, tabId) {
  const storage = chunkStorage.get(transferId);
  if (!storage) {
    throw new Error('Transfer not found: ' + transferId);
  }

  if (storage.received !== storage.totalChunks) {
    throw new Error(`Incomplete transfer: received ${storage.received}/${storage.totalChunks} chunks`);
  }

  // Combine all chunks
  const zipBase64 = storage.chunks.join('');
  chunkStorage.delete(transferId); // Clean up


  // Now process like normal
  await ensureOffscreenDocument();

  const resultPromise = new Promise((resolve, reject) => {
    pendingTranscode = { resolve, reject };
    setTimeout(() => {
      if (pendingTranscode) {
        pendingTranscode.reject(new Error('ZIP extraction timed out'));
        pendingTranscode = null;
      }
    }, 10 * 60 * 1000); // 10 minutes for large files
  });

  // Send to offscreen, chunking if necessary
  const MAX_CHUNK_SIZE = 32 * 1024 * 1024;
  if (zipBase64.length > MAX_CHUNK_SIZE) {
    const totalChunks = Math.ceil(zipBase64.length / MAX_CHUNK_SIZE);

    offscreenPort.postMessage({
      type: 'EXTRACT_ZIP_DATA_START',
      totalChunks: totalChunks,
      totalLength: zipBase64.length,
      tabId: tabId
    });

    for (let i = 0; i < totalChunks; i++) {
      const start = i * MAX_CHUNK_SIZE;
      const end = Math.min(start + MAX_CHUNK_SIZE, zipBase64.length);
      const chunk = zipBase64.substring(start, end);
      offscreenPort.postMessage({
        type: 'EXTRACT_ZIP_DATA_CHUNK',
        chunkIndex: i,
        chunk: chunk
      });
    }

    offscreenPort.postMessage({
      type: 'EXTRACT_ZIP_DATA_END',
      tabId: tabId
    });
  } else {
    offscreenPort.postMessage({
      type: 'EXTRACT_ZIP_DATA',
      zipBase64: zipBase64,
      tabId: tabId
    });
  }

  const result = await resultPromise;

  // Check if result is too large to send in one message (>32MB)
  const MAX_RESPONSE_SIZE = 32 * 1024 * 1024;

  // Calculate total size
  let totalSize = 0;
  if (result.base64) {
    totalSize = result.base64.length;
  } else if (result.multiple && result.videos) {
    totalSize = result.videos.reduce((sum, v) => sum + v.base64.length, 0);
  }

  if (totalSize > MAX_RESPONSE_SIZE) {
    // Store result for chunked retrieval
    const resultId = transferId + '_result';

    if (result.multiple && result.videos) {
      // For multiple videos, serialize to JSON
      const jsonStr = JSON.stringify(result);
      resultStorage.set(resultId, { json: jsonStr, isMultipleVideos: true });

      const totalChunks = Math.ceil(jsonStr.length / MAX_RESPONSE_SIZE);
      return {
        chunked: true,
        resultId: resultId,
        totalChunks: totalChunks,
        totalLength: jsonStr.length,
        isMultipleVideos: true
      };
    } else {
      // Single video
      resultStorage.set(resultId, result);

      const totalChunks = Math.ceil(result.base64.length / MAX_RESPONSE_SIZE);
      return {
        chunked: true,
        resultId: resultId,
        totalChunks: totalChunks,
        totalLength: result.base64.length,
        mimeType: result.mimeType
      };
    }
  }

  return result;
}

async function handleGetResultChunk(resultId, chunkIndex) {
  const stored = resultStorage.get(resultId);
  if (!stored) {
    throw new Error('Result not found: ' + resultId);
  }

  const MAX_CHUNK_SIZE = 32 * 1024 * 1024;

  // Handle both single video (base64) and multiple videos (json)
  const data = stored.isMultipleVideos ? stored.json : stored.base64;
  const start = chunkIndex * MAX_CHUNK_SIZE;
  const end = Math.min(start + MAX_CHUNK_SIZE, data.length);
  const chunk = data.substring(start, end);
  const isLast = end >= data.length;

  // Clean up if this is the last chunk
  if (isLast) {
    resultStorage.delete(resultId);
  }

  return { chunk, isLast };
}

