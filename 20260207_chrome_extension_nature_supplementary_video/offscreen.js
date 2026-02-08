// Offscreen document - runs FFmpeg transcoding and ZIP extraction in extension context

let ffmpeg = null;
let ffmpegLoaded = false;
let port = null;

// Video extensions
const NATIVE_FORMATS = ['mp4', 'webm', 'ogg', 'm4v'];
const TRANSCODE_FORMATS = ['avi', 'mkv', 'flv', 'wmv', 'mov'];

// Max message size for port communication (32MB to stay under 64MB limit)
const MAX_MESSAGE_SIZE = 32 * 1024 * 1024;

// Send result, chunking if necessary
function sendResult(result) {
  // Calculate total size of result
  let totalSize = 0;
  if (result.base64) {
    totalSize = result.base64.length;
  } else if (result.multiple && result.videos) {
    totalSize = result.videos.reduce((sum, v) => sum + v.base64.length, 0);
  }

  // If small enough, send directly
  if (totalSize <= MAX_MESSAGE_SIZE) {
    port.postMessage({ type: 'RESULT', result });
    return;
  }

  // For multiple videos, serialize the whole result to JSON and chunk it
  if (result.multiple && result.videos) {
    const jsonStr = JSON.stringify(result);
    const totalChunks = Math.ceil(jsonStr.length / MAX_MESSAGE_SIZE);

    port.postMessage({
      type: 'RESULT_CHUNKED_START',
      isMultipleVideos: true,
      totalChunks: totalChunks,
      totalLength: jsonStr.length
    });

    for (let i = 0; i < totalChunks; i++) {
      const start = i * MAX_MESSAGE_SIZE;
      const end = Math.min(start + MAX_MESSAGE_SIZE, jsonStr.length);
      const chunk = jsonStr.substring(start, end);
      port.postMessage({
        type: 'RESULT_CHUNK',
        chunkIndex: i,
        chunk: chunk
      });
    }

    port.postMessage({ type: 'RESULT_CHUNKED_END' });
    return;
  }

  // Single video - chunk the base64 data
  const totalChunks = Math.ceil(result.base64.length / MAX_MESSAGE_SIZE);

  port.postMessage({
    type: 'RESULT_CHUNKED_START',
    mimeType: result.mimeType,
    totalChunks: totalChunks,
    totalLength: result.base64.length
  });

  for (let i = 0; i < totalChunks; i++) {
    const start = i * MAX_MESSAGE_SIZE;
    const end = Math.min(start + MAX_MESSAGE_SIZE, result.base64.length);
    const chunk = result.base64.substring(start, end);
    port.postMessage({
      type: 'RESULT_CHUNK',
      chunkIndex: i,
      chunk: chunk
    });
  }

  port.postMessage({ type: 'RESULT_CHUNKED_END' });
}

async function loadFFmpeg() {
  if (ffmpegLoaded) return ffmpeg;

  if (typeof FFmpegWASM === 'undefined') {
    throw new Error('FFmpegWASM is not defined');
  }

  const { FFmpeg } = FFmpegWASM;
  ffmpeg = new FFmpeg();

  await ffmpeg.load({
    coreURL: chrome.runtime.getURL('ffmpeg/ffmpeg-core.js'),
    wasmURL: chrome.runtime.getURL('ffmpeg/ffmpeg-core.wasm'),
  });

  ffmpegLoaded = true;
  return ffmpeg;
}

function getExtension(url) {
  const match = url.toLowerCase().match(/\.(avi|mkv|flv|wmv|mov|mp4|webm)/);
  return match ? match[1] : 'avi';
}

async function transcodeVideo(videoUrl, tabId) {
  function reportProgress(status, progress = 0) {
    if (port) {
      port.postMessage({ type: 'PROGRESS', tabId, status, progress });
    }
  }

  const ff = await loadFFmpeg();
  const ext = getExtension(videoUrl);
  const inputName = `input.${ext}`;
  const outputName = 'output.mp4';

  reportProgress('Downloading video...', 0);

  const response = await fetch(videoUrl, { credentials: 'include' });
  if (!response.ok) {
    throw new Error(`Fetch failed: ${response.status}`);
  }

  const contentLength = response.headers.get('content-length');
  const total = parseInt(contentLength, 10) || 0;
  let loaded = 0;
  const reader = response.body.getReader();
  const chunks = [];

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    loaded += value.length;
    if (total) {
      reportProgress('Downloading...', Math.round(loaded / total * 100));
    }
  }

  const videoData = new Uint8Array(loaded);
  let position = 0;
  for (const chunk of chunks) {
    videoData.set(chunk, position);
    position += chunk.length;
  }

  const progressHandler = ({ progress }) => {
    reportProgress('Transcoding...', Math.round(progress * 100));
  };
  ff.on('progress', progressHandler);

  reportProgress('Transcoding...', 0);
  await ff.writeFile(inputName, videoData);

  await ff.exec([
    '-i', inputName,
    '-c:v', 'libx264',
    '-preset', 'ultrafast',
    '-crf', '28',
    '-c:a', 'aac',
    '-b:a', '128k',
    outputName
  ]);

  ff.off('progress', progressHandler);

  const data = await ff.readFile(outputName);
  await ff.deleteFile(inputName);
  await ff.deleteFile(outputName);

  // Convert to base64 - chunk to avoid stack overflow
  const chunkSize = 32768;
  let base64 = '';
  for (let i = 0; i < data.length; i += chunkSize) {
    const chunk = data.subarray(i, Math.min(i + chunkSize, data.length));
    base64 += String.fromCharCode.apply(null, chunk);
  }
  base64 = btoa(base64);

  return { base64, mimeType: 'video/mp4' };
}

// Convert Uint8Array to base64 in chunks
function uint8ToBase64(data) {
  const chunkSize = 32768;
  let base64 = '';
  for (let i = 0; i < data.length; i += chunkSize) {
    const chunk = data.subarray(i, Math.min(i + chunkSize, data.length));
    base64 += String.fromCharCode.apply(null, chunk);
  }
  return btoa(base64);
}

// Extract video from ZIP file
async function extractZipVideo(zipUrl, tabId) {

  function reportProgress(status, progress = 0) {
    if (port) {
      port.postMessage({ type: 'PROGRESS', tabId, status, progress });
    }
  }

  reportProgress('Downloading ZIP...', 0);

  // Download the zip file with credentials to include cookies
  try {
    const response = await fetch(zipUrl, { credentials: 'include', mode: 'cors' });
    if (!response.ok) {
      throw new Error(`Fetch failed: ${response.status}`);
    }
  } catch (fetchError) {
    throw fetchError;
  }

  // Re-fetch since we consumed the response above for logging
  const response = await fetch(zipUrl, { credentials: 'include', mode: 'cors' });
  if (!response.ok) {
    throw new Error(`Fetch failed: ${response.status}`);
  }

  const contentLength = response.headers.get('content-length');
  const total = parseInt(contentLength, 10) || 0;
  let loaded = 0;
  const reader = response.body.getReader();
  const chunks = [];

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    loaded += value.length;
    if (total) {
      reportProgress('Downloading ZIP...', Math.round(loaded / total * 100));
    }
  }

  const zipData = new Uint8Array(loaded);
  let position = 0;
  for (const chunk of chunks) {
    zipData.set(chunk, position);
    position += chunk.length;
  }

  reportProgress('Extracting ZIP...', 0);

  // Extract the zip
  const zip = await JSZip.loadAsync(zipData);

  // Find video files in the zip
  const allFormats = [...NATIVE_FORMATS, ...TRANSCODE_FORMATS];
  let videoFile = null;
  let videoFileName = null;

  for (const fileName of Object.keys(zip.files)) {
    const lowerName = fileName.toLowerCase();
    // Skip directories and hidden files
    if (zip.files[fileName].dir || lowerName.startsWith('__macosx') || lowerName.startsWith('.')) {
      continue;
    }
    for (const ext of allFormats) {
      if (lowerName.endsWith('.' + ext)) {
        videoFile = zip.files[fileName];
        videoFileName = fileName;
        break;
      }
    }
    if (videoFile) break;
  }

  if (!videoFile) {
    throw new Error('No video file found in ZIP');
  }

  reportProgress('Extracting video...', 50);

  const videoData = await videoFile.async('uint8array');
  const ext = videoFileName.split('.').pop().toLowerCase();

  reportProgress('Processing...', 75);

  // Check if native format or needs transcoding
  if (NATIVE_FORMATS.includes(ext)) {
    // Can play directly
    const mimeTypes = {
      'mp4': 'video/mp4',
      'webm': 'video/webm',
      'ogg': 'video/ogg',
      'm4v': 'video/mp4'
    };
    return {
      base64: uint8ToBase64(videoData),
      mimeType: mimeTypes[ext] || 'video/mp4',
      needsTranscode: false
    };
  } else {
    // Needs transcoding - do it here
    reportProgress('Transcoding video...', 0);

    const ff = await loadFFmpeg();
    const inputName = `input.${ext}`;
    const outputName = 'output.mp4';

    const progressHandler = ({ progress }) => {
      reportProgress('Transcoding...', Math.round(progress * 100));
    };
    ff.on('progress', progressHandler);

    await ff.writeFile(inputName, videoData);

    await ff.exec([
      '-i', inputName,
      '-c:v', 'libx264',
      '-preset', 'ultrafast',
      '-crf', '28',
      '-c:a', 'aac',
      '-b:a', '128k',
      outputName
    ]);

    ff.off('progress', progressHandler);

    const data = await ff.readFile(outputName);
    await ff.deleteFile(inputName);
    await ff.deleteFile(outputName);

    return {
      base64: uint8ToBase64(data),
      mimeType: 'video/mp4',
      needsTranscode: false  // Already transcoded
    };
  }
}

// Extract video from ZIP data (base64 encoded, already downloaded by content script)
async function extractZipFromData(zipBase64, tabId) {

  function reportProgress(status, progress = 0) {
    if (port) {
      port.postMessage({ type: 'PROGRESS', tabId, status, progress });
    }
  }

  reportProgress('Processing ZIP data...', 0);

  // Convert base64 to Uint8Array
  const binaryString = atob(zipBase64);
  const zipData = new Uint8Array(binaryString.length);
  for (let i = 0; i < binaryString.length; i++) {
    zipData[i] = binaryString.charCodeAt(i);
  }


  reportProgress('Extracting ZIP...', 0);

  // Extract the zip
  const zip = await JSZip.loadAsync(zipData);

  // Find ALL video files in the zip
  const allFormats = [...NATIVE_FORMATS, ...TRANSCODE_FORMATS];
  const videoFiles = [];

  for (const fileName of Object.keys(zip.files)) {
    const lowerName = fileName.toLowerCase();
    // Skip directories and hidden files
    if (zip.files[fileName].dir || lowerName.startsWith('__macosx') || lowerName.startsWith('.')) {
      continue;
    }
    for (const ext of allFormats) {
      if (lowerName.endsWith('.' + ext)) {
        videoFiles.push({ file: zip.files[fileName], name: fileName });
        break;
      }
    }
  }

  if (videoFiles.length === 0) {
    throw new Error('No video file found in ZIP');
  }

  // Sort by filename for consistent ordering
  videoFiles.sort((a, b) => a.name.localeCompare(b.name));


  const mimeTypes = {
    'mp4': 'video/mp4',
    'webm': 'video/webm',
    'ogg': 'video/ogg',
    'm4v': 'video/mp4'
  };

  const results = [];

  for (let i = 0; i < videoFiles.length; i++) {
    const { file: videoFile, name: videoFileName } = videoFiles[i];
    reportProgress(`Extracting video ${i + 1}/${videoFiles.length}...`, Math.round((i / videoFiles.length) * 100));

    const videoData = await videoFile.async('uint8array');
    const ext = videoFileName.split('.').pop().toLowerCase();

    // Check if native format or needs transcoding
    if (NATIVE_FORMATS.includes(ext)) {
      // Can play directly
      results.push({
        name: videoFileName,
        base64: uint8ToBase64(videoData),
        mimeType: mimeTypes[ext] || 'video/mp4'
      });
    } else {
      // Needs transcoding
      reportProgress(`Transcoding video ${i + 1}/${videoFiles.length}...`, 0);

      const ff = await loadFFmpeg();
      const inputName = `input_${i}.${ext}`;
      const outputName = `output_${i}.mp4`;

      const progressHandler = ({ progress }) => {
        const baseProgress = (i / videoFiles.length) * 100;
        const videoProgress = (progress * 100) / videoFiles.length;
        reportProgress(`Transcoding video ${i + 1}/${videoFiles.length}...`, Math.round(baseProgress + videoProgress));
      };
      ff.on('progress', progressHandler);

      await ff.writeFile(inputName, videoData);

      await ff.exec([
        '-i', inputName,
        '-c:v', 'libx264',
        '-preset', 'ultrafast',
        '-crf', '28',
        '-c:a', 'aac',
        '-b:a', '128k',
        outputName
      ]);

      ff.off('progress', progressHandler);

      const data = await ff.readFile(outputName);
      await ff.deleteFile(inputName);
      await ff.deleteFile(outputName);

      results.push({
        name: videoFileName,
        base64: uint8ToBase64(data),
        mimeType: 'video/mp4'
      });
    }
  }

  reportProgress('Done!', 100);

  // Return single video for backward compatibility, or array for multiple
  if (results.length === 1) {
    return {
      base64: results[0].base64,
      mimeType: results[0].mimeType,
      needsTranscode: false
    };
  } else {
    return {
      videos: results,
      multiple: true
    };
  }
}

// Connect to background via port
port = chrome.runtime.connect({ name: 'offscreen' });

// Storage for chunked input from background
let chunkedInput = null;

port.onMessage.addListener((message) => {
  if (message.type === 'TRANSCODE') {
    transcodeVideo(message.videoUrl, message.tabId)
      .then(result => {
        sendResult(result);
      })
      .catch(error => {
        port.postMessage({ type: 'RESULT', error: error.message || String(error) });
      });
  }

  if (message.type === 'EXTRACT_ZIP') {
    extractZipVideo(message.zipUrl, message.tabId)
      .then(result => {
        sendResult(result);
      })
      .catch(error => {
        port.postMessage({ type: 'RESULT', error: error.message || String(error) });
      });
  }

  if (message.type === 'EXTRACT_ZIP_DATA') {
    extractZipFromData(message.zipBase64, message.tabId)
      .then(result => {
        sendResult(result);
      })
      .catch(error => {
        port.postMessage({ type: 'RESULT', error: error.message || String(error) });
      });
  }

  // Handle chunked input from background
  if (message.type === 'EXTRACT_ZIP_DATA_START') {
    chunkedInput = {
      totalChunks: message.totalChunks,
      chunks: [],
      received: 0,
      tabId: message.tabId
    };
  }

  if (message.type === 'EXTRACT_ZIP_DATA_CHUNK' && chunkedInput) {
    chunkedInput.chunks[message.chunkIndex] = message.chunk;
    chunkedInput.received++;
  }

  if (message.type === 'EXTRACT_ZIP_DATA_END' && chunkedInput) {
    const zipBase64 = chunkedInput.chunks.join('');
    const tabId = chunkedInput.tabId;
    chunkedInput = null;

    extractZipFromData(zipBase64, tabId)
      .then(result => {
        sendResult(result);
      })
      .catch(error => {
        port.postMessage({ type: 'RESULT', error: error.message || String(error) });
      });
  }
});

port.onDisconnect.addListener(() => {
  port = null;
});

port.postMessage({ type: 'READY' });

