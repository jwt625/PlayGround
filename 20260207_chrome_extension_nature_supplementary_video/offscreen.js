// Offscreen document - runs FFmpeg transcoding and ZIP extraction in extension context

let ffmpeg = null;
let ffmpegLoaded = false;
let port = null;

// Video extensions
const NATIVE_FORMATS = ['mp4', 'webm', 'ogg', 'm4v'];
const TRANSCODE_FORMATS = ['avi', 'mkv', 'flv', 'wmv', 'mov'];

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
  console.log('[Offscreen] extractZipVideo called with URL:', zipUrl);

  function reportProgress(status, progress = 0) {
    if (port) {
      port.postMessage({ type: 'PROGRESS', tabId, status, progress });
    }
  }

  reportProgress('Downloading ZIP...', 0);

  // Download the zip file with credentials to include cookies
  console.log('[Offscreen] Fetching ZIP with credentials...');
  try {
    const response = await fetch(zipUrl, { credentials: 'include', mode: 'cors' });
    console.log('[Offscreen] Fetch response status:', response.status, response.statusText);
    console.log('[Offscreen] Response headers:', [...response.headers.entries()]);
    if (!response.ok) {
      throw new Error(`Fetch failed: ${response.status}`);
    }
  } catch (fetchError) {
    console.error('[Offscreen] Fetch error:', fetchError);
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
  console.log('[Offscreen] extractZipFromData called, data length:', zipBase64.length);

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

  console.log('[Offscreen] ZIP data converted, size:', zipData.length);

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

  console.log('[Offscreen] Found video file:', videoFileName);
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

// Connect to background via port
port = chrome.runtime.connect({ name: 'offscreen' });

port.onMessage.addListener((message) => {
  if (message.type === 'TRANSCODE') {
    transcodeVideo(message.videoUrl, message.tabId)
      .then(result => {
        port.postMessage({ type: 'RESULT', result });
      })
      .catch(error => {
        port.postMessage({ type: 'RESULT', error: error.message || String(error) });
      });
  }

  if (message.type === 'EXTRACT_ZIP') {
    extractZipVideo(message.zipUrl, message.tabId)
      .then(result => {
        port.postMessage({ type: 'RESULT', result });
      })
      .catch(error => {
        port.postMessage({ type: 'RESULT', error: error.message || String(error) });
      });
  }

  if (message.type === 'EXTRACT_ZIP_DATA') {
    extractZipFromData(message.zipBase64, message.tabId)
      .then(result => {
        port.postMessage({ type: 'RESULT', result });
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

