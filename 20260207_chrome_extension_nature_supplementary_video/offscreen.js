// Offscreen document - runs FFmpeg transcoding in extension context (no page CSP)

let ffmpeg = null;
let ffmpegLoaded = false;
let port = null;

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

  const response = await fetch(videoUrl);
  if (!response.ok) {
    throw new Error(`Fetch failed: ${response.status} ${response.statusText}`);
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
});

port.onDisconnect.addListener(() => {
  port = null;
});

port.postMessage({ type: 'READY' });

