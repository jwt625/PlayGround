// Offscreen document - runs FFmpeg transcoding in extension context (no page CSP)

let ffmpeg = null;
let ffmpegLoaded = false;

async function loadFFmpeg() {
  if (ffmpegLoaded) return ffmpeg;

  console.log('[Offscreen] Loading FFmpeg...');
  console.log('[Offscreen] FFmpegWASM available:', typeof FFmpegWASM);

  try {
    if (typeof FFmpegWASM === 'undefined') {
      throw new Error('FFmpegWASM is not defined - script may not have loaded');
    }

    const { FFmpeg } = FFmpegWASM;
    console.log('[Offscreen] FFmpeg constructor:', typeof FFmpeg);

    ffmpeg = new FFmpeg();

    ffmpeg.on('log', ({ message }) => {
      console.log('[FFmpeg]', message);
    });

    const coreURL = chrome.runtime.getURL('ffmpeg/ffmpeg-core.js');
    const wasmURL = chrome.runtime.getURL('ffmpeg/ffmpeg-core.wasm');
    console.log('[Offscreen] Core URL:', coreURL);
    console.log('[Offscreen] WASM URL:', wasmURL);

    await ffmpeg.load({
      coreURL: coreURL,
      wasmURL: wasmURL,
    });

    ffmpegLoaded = true;
    console.log('[Offscreen] FFmpeg loaded successfully');
    return ffmpeg;
  } catch (error) {
    console.error('[Offscreen] Failed to load FFmpeg:', error);
    throw error;
  }
}

function getExtension(url) {
  const match = url.toLowerCase().match(/\.(avi|mkv|flv|wmv|mov|mp4|webm)/);
  return match ? match[1] : 'avi';
}

// Port for communication with background
let port = null;

async function transcodeVideo(videoUrl, tabId) {
  const ff = await loadFFmpeg();
  const ext = getExtension(videoUrl);
  const inputName = `input.${ext}`;
  const outputName = 'output.mp4';

  // Report progress via port
  function reportProgress(status, progress = 0) {
    if (port) {
      port.postMessage({
        type: 'PROGRESS',
        tabId: tabId,
        status: status,
        progress: progress
      });
    }
  }
  
  reportProgress('Downloading video...', 0);
  
  // Fetch the video
  const response = await fetch(videoUrl);
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
  
  console.log('[Offscreen] Downloaded', loaded, 'bytes');
  
  // Set up progress handler
  const progressHandler = ({ progress }) => {
    reportProgress('Transcoding...', Math.round(progress * 100));
  };
  ff.on('progress', progressHandler);
  
  reportProgress('Transcoding...', 0);
  
  // Write input file
  await ff.writeFile(inputName, videoData);
  
  // Transcode to MP4
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
  
  // Read output
  const data = await ff.readFile(outputName);
  
  // Cleanup
  await ff.deleteFile(inputName);
  await ff.deleteFile(outputName);
  
  console.log('[Offscreen] Transcoding complete, output size:', data.length);
  
  // Convert to base64 for transfer
  const base64 = btoa(String.fromCharCode(...data));
  
  return { base64: base64, mimeType: 'video/mp4' };
}

// Connect to background via port
try {
  console.log('[Offscreen] Connecting to background via port...');
  port = chrome.runtime.connect({ name: 'offscreen' });

  port.onMessage.addListener((message) => {
    console.log('[Offscreen] Received port message:', message.type);

    if (message.type === 'TRANSCODE') {
      console.log('[Offscreen] Starting transcode for:', message.videoUrl);
      transcodeVideo(message.videoUrl, message.tabId)
        .then(result => {
          console.log('[Offscreen] Sending result back via port');
          port.postMessage({
            type: 'RESULT',
            result: result
          });
        })
        .catch(error => {
          console.error('[Offscreen] Transcode error:', error);
          port.postMessage({
            type: 'RESULT',
            error: error.message
          });
        });
    }
  });

  port.onDisconnect.addListener(() => {
    console.log('[Offscreen] Port disconnected');
    port = null;
  });

  // Signal that we're ready to receive messages
  port.postMessage({ type: 'READY' });
  console.log('[Offscreen] Document ready, sent READY signal');
} catch (err) {
  console.error('[Offscreen] Setup error:', err);
}

