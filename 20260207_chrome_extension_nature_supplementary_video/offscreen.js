// Offscreen document - runs FFmpeg transcoding in extension context (no page CSP)
console.log('[Offscreen] Script starting...');

let ffmpeg = null;
let ffmpegLoaded = false;

// Debug helper - sends to background so we can see it
function debug(msg, data) {
  console.log('[Offscreen]', msg, data !== undefined ? data : '');
  if (port) {
    port.postMessage({ type: 'DEBUG', msg: msg + (data !== undefined ? ' ' + JSON.stringify(data) : '') });
  }
}

console.log('[Offscreen] FFmpegWASM check:', typeof FFmpegWASM);
if (typeof FFmpegWASM !== 'undefined') {
  console.log('[Offscreen] FFmpegWASM keys:', Object.keys(FFmpegWASM));
}

async function loadFFmpeg() {
  debug('loadFFmpeg called, ffmpegLoaded=' + ffmpegLoaded);
  if (ffmpegLoaded) {
    debug('FFmpeg already loaded, returning cached');
    return ffmpeg;
  }

  debug('Loading FFmpeg...');
  debug('FFmpegWASM type: ' + typeof FFmpegWASM);

  try {
    if (typeof FFmpegWASM === 'undefined') {
      throw new Error('FFmpegWASM is not defined - script may not have loaded');
    }

    debug('FFmpegWASM keys: ' + Object.keys(FFmpegWASM).join(', '));
    const { FFmpeg } = FFmpegWASM;
    debug('FFmpeg constructor type: ' + typeof FFmpeg);

    ffmpeg = new FFmpeg();
    debug('FFmpeg instance created');

    ffmpeg.on('log', ({ message }) => {
      debug('FFmpeg log: ' + message);
    });

    const coreURL = chrome.runtime.getURL('ffmpeg/ffmpeg-core.js');
    const wasmURL = chrome.runtime.getURL('ffmpeg/ffmpeg-core.wasm');
    debug('Core URL: ' + coreURL);
    debug('WASM URL: ' + wasmURL);

    debug('Calling ffmpeg.load()...');
    await ffmpeg.load({
      coreURL: coreURL,
      wasmURL: wasmURL,
    });

    ffmpegLoaded = true;
    debug('FFmpeg loaded successfully!');
    return ffmpeg;
  } catch (error) {
    debug('Failed to load FFmpeg: ' + error.message);
    debug('Stack: ' + error.stack);
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
  debug('transcodeVideo called with: ' + videoUrl);

  // Report progress via port
  function reportProgress(status, progress = 0) {
    debug('Progress: ' + status + ' ' + progress + '%');
    if (port) {
      port.postMessage({
        type: 'PROGRESS',
        tabId: tabId,
        status: status,
        progress: progress
      });
    }
  }

  try {
    debug('About to call loadFFmpeg...');
    const ff = await loadFFmpeg();
    debug('loadFFmpeg returned successfully');

    const ext = getExtension(videoUrl);
    const inputName = `input.${ext}`;
    const outputName = 'output.mp4';
    debug('Input: ' + inputName + ', Output: ' + outputName);

    reportProgress('Downloading video...', 0);

    debug('Fetching video from: ' + videoUrl);
    const response = await fetch(videoUrl);
    debug('Fetch response status: ' + response.status);

    if (!response.ok) {
      throw new Error(`Fetch failed: ${response.status} ${response.statusText}`);
    }

    const contentLength = response.headers.get('content-length');
    const total = parseInt(contentLength, 10) || 0;
    debug('Content-Length: ' + total);

    let loaded = 0;
    const reader = response.body.getReader();
    const chunks = [];

    debug('Starting download stream...');
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      chunks.push(value);
      loaded += value.length;
      if (total && loaded % 500000 < value.length) { // Log every ~500KB
        debug('Downloaded ' + loaded + ' / ' + total + ' bytes');
      }
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

    debug('Download complete: ' + loaded + ' bytes');

    // Set up progress handler
    const progressHandler = ({ progress }) => {
      reportProgress('Transcoding...', Math.round(progress * 100));
    };
    ff.on('progress', progressHandler);

    reportProgress('Transcoding...', 0);

    debug('Writing input file to FFmpeg FS...');
    await ff.writeFile(inputName, videoData);
    debug('Input file written, starting transcode exec...');

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

    debug('Transcode exec complete');
    ff.off('progress', progressHandler);

    // Read output
    debug('Reading output file...');
    const data = await ff.readFile(outputName);

    // Cleanup
    debug('Cleaning up temp files...');
    await ff.deleteFile(inputName);
    await ff.deleteFile(outputName);

    debug('Transcoding complete, output size: ' + data.length);

    // Convert to base64 for transfer - chunk it to avoid stack overflow
    debug('Converting to base64...');
    const chunkSize = 32768;
    let base64 = '';
    for (let i = 0; i < data.length; i += chunkSize) {
      const chunk = data.subarray(i, Math.min(i + chunkSize, data.length));
      base64 += String.fromCharCode.apply(null, chunk);
    }
    base64 = btoa(base64);
    debug('Base64 length: ' + base64.length);

    return { base64: base64, mimeType: 'video/mp4' };
  } catch (error) {
    console.error('[Offscreen] transcodeVideo error:', error);
    throw error;
  }
}

// Connect to background via port
try {
  console.log('[Offscreen] Connecting to background via port...');
  port = chrome.runtime.connect({ name: 'offscreen' });

  port.onMessage.addListener((message) => {
    console.log('[Offscreen] Received port message:', message.type);

    // Echo back to confirm receipt
    port.postMessage({ type: 'ECHO', received: message.type });

    if (message.type === 'TRANSCODE') {
      console.log('[Offscreen] Starting transcode for:', message.videoUrl);
      port.postMessage({ type: 'DEBUG', msg: 'About to call transcodeVideo' });

      (async () => {
        try {
          port.postMessage({ type: 'DEBUG', msg: 'Inside async, calling transcodeVideo' });
          const result = await transcodeVideo(message.videoUrl, message.tabId);
          console.log('[Offscreen] Sending result back via port');
          port.postMessage({
            type: 'RESULT',
            result: result
          });
        } catch (error) {
          console.error('[Offscreen] Transcode error:', error);
          port.postMessage({
            type: 'DEBUG',
            msg: 'Error: ' + (error.message || String(error)),
            stack: error.stack
          });
          port.postMessage({
            type: 'RESULT',
            error: error.message || String(error)
          });
        }
      })();
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

