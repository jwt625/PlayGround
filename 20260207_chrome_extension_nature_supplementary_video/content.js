// Inline Supplementary Video Player
// Replaces download links with inline video players
// Uses FFmpeg.wasm via offscreen document for unsupported formats (AVI, MKV, etc.)

(function() {
  'use strict';

  console.log('[Video Player] Extension loaded on:', window.location.href);

  // Formats natively supported by HTML5 video
  const NATIVE_FORMATS = ['.mp4', '.webm', '.ogg', '.m4v'];
  // Formats that need transcoding
  const TRANSCODE_FORMATS = ['.avi', '.mkv', '.flv', '.wmv', '.mov'];

  // Track active transcode status callbacks by video URL
  const transcodeCallbacks = new Map();

  // Listen for progress updates from background
  chrome.runtime.onMessage.addListener((message) => {
    if (message.type === 'TRANSCODE_PROGRESS') {
      // Update all registered callbacks
      transcodeCallbacks.forEach((callback) => {
        callback(`${message.status} ${message.progress}%`);
      });
    }
  });

  // Check if a link is a video link
  function isVideoLink(href) {
    if (!href) return false;
    const videoExtensions = [...NATIVE_FORMATS, ...TRANSCODE_FORMATS];
    const lowerHref = href.toLowerCase();
    return videoExtensions.some(ext => lowerHref.includes(ext));
  }

  // Check if format needs transcoding
  function needsTranscoding(href) {
    if (!href) return false;
    const lowerHref = href.toLowerCase();
    return TRANSCODE_FORMATS.some(ext => lowerHref.includes(ext));
  }

  // Transcode video via background/offscreen document
  async function transcodeVideo(videoUrl, statusCallback) {
    transcodeCallbacks.set(videoUrl, statusCallback);
    statusCallback('Starting transcoding...');

    try {
      const response = await chrome.runtime.sendMessage({
        type: 'TRANSCODE_VIDEO',
        videoUrl: videoUrl
      });

      transcodeCallbacks.delete(videoUrl);

      if (response.error) {
        throw new Error(response.error);
      }

      // Convert base64 back to blob URL
      const binaryString = atob(response.base64);
      const bytes = new Uint8Array(binaryString.length);
      for (let i = 0; i < binaryString.length; i++) {
        bytes[i] = binaryString.charCodeAt(i);
      }
      const blob = new Blob([bytes], { type: response.mimeType });
      return URL.createObjectURL(blob);
    } catch (error) {
      transcodeCallbacks.delete(videoUrl);
      throw error;
    }
  }

  // Create video player element with custom controls
  function createVideoPlayer(videoUrl, description) {
    const container = document.createElement('div');
    container.className = 'svp-video-player-container';

    const requiresTranscode = needsTranscoding(videoUrl);

    // Create wrapper for video and controls
    const wrapper = document.createElement('div');
    wrapper.className = 'svp-video-wrapper';

    // Create video element
    const video = document.createElement('video');
    video.className = 'svp-video-player';
    video.controls = true;
    video.preload = 'metadata';

    // Status element for transcoding
    const status = document.createElement('div');
    status.className = 'svp-video-status';
    status.style.display = 'none';

    if (requiresTranscode) {
      // Show transcode button instead of loading video directly
      const transcodeBtn = document.createElement('button');
      transcodeBtn.className = 'svp-transcode-btn';
      transcodeBtn.textContent = '▶ Click to load video (requires conversion)';
      transcodeBtn.addEventListener('click', async () => {
        transcodeBtn.style.display = 'none';
        status.style.display = 'block';
        status.textContent = 'Initializing converter...';

        try {
          const mp4Url = await transcodeVideo(videoUrl, (msg) => {
            status.textContent = msg;
          });
          status.style.display = 'none';
          video.src = mp4Url;
          video.style.display = 'block';
        } catch (error) {
          status.textContent = 'Conversion failed: ' + error.message + '. Please download the video instead.';
          status.className = 'svp-video-status error';
          console.error('[Video Player] Transcode error:', error);
        }
      });
      wrapper.appendChild(transcodeBtn);
      video.style.display = 'none';
    } else {
      video.src = videoUrl;
    }

    // Fallback message
    video.textContent = 'Your browser does not support HTML5 video.';

    wrapper.appendChild(status);
    wrapper.appendChild(video);

    // Create custom controls bar
    const controlsBar = document.createElement('div');
    controlsBar.className = 'svp-video-controls';

    // Speed control
    const speedLabel = document.createElement('span');
    speedLabel.textContent = 'Speed: ';
    speedLabel.className = 'svp-control-label';

    const speedSelect = document.createElement('select');
    speedSelect.className = 'svp-speed-select';
    [0.5, 0.75, 1, 1.25, 1.5, 2].forEach(rate => {
      const option = document.createElement('option');
      option.value = rate;
      option.textContent = rate + 'x';
      if (rate === 1) option.selected = true;
      speedSelect.appendChild(option);
    });
    speedSelect.addEventListener('change', () => {
      video.playbackRate = parseFloat(speedSelect.value);
    });

    // Download link
    const downloadLink = document.createElement('a');
    downloadLink.href = videoUrl;
    downloadLink.className = 'svp-download-link';
    downloadLink.textContent = '⬇ Download';
    downloadLink.download = '';

    // Fullscreen button
    const fullscreenBtn = document.createElement('button');
    fullscreenBtn.className = 'svp-fullscreen-btn';
    fullscreenBtn.textContent = '⛶ Fullscreen';
    fullscreenBtn.addEventListener('click', () => {
      if (video.requestFullscreen) {
        video.requestFullscreen();
      } else if (video.webkitRequestFullscreen) {
        video.webkitRequestFullscreen();
      }
    });

    controlsBar.appendChild(speedLabel);
    controlsBar.appendChild(speedSelect);
    controlsBar.appendChild(fullscreenBtn);
    controlsBar.appendChild(downloadLink);

    container.appendChild(wrapper);
    container.appendChild(controlsBar);

    // Add description if available
    if (description) {
      const descDiv = document.createElement('div');
      descDiv.className = 'svp-video-description';
      descDiv.textContent = description;
      container.appendChild(descDiv);
    }

    return container;
  }

  // Replace video links with players
  function replaceVideoLinks() {
    console.log('[Video Player] Scanning for video links...');

    // Find all supplementary items
    const suppItems = document.querySelectorAll('.c-article-supplementary__item[data-test="supp-item"]');
    console.log('[Video Player] Found supplementary items:', suppItems.length);

    suppItems.forEach((item, index) => {
      // Find the link
      const link = item.querySelector('a[data-test="supp-info-link"]');
      if (!link) {
        console.log('[Video Player] Item', index, '- no link found');
        return;
      }

      const href = link.getAttribute('href');
      console.log('[Video Player] Item', index, '- href:', href);

      if (!isVideoLink(href)) {
        console.log('[Video Player] Item', index, '- not a video link');
        return;
      }

      console.log('[Video Player] Item', index, '- IS a video, replacing...');

      // Get title and description
      const title = link.textContent.trim();
      const descElement = item.querySelector('.c-article-supplementary__description p');
      const description = descElement ? descElement.textContent.trim() : '';

      // Check if already replaced
      if (item.querySelector('.svp-video-player-container')) return;

      // Create player
      const playerContainer = createVideoPlayer(href, description);

      // Replace the title link with just text
      const titleElement = item.querySelector('.c-article-supplementary__title');
      if (titleElement) {
        titleElement.innerHTML = '';
        const titleText = document.createElement('span');
        titleText.className = 'svp-video-title';
        titleText.textContent = title;
        titleElement.appendChild(titleText);
      }

      // Insert player after title
      if (titleElement) {
        titleElement.parentNode.insertBefore(playerContainer, titleElement.nextSibling);
      }

      console.log('[Video Player] Item', index, '- replaced successfully');
    });
  }

  // Initialize
  function init() {
    console.log('[Video Player] Initializing...');

    // Replace videos on initial load
    replaceVideoLinks();

    // Debounced observer - only run once after DOM settles
    let debounceTimer = null;
    let hasRun = false;

    const observer = new MutationObserver(() => {
      // Only run observer callback once, shortly after page load
      if (hasRun) return;

      clearTimeout(debounceTimer);
      debounceTimer = setTimeout(() => {
        replaceVideoLinks();
        hasRun = true;
        observer.disconnect(); // Stop observing after first debounced run
        console.log('[Video Player] Observer disconnected after initial scan');
      }, 500);
    });

    observer.observe(document.body, {
      childList: true,
      subtree: true
    });

    console.log('[Video Player] Initialization complete');
  }

  // Start when DOM is ready
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }

})();

