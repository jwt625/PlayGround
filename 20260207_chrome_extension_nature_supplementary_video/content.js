// Inline Supplementary Video Player
// Replaces download links with inline video players
// Uses FFmpeg.wasm via offscreen document for unsupported formats (AVI, MKV, etc.)

(function() {
  'use strict';


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

  // Check if a link is a zipped video (common on Science.org)
  function isZippedVideo(href, description) {
    if (!href) return false;
    const lowerHref = href.toLowerCase();
    const lowerDesc = (description || '').toLowerCase();
    // Check if it's a zip file with movie/video in the name or description
    return lowerHref.includes('.zip') &&
           (lowerHref.includes('movie') || lowerHref.includes('video') ||
            lowerDesc.includes('movie') || lowerDesc.includes('video'));
  }

  // Check if format needs transcoding
  function needsTranscoding(href) {
    if (!href) return false;
    const lowerHref = href.toLowerCase();
    return TRANSCODE_FORMATS.some(ext => lowerHref.includes(ext));
  }

  // Helper to convert base64 to blob URL
  function base64ToBlobUrl(base64, mimeType) {
    const binaryString = atob(base64);
    const bytes = new Uint8Array(binaryString.length);
    for (let i = 0; i < binaryString.length; i++) {
      bytes[i] = binaryString.charCodeAt(i);
    }
    const blob = new Blob([bytes], { type: mimeType });
    return URL.createObjectURL(blob);
  }

  // Extract video from ZIP via background/offscreen document
  // Returns either a single blob URL (string) or an array of { name, url } objects
  async function extractZipVideo(zipUrl, statusCallback) {
    transcodeCallbacks.set(zipUrl, statusCallback);
    statusCallback('Downloading ZIP...');

    try {
      // Download the ZIP in content script (has access to page cookies)
      const fetchResponse = await fetch(zipUrl, { credentials: 'include' });
      if (!fetchResponse.ok) {
        throw new Error(`Download failed: ${fetchResponse.status}`);
      }

      const arrayBuffer = await fetchResponse.arrayBuffer();

      // Convert to base64 and send in chunks (max ~32MB per message to stay under 64MB limit)
      statusCallback('Processing ZIP...');
      const uint8Array = new Uint8Array(arrayBuffer);
      let binary = '';
      const convChunkSize = 8192;
      for (let i = 0; i < uint8Array.length; i += convChunkSize) {
        const chunk = uint8Array.subarray(i, i + convChunkSize);
        binary += String.fromCharCode.apply(null, chunk);
      }
      const zipBase64 = btoa(binary);

      // Send in chunks if too large (32MB chunks to stay well under 64MB limit)
      const MAX_CHUNK_SIZE = 32 * 1024 * 1024; // 32MB
      const transferId = Date.now().toString() + Math.random().toString(36).substr(2, 9);

      if (zipBase64.length > MAX_CHUNK_SIZE) {
        const totalChunks = Math.ceil(zipBase64.length / MAX_CHUNK_SIZE);

        for (let i = 0; i < totalChunks; i++) {
          const start = i * MAX_CHUNK_SIZE;
          const end = Math.min(start + MAX_CHUNK_SIZE, zipBase64.length);
          const chunk = zipBase64.substring(start, end);

          statusCallback(`Transferring... ${Math.round((i + 1) / totalChunks * 100)}%`);

          const chunkResponse = await chrome.runtime.sendMessage({
            type: 'EXTRACT_ZIP_CHUNK',
            transferId: transferId,
            chunkIndex: i,
            totalChunks: totalChunks,
            chunk: chunk
          });

          if (chunkResponse.error) {
            throw new Error(chunkResponse.error);
          }
        }

        // Now request extraction
        statusCallback('Extracting...');
        const response = await chrome.runtime.sendMessage({
          type: 'EXTRACT_ZIP_PROCESS',
          transferId: transferId
        });

        transcodeCallbacks.delete(zipUrl);

        if (response.error) {
          throw new Error(response.error);
        }

        // Check if response is chunked
        if (response.chunked) {
          const chunks = [];
          for (let i = 0; i < response.totalChunks; i++) {
            statusCallback(`Receiving result... ${Math.round((i + 1) / response.totalChunks * 100)}%`);
            const chunkResp = await chrome.runtime.sendMessage({
              type: 'GET_RESULT_CHUNK',
              resultId: response.resultId,
              chunkIndex: i
            });
            if (chunkResp.error) {
              throw new Error(chunkResp.error);
            }
            chunks.push(chunkResp.chunk);
          }
          const combined = chunks.join('');

          // Check if this is multiple videos (JSON) or single video (base64)
          if (response.isMultipleVideos) {
            const parsed = JSON.parse(combined);
            return parsed.videos.map(v => ({
              name: v.name,
              url: base64ToBlobUrl(v.base64, v.mimeType)
            }));
          }

          return base64ToBlobUrl(combined, response.mimeType);
        }

        // Check if multiple videos
        if (response.multiple && response.videos) {
          return response.videos.map(v => ({
            name: v.name,
            url: base64ToBlobUrl(v.base64, v.mimeType)
          }));
        }

        // Single video - convert base64 back to blob URL
        return base64ToBlobUrl(response.base64, response.mimeType);
      } else {
        // Small enough to send in one message
        const response = await chrome.runtime.sendMessage({
          type: 'EXTRACT_ZIP_DATA',
          zipBase64: zipBase64
        });

        transcodeCallbacks.delete(zipUrl);

        if (response.error) {
          throw new Error(response.error);
        }

        // Check if response is chunked (video might be larger than ZIP)
        if (response.chunked) {
          const chunks = [];
          for (let i = 0; i < response.totalChunks; i++) {
            statusCallback(`Receiving result... ${Math.round((i + 1) / response.totalChunks * 100)}%`);
            const chunkResp = await chrome.runtime.sendMessage({
              type: 'GET_RESULT_CHUNK',
              resultId: response.resultId,
              chunkIndex: i
            });
            if (chunkResp.error) {
              throw new Error(chunkResp.error);
            }
            chunks.push(chunkResp.chunk);
          }
          const combined = chunks.join('');

          // Check if this is multiple videos (JSON) or single video (base64)
          if (response.isMultipleVideos) {
            const parsed = JSON.parse(combined);
            return parsed.videos.map(v => ({
              name: v.name,
              url: base64ToBlobUrl(v.base64, v.mimeType)
            }));
          }

          return base64ToBlobUrl(combined, response.mimeType);
        }

        // Check if multiple videos
        if (response.multiple && response.videos) {
          return response.videos.map(v => ({
            name: v.name,
            url: base64ToBlobUrl(v.base64, v.mimeType)
          }));
        }

        // Single video - convert base64 back to blob URL
        return base64ToBlobUrl(response.base64, response.mimeType);
      }
    } catch (error) {
      transcodeCallbacks.delete(zipUrl);
      throw error;
    }
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
      transcodeBtn.textContent = 'Click to load video (requires conversion)';
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
    downloadLink.textContent = 'Download';
    downloadLink.download = '';

    // Fullscreen button
    const fullscreenBtn = document.createElement('button');
    fullscreenBtn.className = 'svp-fullscreen-btn';
    fullscreenBtn.textContent = 'Fullscreen';
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

  // Create video player for zipped videos (Science.org)
  function createZipVideoPlayer(zipUrl, description) {
    const container = document.createElement('div');
    container.className = 'svp-video-player-container';

    // Create wrapper for video and controls
    const wrapper = document.createElement('div');
    wrapper.className = 'svp-video-wrapper';

    // Status element
    const status = document.createElement('div');
    status.className = 'svp-video-status';
    status.style.display = 'none';

    // Container for tabs (for multiple videos)
    const tabsContainer = document.createElement('div');
    tabsContainer.className = 'svp-video-tabs';
    tabsContainer.style.display = 'none';

    // Container for all videos
    const videosContainer = document.createElement('div');
    videosContainer.className = 'svp-videos-container';

    // Show extract button
    const extractBtn = document.createElement('button');
    extractBtn.className = 'svp-transcode-btn';
    extractBtn.textContent = 'Click to extract and play video from ZIP';
    extractBtn.addEventListener('click', async () => {
      extractBtn.style.display = 'none';
      status.style.display = 'block';
      status.textContent = 'Downloading ZIP...';

      try {
        const result = await extractZipVideo(zipUrl, (msg) => {
          status.textContent = msg;
        });
        status.style.display = 'none';

        // Check if result is array (multiple videos)
        if (Array.isArray(result)) {

          // Create tabs and video elements for each video
          result.forEach((videoInfo, index) => {
            // Create tab button
            const tab = document.createElement('button');
            tab.className = 'svp-video-tab' + (index === 0 ? ' active' : '');
            tab.textContent = videoInfo.name.split('/').pop(); // Show only filename
            tab.dataset.index = index;
            tab.addEventListener('click', () => {
              // Switch active tab
              tabsContainer.querySelectorAll('.svp-video-tab').forEach(t => t.classList.remove('active'));
              tab.classList.add('active');
              // Switch visible video
              videosContainer.querySelectorAll('.svp-video-item').forEach((v, i) => {
                v.style.display = i === index ? 'block' : 'none';
              });
            });
            tabsContainer.appendChild(tab);

            // Create video container
            const videoItem = document.createElement('div');
            videoItem.className = 'svp-video-item';
            videoItem.style.display = index === 0 ? 'block' : 'none';

            const video = document.createElement('video');
            video.className = 'svp-video-player';
            video.controls = true;
            video.preload = 'metadata';
            video.src = videoInfo.url;

            const videoLabel = document.createElement('div');
            videoLabel.className = 'svp-video-label';
            videoLabel.textContent = videoInfo.name.split('/').pop();

            videoItem.appendChild(video);
            videoItem.appendChild(videoLabel);
            videosContainer.appendChild(videoItem);
          });

          tabsContainer.style.display = 'flex';
        } else {
          // Single video
          const video = document.createElement('video');
          video.className = 'svp-video-player';
          video.controls = true;
          video.preload = 'metadata';
          video.src = result;
          videosContainer.appendChild(video);
        }
      } catch (error) {
        status.textContent = 'Extraction failed: ' + error.message;
        status.className = 'svp-video-status error';
      }
    });

    wrapper.appendChild(extractBtn);
    wrapper.appendChild(status);
    wrapper.appendChild(tabsContainer);
    wrapper.appendChild(videosContainer);

    // Create custom controls bar
    const controlsBar = document.createElement('div');
    controlsBar.className = 'svp-video-controls';

    // Download link
    const downloadLink = document.createElement('a');
    downloadLink.href = zipUrl;
    downloadLink.className = 'svp-download-link';
    downloadLink.textContent = 'Download ZIP';
    downloadLink.download = '';

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

    // Find all supplementary items
    const suppItems = document.querySelectorAll('.c-article-supplementary__item[data-test="supp-item"]');

    suppItems.forEach((item, index) => {
      // Find the link
      const link = item.querySelector('a[data-test="supp-info-link"]');
      if (!link) {
        return;
      }

      const href = link.getAttribute('href');

      if (!isVideoLink(href)) {
        return;
      }


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

    });
  }

  // Replace video links on Science.org
  function replaceScienceVideoLinks() {

    // Find all supplementary material items
    const suppItems = document.querySelectorAll('.core-supplementary-material');

    suppItems.forEach((item, index) => {
      // Find the download link
      const linkContainer = item.querySelector('.core-link');
      const link = linkContainer ? linkContainer.querySelector('a[download]') : null;
      if (!link) {
        return;
      }

      const rawHref = link.getAttribute('href');
      // Convert relative URL to absolute
      const href = rawHref.startsWith('/') ? window.location.origin + rawHref : rawHref;
      // Get description from .core-description
      const descElement = item.querySelector('.core-description');
      const description = descElement ? descElement.textContent.trim() : '';


      // Check if it's a direct video link
      if (isVideoLink(href)) {

        // Check if already replaced
        if (item.querySelector('.svp-video-player-container')) return;

        // Create player
        const playerContainer = createVideoPlayer(href, description);

        // Insert player after description
        if (descElement) {
          descElement.parentNode.insertBefore(playerContainer, descElement.nextSibling);
        } else {
          item.insertBefore(playerContainer, linkContainer);
        }

      }
      // Check if it's a zipped video
      else if (isZippedVideo(href, description)) {

        // Check if already processed
        if (item.querySelector('.svp-video-player-container')) return;

        // Create a player container for zipped video
        const playerContainer = createZipVideoPlayer(href, description);

        // Insert player after description
        if (descElement) {
          descElement.parentNode.insertBefore(playerContainer, descElement.nextSibling);
        } else {
          item.insertBefore(playerContainer, linkContainer);
        }

      }
    });
  }

  // Initialize
  function init() {

    // Detect which site we're on and use appropriate handler
    const hostname = window.location.hostname;

    if (hostname.includes('science.org')) {
      replaceScienceVideoLinks();
    } else {
      // Nature/Springer
      replaceVideoLinks();
    }

    // Debounced observer - only run once after DOM settles
    let debounceTimer = null;
    let hasRun = false;

    const scanHandler = hostname.includes('science.org')
      ? replaceScienceVideoLinks
      : replaceVideoLinks;

    const observer = new MutationObserver(() => {
      // Only run observer callback once, shortly after page load
      if (hasRun) return;

      clearTimeout(debounceTimer);
      debounceTimer = setTimeout(() => {
        scanHandler();
        hasRun = true;
        observer.disconnect(); // Stop observing after first debounced run
      }, 500);
    });

    observer.observe(document.body, {
      childList: true,
      subtree: true
    });

  }

  // Start when DOM is ready
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }

})();

