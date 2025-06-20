// Combined background script - supports both SPA and navigation modes

console.log('Combined background script loaded');

// Import both modes
importScripts('simple_background.js', 'spa_downloader.js');