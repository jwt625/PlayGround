# RFD-000: YouTube Liked Videos Backup Chrome Extension

## Status
- **Status**: Draft
- **Author**: Wentao Jiang
- **Created**: 2025-08-16
- **Updated**: 2025-08-16

## Summary

Design and implementation of a Chrome extension to backup metadata from YouTube liked videos with automatic removal capability to circumvent the 5000 video limit that causes loss of older liked videos.

## Problem Statement

YouTube's liked videos playlist has a hard limit of 5000 videos. Once this limit is reached, older liked videos are automatically removed from the list, causing permanent loss of video references and metadata. Users need a reliable way to:

1. Backup comprehensive metadata of all liked videos
2. Verify the backup integrity before removal
3. Safely remove videos from the liked list to allow older videos to reappear
4. Export and manage the backed-up data

## Goals

### Primary Goals
- Preserve complete metadata of YouTube liked videos before they're lost
- Implement safe removal mechanism with verification
- Provide reliable data export capabilities
- Handle YouTube's dynamic UI and anti-bot measures

### Secondary Goals
- Support incremental backups for efficiency
- Provide data analysis and search capabilities
- Enable restoration of accidentally removed videos
- Maintain compatibility with YouTube UI changes

## Architecture Overview

### Extension Structure
```
youtube-liked-backup/
├── manifest.json                 # Extension configuration & permissions
├── background/
│   ├── background.js            # Service worker for data management
│   ├── storage-manager.js       # Local storage operations
│   └── export-manager.js        # Data export functionality
├── content/
│   ├── content.js              # Main scraping logic
│   ├── video-scraper.js        # Video metadata extraction
│   ├── pagination-handler.js   # Handle infinite scroll/pagination
│   └── removal-handler.js      # Unlike video functionality
├── popup/
│   ├── popup.html              # Extension popup interface
│   ├── popup.js                # Popup logic and controls
│   └── popup.css               # Styling
├── options/
│   ├── options.html            # Settings page
│   ├── options.js              # Configuration management
│   └── options.css             # Settings styling
└── utils/
    ├── youtube-api.js          # YouTube DOM interaction utilities
    ├── data-validator.js       # Metadata validation
    └── constants.js            # Configuration constants
```

## Extension Permissions & Manifest

### Required Permissions
```json
{
  "manifest_version": 3,
  "name": "YouTube Liked Videos Backup",
  "version": "1.0.0",
  "permissions": [
    "storage",              // Chrome storage API for data persistence
    "downloads",            // Export data to user's file system
    "tabs",                 // Background tab management for detailed scraping
    "activeTab",            // Access to current YouTube tab
    "scripting"             // Inject content scripts
  ],
  "host_permissions": [
    "https://www.youtube.com/*",     // YouTube domain access
    "https://i.ytimg.com/*"          // Thumbnail verification
  ],
  "optional_permissions": [
    "unlimitedStorage"      // For users with large datasets (>5MB)
  ]
}
```

### Storage Architecture

#### Primary Storage: Chrome Storage API
```javascript
// Storage structure in chrome.storage.local
const storageSchema = {
  // Video records indexed by videoId
  "videos": {
    [videoId]: VideoRecord,
    // ... up to storage quota
  },

  // Backup sessions and metadata
  "sessions": {
    [sessionId]: BackupSession,
    // ...
  },

  // User settings and configuration
  "settings": BackupSettings,

  // Extension state and progress tracking
  "state": {
    lastBackupDate: Date,
    totalVideosBackedUp: number,
    currentSession: string,
    // ...
  }
}
```

#### Secondary Storage: IndexedDB (For Large Datasets)
```javascript
// IndexedDB schema for users exceeding Chrome storage limits
const indexedDBSchema = {
  database: "YouTubeLikedBackup",
  version: 1,
  stores: {
    videos: {
      keyPath: "videoId",
      indexes: ["channelId", "likedDate", "backupStatus"]
    },
    sessions: {
      keyPath: "sessionId",
      indexes: ["startTime"]
    },
    exports: {
      keyPath: "exportId",
      indexes: ["createdDate", "format"]
    }
  }
}
```

## Core Features

### 1. Backup Operations

#### User Actions
- **Manual Trigger**: Click "Start Backup" in popup
- **Automatic Mode**: Enable continuous monitoring when on liked videos page
- **Incremental Backup**: "Update Recent" - only scan new videos since last backup
- **Full Rescan**: "Complete Backup" - rescan all videos for updated metadata

#### Implementation Logic
- Detect YouTube liked videos page (`/playlist?list=LL`)
- Handle both grid and list view layouts
- Manage infinite scroll pagination
- Extract metadata from DOM elements and network requests
- Validate data completeness before storage

### 2. Verification & Removal System

#### User Actions
- **Verify & Remove Mode**: Toggle to enable automatic removal after successful backup
- **Manual Review**: Review videos before removal with confirmation dialog
- **Batch Operations**: Select multiple videos for verification/removal
- **Rollback Protection**: Undo recent removals (restore from backup)

#### Implementation Logic
Multi-stage verification process:
1. DOM metadata extraction
2. Network request interception for additional data
3. Thumbnail download verification
4. Cross-reference with YouTube's internal video data

Safety measures:
- Only remove after successful verification and storage
- Implement removal rate limiting to avoid triggering YouTube's spam detection
- Queue-based removal system with retry logic

### 3. Data Management

#### User Actions
- **Export Data**: JSON, CSV, or custom formats
- **Import Previous Backup**: Merge with existing data
- **Data Cleanup**: Remove duplicates, fix corrupted entries
- **Search & Filter**: Find specific videos in backup

## Data Schema

### Primary Video Record
```typescript
interface VideoRecord {
  // Core Identifiers
  videoId: string;                    // YouTube video ID (11 chars)
  url: string;                        // Full YouTube URL
  
  // Metadata
  title: string;                      // Video title
  description?: string;               // Video description (if available)
  duration: number;                   // Duration in seconds
  durationFormatted: string;          // Human readable (e.g., "10:23")
  
  // Channel Information
  channelId: string;                  // Channel ID
  channelName: string;                // Channel display name
  channelUrl: string;                 // Channel URL
  channelHandle?: string;             // @handle if available
  
  // Timestamps
  uploadDate?: Date;                  // Original upload date
  likedDate: Date;                    // When user liked it
  scrapedDate: Date;                  // When we scraped it
  lastVerified: Date;                 // Last verification timestamp
  
  // Media Assets
  thumbnails: ThumbnailSet;           // Multiple resolution thumbnails
  
  // Engagement Data
  viewCount?: number;                 // View count at time of scraping
  likeCount?: number;                 // Like count (if public)
  
  // Backup Metadata
  backupStatus: BackupStatus;         // Verification and backup state
  removalStatus: RemovalStatus;       // Removal tracking
  dataSource: DataSource[];           // How data was obtained
  
  // Quality Assurance
  verificationScore: number;          // 0-100 confidence score
  missingFields: string[];            // List of fields that couldn't be scraped
  
  // Additional Context
  playlistPosition?: number;          // Position in liked videos at time of scraping
  tags?: string[];                    // Video tags if available
  category?: string;                  // Video category
  language?: string;                  // Video language
}
```

### Supporting Schemas
```typescript
interface ThumbnailSet {
  default: string;      // 120x90
  medium: string;       // 320x180
  high: string;         // 480x360
  standard?: string;    // 640x480
  maxres?: string;      // 1280x720
}

enum BackupStatus {
  PENDING = 'pending',
  SCRAPED = 'scraped',
  VERIFIED = 'verified',
  EXPORTED = 'exported',
  FAILED = 'failed'
}

enum RemovalStatus {
  KEPT = 'kept',
  QUEUED_FOR_REMOVAL = 'queued',
  REMOVED = 'removed',
  REMOVAL_FAILED = 'failed',
  RESTORED = 'restored'
}

enum DataSource {
  DOM_SCRAPING = 'dom',
  NETWORK_INTERCEPT = 'network',
  YOUTUBE_API = 'api',
  THUMBNAIL_DOWNLOAD = 'thumbnail'
}

interface BackupSession {
  sessionId: string;
  startTime: Date;
  endTime?: Date;
  videosProcessed: number;
  videosRemoved: number;
  errors: ErrorLog[];
  settings: BackupSettings;
}

interface BackupSettings {
  autoRemoveAfterBackup: boolean;
  verificationThreshold: number;      // Minimum score to consider verified
  removalRateLimit: number;           // Max removals per minute
  exportFormat: 'json' | 'csv' | 'both';
  includeDescriptions: boolean;
  downloadThumbnails: boolean;
  maxRetries: number;
}
```

## Scraping Logic & Strategy

### 1. Multi-Source Data Collection

#### Playlist-Level Data (Primary Source)
**DOM Scraping from Playlist View:**
- Video ID, title, channel name, duration, thumbnails
- View count (when visible), relative upload date
- Playlist position and basic engagement metrics

**Network Request Interception:**
- YouTube's internal API responses contain richer metadata
- Exact timestamps, detailed channel info, video statistics
- Often includes description snippets and category data

#### Video-Level Data (Secondary Source)
**Background Tab Navigation:**
- For missing critical metadata (full descriptions, exact dates)
- Opens video pages in background tabs for detailed scraping
- Automatic tab management to avoid overwhelming browser

**Navigation Strategy:**
```javascript
// Selective navigation for high-value missing data
if (video.missingCriticalFields.length > 0) {
  await scrapeVideoPage(video.videoId);
}
```

#### Data Collection Priority
1. **Network Interception** (fastest, most complete)
2. **DOM Scraping** (reliable fallback)
3. **Background Navigation** (for missing critical data only)
4. **Thumbnail Verification** (accessibility check)

### 2. Pagination Handling
- **Infinite Scroll Detection**: Monitor for new video cards loading
- **Batch Processing**: Process videos in chunks to avoid overwhelming the page
- **Progress Tracking**: Maintain position in case of interruption
- **Rate Limiting**: Respect YouTube's loading patterns

### 3. Verification Pipeline
```
Video Detected → DOM Scrape → Network Data → Thumbnail Check → Validation → Storage → Queue for Removal
```

#### Verification Criteria
- All required fields present (videoId, title, channelName, duration)
- Thumbnail URLs accessible
- Video still exists (not deleted/private)
- Data consistency across sources
- Minimum confidence score threshold

### 4. Removal Strategy
- **Safety First**: Never remove without successful backup
- **Rate Limiting**: Max 1 removal per 2-3 seconds to avoid detection
- **Error Handling**: Retry failed removals, log issues
- **Rollback Capability**: Track removals for potential restoration

## Export & Data Management

### 1. Export Formats

#### JSON Export
- Complete metadata with full schema
- Nested structure for easy programmatic access
- Includes backup session metadata

#### CSV Export
- Flattened structure for spreadsheet analysis
- Configurable column selection
- Multiple CSV files for different data aspects

#### Custom Formats
- YouTube playlist format for re-importing
- Markdown format for documentation
- RSS/OPML for feed readers

### 2. Data Integrity Features
- **Checksums**: Verify export file integrity
- **Incremental Exports**: Only export new/changed data
- **Compression**: Automatic compression for large datasets
- **Encryption**: Optional encryption for sensitive data

### 3. Import/Merge Capabilities
- **Duplicate Detection**: Smart merging of overlapping datasets
- **Data Migration**: Import from other backup tools
- **Conflict Resolution**: Handle conflicting metadata

## User Interface Design

### 1. Popup Interface
- **Status Dashboard**: Current backup progress, total videos backed up
- **Quick Actions**: Start backup, export data, view recent activity
- **Settings Access**: Link to options page
- **Emergency Stop**: Cancel ongoing operations

### 2. Options Page
- **Backup Settings**: Configure automatic removal, verification thresholds
- **Export Preferences**: Choose formats, compression options
- **Data Management**: View statistics, cleanup tools
- **Advanced Options**: Rate limiting, retry settings

### 3. Progress Indicators
- **Real-time Progress**: Videos processed, estimated time remaining
- **Error Reporting**: Failed videos with retry options
- **Success Metrics**: Verification scores, removal success rate

### 5. Extension Navigation Capabilities

#### Background Tab Management
```javascript
// Open video in background tab for detailed scraping
const backgroundTab = await chrome.tabs.create({
  url: `https://www.youtube.com/watch?v=${videoId}`,
  active: false,
  pinned: false
});

// Inject content script and wait for data
const videoData = await scrapeVideoPage(backgroundTab.id);

// Close background tab
await chrome.tabs.remove(backgroundTab.id);
```

#### Navigation Rate Limiting
- Maximum 1 background tab per 5 seconds
- Queue system for batch video processing
- Automatic retry for failed navigations
- Tab cleanup to prevent memory issues

#### Metadata Completeness Strategy
```javascript
const metadataCompleteness = {
  BASIC: ['videoId', 'title', 'channelName', 'duration'],
  STANDARD: [...BASIC, 'uploadDate', 'viewCount', 'thumbnails'],
  COMPLETE: [...STANDARD, 'description', 'tags', 'category']
};

// Only navigate for missing STANDARD+ fields
if (!hasStandardMetadata(video)) {
  await enhanceWithVideoPage(video);
}
```

## Technical Considerations

### 1. YouTube Anti-Bot Measures
- **Rate Limiting**: Respect YouTube's request patterns
- **User Agent Spoofing**: Maintain consistent browser fingerprint
- **Behavioral Mimicking**: Simulate human interaction patterns
- **Error Handling**: Graceful degradation when blocked

### 2. Data Storage & Persistence

#### Chrome Storage API (Primary)
- **Quota**: ~5MB for chrome.storage.local (sufficient for ~10,000 video records)
- **Sync Option**: chrome.storage.sync for cross-device synchronization (100KB limit)
- **No File System Access**: All data stored within browser extension sandbox
- **Automatic Backup**: Data persists across browser sessions

#### IndexedDB (Secondary for Large Datasets)
- **Unlimited Storage**: For users with >10,000 videos
- **Complex Queries**: Efficient searching and filtering
- **Structured Data**: Proper database relationships
- **Performance**: Optimized for large dataset operations

#### Storage Implementation
```javascript
class StorageManager {
  async saveVideo(video) {
    // Try Chrome storage first
    try {
      await chrome.storage.local.set({[`video_${video.videoId}`]: video});
    } catch (error) {
      if (error.message.includes('QUOTA_EXCEEDED')) {
        // Fallback to IndexedDB
        await this.saveToIndexedDB(video);
      }
    }
  }

  async getStorageUsage() {
    const usage = await chrome.storage.local.getBytesInUse();
    const quota = chrome.storage.local.QUOTA_BYTES;
    return { usage, quota, percentage: (usage / quota) * 100 };
  }
}
```

#### Export to File System
- **Downloads API**: Export data to user's Downloads folder
- **User Control**: User chooses export location and format
- **No Direct File Access**: Extension cannot read/write arbitrary files
- **Supported Formats**: JSON, CSV, custom formats via blob downloads

## Data Flow & Storage Lifecycle

### Runtime Data Flow
```
YouTube Page → Content Script → Background Script → Storage API → Export
     ↓              ↓               ↓                ↓           ↓
  DOM/Network → Video Metadata → Validation → Chrome Storage → Downloads
```

### Storage Lifecycle Management
```javascript
// Data persistence strategy
const dataLifecycle = {
  // Immediate storage during scraping
  tempStorage: 'chrome.storage.session',  // Cleared on browser restart

  // Persistent storage for verified data
  mainStorage: 'chrome.storage.local',    // Persists across sessions

  // Long-term storage for large datasets
  archiveStorage: 'IndexedDB',            // Unlimited storage

  // User exports
  exportStorage: 'Downloads folder'       // User's file system
};
```

### Storage Quota Management
```javascript
class QuotaManager {
  async checkQuota() {
    const usage = await chrome.storage.local.getBytesInUse();
    const quota = chrome.storage.local.QUOTA_BYTES; // ~5MB

    if (usage > quota * 0.8) {
      // Warn user and suggest export/cleanup
      await this.suggestDataMaintenance();
    }

    if (usage > quota * 0.95) {
      // Auto-migrate to IndexedDB
      await this.migrateToIndexedDB();
    }
  }

  async migrateToIndexedDB() {
    // Move older data to IndexedDB
    // Keep recent data in Chrome storage for quick access
  }
}
```

### Data Backup Strategy
1. **Real-time**: Data saved immediately after scraping each video
2. **Session Backup**: Complete session data saved on completion
3. **Auto-Export**: Weekly automatic exports (user configurable)
4. **Manual Export**: User-triggered exports in multiple formats
5. **Cloud Sync**: Optional sync via chrome.storage.sync (limited to settings)

### 3. Privacy & Security
- **Local Processing**: All data processing happens locally
- **No External Servers**: No data transmission to third parties
- **User Consent**: Clear permissions and data usage disclosure
- **Data Encryption**: Optional encryption for sensitive exports

## Implementation Phases

### Phase 1: Core Functionality
- Basic video metadata scraping
- DOM-based data extraction
- Simple export to JSON/CSV
- Manual backup triggers

### Phase 2: Verification & Removal
- Multi-source verification system
- Safe removal mechanism
- Progress tracking and error handling
- Rollback capabilities

### Phase 3: Advanced Features
- Network request interception
- Automatic backup scheduling
- Advanced export formats
- Data analysis tools

### Phase 4: Polish & Optimization
- Performance optimizations
- Enhanced UI/UX
- Comprehensive error handling
- Documentation and user guides

## Success Metrics

### Primary Metrics
- **Data Completeness**: Percentage of videos with complete metadata
- **Verification Accuracy**: False positive/negative rates for verification
- **Removal Success Rate**: Percentage of successful removals without errors
- **Data Integrity**: Zero data loss during backup and export operations

### Secondary Metrics
- **Performance**: Time to backup 1000 videos
- **User Adoption**: Extension usage and retention rates
- **Error Rates**: Frequency and types of errors encountered
- **Recovery Success**: Ability to restore from backups

## Risks & Mitigation

### Technical Risks
- **YouTube UI Changes**: Regular updates may break scraping logic
  - *Mitigation*: Modular design with easy update mechanisms
- **Anti-Bot Detection**: YouTube may block or limit extension functionality
  - *Mitigation*: Conservative rate limiting and human-like behavior patterns
- **Data Loss**: Storage corruption or browser issues
  - *Mitigation*: Regular exports and backup validation

### User Experience Risks
- **Complexity**: Too many features may confuse users
  - *Mitigation*: Progressive disclosure and sensible defaults
- **Performance**: Slow backup process may frustrate users
  - *Mitigation*: Background processing and clear progress indicators

## Future Enhancements

### Potential Features
- **Cross-Platform Sync**: Sync backups across multiple devices
- **Advanced Analytics**: Video watching patterns and statistics
- **Playlist Management**: Create custom playlists from backed-up videos
- **Social Features**: Share curated video collections
- **API Integration**: Connect with other video management tools

### Scalability Considerations
- **Large Datasets**: Handle users with 10,000+ backed-up videos
- **Performance Optimization**: Efficient search and filtering
- **Storage Management**: Automatic cleanup of old data
- **Export Optimization**: Streaming exports for large datasets

## Conclusion

This Chrome extension provides a comprehensive solution to YouTube's 5000 liked videos limit by implementing a robust backup and removal system. The multi-source verification approach ensures data integrity while the flexible export system maintains long-term accessibility. The phased implementation approach allows for iterative development and user feedback incorporation.

The extension addresses the core problem while providing additional value through advanced data management and export capabilities, making it a valuable tool for YouTube power users who want to preserve their video history beyond platform limitations.
