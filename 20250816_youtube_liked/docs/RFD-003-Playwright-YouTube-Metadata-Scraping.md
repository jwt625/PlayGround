# RFD-003: Playwright YouTube Metadata Scraping Implementation

**Status**: Proposed  
**Author**: AI Assistant  
**Date**: 2025-08-17  
**Related**: RFD-000 (YouTube Liked Videos Backup Extension), RFD-002 (Playwright YouTube Video Removal)

## Summary

This RFD proposes a Playwright-based system to extract comprehensive metadata from YouTube videos in the liked videos list. Building on the successful video removal automation (RFD-002), this system will navigate to individual video pages to collect detailed metadata that cannot be obtained from the playlist view alone.

## Background

### Current State

The existing system provides:
- **Video List**: 4,955 liked videos with basic metadata (title, channel, videoId) from `youtube_liked.json`
- **Removal Automation**: Successful Playwright-based video removal system
- **Sample Detailed Data**: Examples in `detailed.json` showing rich metadata structure
- **Chrome Extension**: Working metadata extraction from individual video pages

### Problem Statement

The current video list contains only basic metadata extracted from the playlist view:
- Title, channel name, video ID, URL
- Missing: descriptions, view counts, like counts, upload dates, duration, subscriber counts

To create a comprehensive backup before video removal, we need detailed metadata for each video, requiring navigation to individual video pages.

## Goals

### Primary Goals
1. **Comprehensive Metadata Collection**: Extract all available metadata from video pages
2. **Batch Processing**: Process videos from the liked list efficiently
3. **Resume Capability**: Handle interruptions and resume from last processed video
4. **Data Validation**: Ensure metadata completeness and accuracy
5. **Rate Limiting**: Respect YouTube's usage patterns to avoid detection

### Secondary Goals
1. **Progress Tracking**: Real-time progress reporting and logging
2. **Error Handling**: Graceful handling of unavailable/private videos
3. **Export Integration**: Compatible output format for existing systems
4. **Performance Optimization**: Minimize processing time while maintaining reliability

## Architecture Overview

### System Components

```
YouTube Metadata Scraper
├── metadata_scraper.py          # Main orchestration script
├── video_processor.py           # Individual video page processing
├── data_models.py              # Data structures and validation
├── session_manager.py          # Browser session and authentication
├── progress_tracker.py         # Progress persistence and resumption
├── rate_limiter.py            # Request throttling and timing
└── export_manager.py          # Data export and formatting
```

### Data Flow

```
youtube_liked.json → Video Queue → Individual Video Processing → Metadata Extraction → Validation → Storage → Export
```

## Implementation Strategy

### Phase 1: Core Metadata Extraction

#### Input Data Source
- Load video list from `youtube_liked.json` (4,955 videos)
- Extract video IDs and basic information
- Create processing queue with resume capability

#### Video Page Navigation
```python
async def process_video(video_id: str, basic_info: dict) -> dict:
    """Navigate to video page and extract comprehensive metadata."""
    url = f"https://www.youtube.com/watch?v={video_id}"
    await page.goto(url)
    await page.wait_for_selector('ytd-watch-metadata')
    
    metadata = await extract_video_metadata(page)
    return merge_with_basic_info(basic_info, metadata)
```

#### Metadata Extraction Strategy
Based on the successful Chrome extension approach (`youtube-simple/content.js`):

1. **Primary Data Sources**:
   - `ytd-watch-metadata` element (main container)
   - Video player elements for duration
   - Comment section for engagement metrics

2. **Extracted Fields** (matching `detailed.json` format exactly):
   ```python
   VideoMetadata = {
       # Core identifiers (from playlist + video page)
       'videoId': str,              # "rCsfW22MCO0"
       'title': str,                # "My net worth in salt forks"
       'url': str,                  # "https://www.youtube.com/watch?v=rCsfW22MCO0&ab_channel=BenWalker"

       # Channel information
       'channel': str,              # "Ben Walker"
       'channelUrl': str,           # "https://www.youtube.com/@bnwlkr"
       'subscriberCount': str,      # "234K subscribers"

       # Content metadata
       'description': str,          # Full or snippet description
       'descriptionSource': str,    # "snippet", "expanded", "container", "fallback", "none"
       'duration': str,             # "10:43"

       # Upload and timing info
       'uploadDate': str,           # "2 days ago" or "Streamed live on Jul 16, 2025"
       'preciseDate': str,          # "230,090 views • Aug 14, 2025"

       # Engagement metrics
       'viewCount': str,            # "230K views" or "1,025,114 views"
       'likeCount': str,            # "22K" or "44K"
       'likeAriaLabel': str,        # "like this video along with 22,765 other people"
       'dislikeCount': str,         # "200" or "70"
       'commentCount': str,         # "3,860 Comments" (optional, not always present)

       # Processing metadata
       'scrapedAt': str,            # "2025-08-17T00:24:33.289Z"
       'clickedAt': str,            # "2025-08-17T00:24:33.289Z" (same as scrapedAt)
       'source': str,               # "YouTube Video Info Scraper"
   }
   ```

### Phase 2: Batch Processing and Rate Limiting

#### Processing Strategy
- **Batch Size**: Process 10-20 videos per batch
- **Rate Limiting**: 2-3 second delay between video page loads
- **Session Management**: Reuse browser context with saved authentication
- **Progress Persistence**: Save progress every 50 videos

#### Error Handling
```python
class VideoProcessingError(Exception):
    """Custom exception for video processing failures."""
    pass

async def safe_process_video(video_info: dict) -> dict:
    """Process video with comprehensive error handling."""
    try:
        return await process_video(video_info['videoId'], video_info)
    except VideoProcessingError as e:
        logger.warning(f"Failed to process {video_info['videoId']}: {e}")
        return create_error_record(video_info, str(e))
```

### Phase 3: Data Management and Export

#### Storage Strategy
- **Incremental Storage**: Save metadata as videos are processed
- **Resume Capability**: Track processed videos to enable resumption
- **Data Validation**: Verify required fields are present
- **Backup Creation**: Regular backups during processing

#### Export Formats
1. **JSON**: Complete metadata with nested structure (compatible with `detailed.json`)
2. **CSV**: Flattened format for analysis
3. **Incremental Updates**: Merge with existing data

## Technical Implementation

### Browser Session Management
```python
class YouTubeSession:
    """Manages Playwright browser session with authentication."""
    
    async def initialize(self):
        """Initialize browser with saved session."""
        self.context = await self.browser.new_context(
            storage_state="sessions/youtube_session.json"
        )
        self.page = await self.context.new_page()
    
    async def navigate_to_video(self, video_id: str):
        """Navigate to video page with error handling."""
        url = f"https://www.youtube.com/watch?v={video_id}"
        await self.page.goto(url, wait_until="domcontentloaded")
```

### Metadata Extraction Engine
Based on proven selectors from `youtube-simple/content.js`:

```python
class MetadataExtractor:
    """Extracts metadata from YouTube video pages."""
    
    async def extract_all_metadata(self, page) -> dict:
        """Extract comprehensive metadata from video page."""
        metadata = {}
        
        # Title and basic info
        metadata.update(await self.extract_title_info(page))
        
        # Channel information
        metadata.update(await self.extract_channel_info(page))
        
        # Engagement metrics
        metadata.update(await self.extract_engagement_metrics(page))
        
        # Description and content
        metadata.update(await self.extract_description(page))
        
        # Technical details
        metadata.update(await self.extract_technical_info(page))
        
        return metadata
```

### Progress Tracking System
```python
class ProgressTracker:
    """Tracks processing progress and enables resumption."""
    
    def __init__(self, total_videos: int):
        self.total_videos = total_videos
        self.processed_videos = []
        self.failed_videos = []
        self.current_index = 0
    
    def save_progress(self):
        """Save current progress to disk."""
        progress_data = {
            'processed_count': len(self.processed_videos),
            'failed_count': len(self.failed_videos),
            'current_index': self.current_index,
            'timestamp': datetime.now().isoformat()
        }
        with open('progress.json', 'w') as f:
            json.dump(progress_data, f, indent=2)
```

## Rate Limiting and Anti-Detection

### Request Timing
- **Page Load Delay**: 2-3 seconds between video page navigations
- **Random Jitter**: ±500ms variation to simulate human behavior
- **Batch Breaks**: 30-second pause every 50 videos
- **Session Rotation**: New browser context every 500 videos

### Human-like Behavior
```python
async def human_like_navigation(page, video_id: str):
    """Navigate with human-like timing and behavior."""
    # Random delay before navigation
    await asyncio.sleep(random.uniform(2.0, 3.5))
    
    # Navigate to video
    await page.goto(f"https://www.youtube.com/watch?v={video_id}")
    
    # Wait for content to load
    await page.wait_for_selector('ytd-watch-metadata', timeout=10000)
    
    # Small delay to simulate reading
    await asyncio.sleep(random.uniform(0.5, 1.5))
```

## Data Schema and Validation

### Enhanced Video Record
```python
@dataclass
class VideoRecord:
    # Core identifiers
    videoId: str
    title: str
    url: str
    
    # Channel information
    channel: str
    channelUrl: Optional[str] = None
    subscriberCount: Optional[str] = None
    
    # Content metadata
    description: Optional[str] = None
    duration: Optional[str] = None
    uploadDate: Optional[str] = None
    preciseDate: Optional[str] = None
    
    # Engagement metrics
    viewCount: Optional[str] = None
    likeCount: Optional[str] = None
    dislikeCount: Optional[str] = None
    commentCount: Optional[str] = None
    
    # Processing metadata
    scrapedAt: str = field(default_factory=lambda: datetime.now().isoformat())
    source: str = "Playwright YouTube Metadata Scraper"
    processingStatus: str = "success"
    errorMessage: Optional[str] = None
```

## Testing Strategy

### Initial Testing Phase
1. **Small Batch Test**: Process first 10 videos from the list
2. **Data Validation**: Compare output with `detailed.json` examples
3. **Error Handling**: Test with known private/deleted videos
4. **Performance Measurement**: Time per video and total throughput

### Validation Criteria
- All required fields present for accessible videos
- Consistent data format with existing examples
- Proper error handling for inaccessible content
- Resume functionality works correctly

## Success Metrics

### Primary Metrics
- **Completion Rate**: Percentage of videos successfully processed
- **Data Completeness**: Average number of fields populated per video
- **Processing Speed**: Videos processed per hour
- **Error Rate**: Percentage of videos that fail processing

### Secondary Metrics
- **Resume Accuracy**: Successful resumption after interruption
- **Rate Limit Compliance**: No YouTube blocking or throttling
- **Data Quality**: Consistency with manual verification samples

## Risk Mitigation

### Technical Risks
1. **YouTube UI Changes**: Use robust selectors with fallbacks
2. **Rate Limiting**: Conservative timing with monitoring
3. **Session Expiration**: Automatic re-authentication
4. **Memory Usage**: Process in batches with cleanup

### Operational Risks
1. **Long Processing Time**: ~4-6 hours for 5000 videos
2. **Network Interruptions**: Resume capability and progress saving
3. **Data Loss**: Regular backups and incremental saves

## Future Enhancements

### Potential Improvements
1. **Parallel Processing**: Multiple browser contexts for faster processing
2. **Smart Caching**: Skip recently processed videos
3. **Quality Scoring**: Metadata completeness scoring
4. **Integration**: Direct integration with removal system

## Conclusion

This Playwright-based metadata scraping system builds on the proven success of the video removal automation while addressing the need for comprehensive video metadata collection. The phased approach ensures reliability while the robust error handling and resume capability make it suitable for processing large video collections.

The system will provide the detailed metadata needed for comprehensive video backup before removal, enabling users to maintain complete records of their YouTube liked videos beyond the platform's 5000 video limit.
