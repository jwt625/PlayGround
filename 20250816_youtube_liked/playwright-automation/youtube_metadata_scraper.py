"""
YouTube metadata scraper using Playwright.
Minimal implementation of RFD-003 specifications.
"""

import argparse
import asyncio
import json
import logging
import random
import re
import signal
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
from urllib.parse import urlparse, parse_qs

from playwright.async_api import async_playwright, Page, BrowserContext

from config import (
    DEFAULT_SCRAPE_COUNT, DEFAULT_AVERAGE_PAUSE, METADATA_TIMEOUT,
    METADATA_CONTAINER, VIDEO_TITLE_SELECTOR, CHANNEL_NAME_SELECTOR,
    SUBSCRIBER_COUNT_SELECTOR, VIEW_DATE_INFO_SELECTOR, PRECISE_DATE_SELECTOR,
    LIKE_BUTTON_SELECTOR, DISLIKE_BUTTON_SELECTOR, BUTTON_TEXT_SELECTOR,
    DURATION_SELECTOR, COMMENT_COUNT_SELECTOR, DESCRIPTION_SELECTORS,
    HEADLESS_MODE, LOG_LEVEL, LOG_FORMAT
)
from utils.logging import setup_logger


class YouTubeMetadataScraper:
    """
    Playwright-based YouTube metadata scraper.
    """
    
    def __init__(self, headless: bool = HEADLESS_MODE):
        self.logger = setup_logger("metadata_scraper")
        self.headless = headless
        self.scraped_videos: List[str] = []  # Use list to preserve order
        self.scraped_metadata: List[Dict] = []
        self.progress_file = Path("metadata_progress.json")
        self.incremental_file = Path("incremental_metadata.jsonl")  # JSONL format for appending
        
    def load_youtube_liked_json(self, file_path: str) -> List[Dict]:
        """Load and parse youtube_liked.json file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            videos = data.get('videos', [])
            self.logger.info(f"Loaded {len(videos)} videos from {file_path}")
            return videos
            
        except Exception as e:
            self.logger.error(f"Failed to load {file_path}: {e}")
            raise
    
    def clean_video_url(self, url: str) -> str:
        """Remove extra parameters from YouTube URL, keeping only video ID."""
        try:
            parsed = urlparse(url)
            query_params = parse_qs(parsed.query)
            video_id = query_params.get('v', [None])[0]
            
            if video_id:
                return f"https://www.youtube.com/watch?v={video_id}"
            else:
                self.logger.warning(f"No video ID found in URL: {url}")
                return url
                
        except Exception as e:
            self.logger.warning(f"Failed to clean URL {url}: {e}")
            return url
    
    def load_progress(self) -> None:
        """Load previous scraping progress."""
        if self.progress_file.exists():
            try:
                with open(self.progress_file, 'r') as f:
                    progress = json.load(f)
                
                self.scraped_videos = progress.get('scraped_video_ids', [])
                self.scraped_metadata = progress.get('scraped_metadata', [])
                
                self.logger.info(f"Loaded progress: {len(self.scraped_videos)} videos already scraped")
                
            except Exception as e:
                self.logger.warning(f"Failed to load progress: {e}")
    
    def save_progress(self) -> None:
        """Save current scraping progress."""
        try:
            progress = {
                'scraped_video_ids': self.scraped_videos,
                'scraped_metadata': self.scraped_metadata,
                'last_updated': datetime.now().isoformat()
            }

            with open(self.progress_file, 'w') as f:
                json.dump(progress, f, indent=2)

        except Exception as e:
            self.logger.error(f"Failed to save progress: {e}")

    def save_incremental_metadata(self, metadata: Dict) -> None:
        """Append individual video metadata to incremental file immediately for safety."""
        try:
            # Append to JSONL file (one JSON object per line)
            with open(self.incremental_file, 'a', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False)
                f.write('\n')  # New line for JSONL format

        except Exception as e:
            self.logger.warning(f"Failed to save incremental metadata: {e}")

    def cleanup_memory_periodically(self, current_index: int, batch_size: int = 100) -> None:
        """Clean up memory periodically to handle large datasets."""
        if current_index > 0 and current_index % batch_size == 0:
            # Keep only essential data in memory for very large runs
            if len(self.scraped_metadata) > batch_size * 2:
                self.logger.info(f"Memory cleanup: keeping last {batch_size} records in memory")
                # Keep only recent records in memory, older ones are saved to disk
                self.scraped_metadata = self.scraped_metadata[-batch_size:]
    


    async def wait_for_video_load(self, page: Page) -> None:
        """Wait for video player to load without interfering with autoplay."""
        try:
            # Just wait for video player to load - don't click anything
            await page.wait_for_timeout(500)
            self.logger.debug("Waited for video player to load")

        except Exception as e:
            self.logger.debug(f"Failed to wait for video load: {e}")

    async def skip_ads(self, page: Page) -> None:
        """Skip YouTube ads if they appear."""
        try:
            # Wait for video player to load
            await page.wait_for_timeout(1000)

            # Check video duration - if > 10 minutes, it's not an ad
            try:
                duration_element = page.locator('.ytp-time-duration')
                if await duration_element.count() > 0:
                    duration_text = await duration_element.text_content(timeout=2000)
                    if duration_text:
                        # Parse duration (format: "MM:SS" or "HH:MM:SS")
                        duration_parts = duration_text.strip().split(':')
                        if len(duration_parts) >= 2:
                            if len(duration_parts) == 3:  # HH:MM:SS format
                                hours = int(duration_parts[0])
                                minutes = int(duration_parts[1])
                                if hours > 0 or minutes >= 10:  # > 10 minutes
                                    self.logger.debug(f"Video duration {duration_text} > 10min, skipping ad detection")
                                    return
                            elif len(duration_parts) == 2:  # MM:SS format
                                minutes = int(duration_parts[0])
                                if minutes >= 10:  # > 10 minutes
                                    self.logger.debug(f"Video duration {duration_text} > 10min, skipping ad detection")
                                    return
            except Exception as e:
                self.logger.debug(f"Could not check video duration: {e}, proceeding with ad detection")

            # Common ad skip selectors
            skip_selectors = [
                '.ytp-ad-skip-button',
                '.ytp-skip-ad-button',
                'button[class*="skip"]',
                '.videoAdUiSkipButton',
                '.ytp-ad-skip-button-modern',
                'button:has-text("Skip Ad")',
                'button:has-text("Skip Ads")',
                '[aria-label*="Skip ad"]'
            ]

            # Wait up to 10 seconds for skip button to appear
            skip_button_found = False
            for attempt in range(15):  # 10 seconds max wait for skip button
                for selector in skip_selectors:
                    try:
                        skip_button = page.locator(selector)
                        if await skip_button.count() > 0:
                            await skip_button.click()
                            self.logger.debug(f"Skipped ad after {attempt + 1} seconds")
                            await page.wait_for_timeout(500)  # Wait for skip to process
                            skip_button_found = True
                            break
                    except Exception as e:
                        self.logger.debug(f"Skip selector {selector} failed: {e}")
                        continue

                if skip_button_found:
                    break

                await page.wait_for_timeout(500)  # Wait 0.5 second before next attempt

            # If no skip button found, check if we're still in an ad and wait it out
            if not skip_button_found:
                ad_indicators = [
                    '.ytp-ad-player-overlay',
                    '.ytp-ad-text',
                    '.video-ads',
                    '[class*="ad-showing"]'
                ]

                for _ in range(10):  # Check for 10 seconds max for unskippable ads
                    ad_present = False
                    for indicator in ad_indicators:
                        if await page.locator(indicator).count() > 0:
                            ad_present = True
                            break

                    if not ad_present:
                        break

                    await page.wait_for_timeout(500)  # Wait 0.5 second and check again

        except Exception as e:
            self.logger.debug(f"Ad skipping failed: {e}")

    async def extract_video_metadata(self, page: Page, video_info: Dict) -> Dict:
        """Extract comprehensive metadata from a video page."""
        video_id = video_info['videoId']
        clean_url = self.clean_video_url(video_info['url'])
        start_time = time.time()

        try:
            # Navigate to video page
            nav_start = time.time()
            await page.goto(clean_url, timeout=2000)
            nav_time = time.time() - nav_start
            self.logger.debug(f"Navigation took {nav_time:.2f}s")

            # Wait for video player to load (autoplay will handle playback)
            load_start = time.time()
            await self.wait_for_video_load(page)
            load_time = time.time() - load_start
            self.logger.debug(f"Video load wait took {load_time:.2f}s")

            # Skip any ads that might appear
            ad_start = time.time()
            await self.skip_ads(page)
            ad_time = time.time() - ad_start
            self.logger.debug(f"Ad skipping took {ad_time:.2f}s")

            # Wait for main metadata container
            container_start = time.time()
            await page.wait_for_selector(METADATA_CONTAINER, timeout=METADATA_TIMEOUT)
            container_time = time.time() - container_start
            self.logger.debug(f"Metadata container wait took {container_time:.2f}s")

            # Initialize metadata with processing info
            metadata = {
                'scrapedAt': datetime.now().isoformat(),
                'clickedAt': datetime.now().isoformat(),
                'source': 'YouTube Video Info Scraper',
                'url': page.url,
                'videoId': video_id
            }

            # Extract all metadata with timing
            extract_start = time.time()

            # Extract title
            title_element = page.locator(VIDEO_TITLE_SELECTOR)
            if await title_element.count() > 0:
                title = await title_element.text_content()
                if title:
                    metadata['title'] = title.strip()

            # Extract channel information
            channel_element = page.locator(CHANNEL_NAME_SELECTOR)
            if await channel_element.count() > 0:
                channel_name = await channel_element.text_content()
                channel_url = await channel_element.get_attribute('href')
                if channel_name:
                    metadata['channel'] = channel_name.strip()
                if channel_url:
                    metadata['channelUrl'] = channel_url

            # Extract subscriber count
            subscriber_element = page.locator(SUBSCRIBER_COUNT_SELECTOR)
            if await subscriber_element.count() > 0:
                subscriber_count = await subscriber_element.text_content()
                if subscriber_count:
                    metadata['subscriberCount'] = subscriber_count.strip()

            # Extract view count and upload date
            await self._extract_view_and_date_info(page, metadata)

            # Extract engagement metrics
            await self._extract_engagement_metrics(page, metadata)

            # Extract description with fallback strategy
            await self._extract_description_with_fallback(page, metadata)

            # Extract duration
            duration_element = page.locator(DURATION_SELECTOR)
            if await duration_element.count() > 0:
                duration = await duration_element.text_content()
                if duration:
                    metadata['duration'] = duration.strip()

            # Extract comment count (optional)
            comment_element = page.locator(COMMENT_COUNT_SELECTOR)
            if await comment_element.count() > 0:
                comment_count = await comment_element.text_content()
                if comment_count:
                    metadata['commentCount'] = comment_count.strip()

            extract_time = time.time() - extract_start
            total_time = time.time() - start_time
            self.logger.info(f"Successfully extracted metadata for video: {video_id} (extract: {extract_time:.2f}s, total: {total_time:.2f}s)")
            return metadata
            
        except Exception as e:
            self.logger.error(f"Failed to extract metadata for {video_id}: {e}")
            # Return minimal metadata with error info
            return {
                'scrapedAt': datetime.now().isoformat(),
                'clickedAt': datetime.now().isoformat(),
                'source': 'YouTube Video Info Scraper',
                'url': clean_url,
                'videoId': video_id,
                'title': video_info.get('title', ''),
                'channel': video_info.get('channel', ''),
                'error': str(e)
            }
    
    async def _extract_view_and_date_info(self, page: Page, metadata: Dict) -> None:
        """Extract view count and upload date information."""
        try:
            info_element = page.locator(VIEW_DATE_INFO_SELECTOR)
            if await info_element.count() > 0:
                info_text = await info_element.text_content()
                if info_text:
                    info_text = info_text.strip()
                    # Split on multiple spaces like youtube-simple does
                    parts = [part.strip() for part in info_text.split('  ') if part.strip()]
                    if len(parts) >= 2:
                        metadata['viewCount'] = parts[0]
                        metadata['uploadDate'] = parts[1]
                    else:
                        # Fallback parsing for edge cases
                        if 'view' in info_text.lower():
                            view_match = re.search(r'[\d,KMB.]+\s*views?', info_text, re.IGNORECASE)
                            if view_match:
                                metadata['viewCount'] = view_match.group(0)

            # Extract precise date from tooltip
            precise_element = page.locator(PRECISE_DATE_SELECTOR)
            if await precise_element.count() > 0:
                precise_date = await precise_element.text_content()
                if precise_date:
                    metadata['preciseDate'] = precise_date.strip()
                    
        except Exception as e:
            self.logger.warning(f"Failed to extract view/date info: {e}")
    
    async def _extract_engagement_metrics(self, page: Page, metadata: Dict) -> None:
        """Extract like/dislike counts."""
        try:
            # Like button extraction with timeout
            try:
                like_buttons = page.locator(LIKE_BUTTON_SELECTOR)
                like_count = await like_buttons.count()
                if like_count > 0:
                    like_button = like_buttons.first()

                    # Try to get like text with timeout
                    try:
                        like_text_elements = like_button.locator(BUTTON_TEXT_SELECTOR)
                        if await like_text_elements.count() > 0:
                            like_text = await like_text_elements.first().text_content(timeout=3000)
                            if like_text and like_text.strip():
                                metadata['likeCount'] = like_text.strip()
                    except Exception as e:
                        self.logger.debug(f"Failed to get like text: {e}")

                    # Get aria-label for detailed like info with timeout
                    try:
                        aria_label = await like_button.get_attribute('aria-label', timeout=3000)
                        if aria_label:
                            metadata['likeAriaLabel'] = aria_label
                    except Exception as e:
                        self.logger.debug(f"Failed to get like aria-label: {e}")

            except Exception as e:
                self.logger.debug(f"Failed to process like button: {e}")

            # Dislike button extraction with timeout
            try:
                dislike_buttons = page.locator(DISLIKE_BUTTON_SELECTOR)
                dislike_count = await dislike_buttons.count()
                if dislike_count > 0:
                    dislike_button = dislike_buttons.first()

                    # Try to get dislike text with timeout
                    try:
                        dislike_text_elements = dislike_button.locator(BUTTON_TEXT_SELECTOR)
                        if await dislike_text_elements.count() > 0:
                            dislike_text = await dislike_text_elements.first().text_content(timeout=3000)
                            if dislike_text and dislike_text.strip():
                                metadata['dislikeCount'] = dislike_text.strip()
                    except Exception as e:
                        self.logger.debug(f"Failed to get dislike text: {e}")

            except Exception as e:
                self.logger.debug(f"Failed to process dislike button: {e}")

        except Exception as e:
            self.logger.warning(f"Failed to extract engagement metrics: {e}")
    
    async def _extract_description_with_fallback(self, page: Page, metadata: Dict) -> None:
        """Extract description using 4-tier fallback strategy."""
        description_sources = ['expanded', 'snippet', 'container', 'fallback']

        try:
            # Try to click "Show more" button to expand description
            show_more_selectors = [
                'tp-yt-paper-button#expand',
                'ytd-text-inline-expander tp-yt-paper-button',
                'button:has-text("Show more")',
                '#expand'
            ]

            for selector in show_more_selectors:
                try:
                    show_more = page.locator(selector)
                    if await show_more.count() > 0:
                        await show_more.click()
                        await page.wait_for_timeout(500)  # Wait for expansion
                        self.logger.debug("Clicked 'Show more' button")
                        break
                except:
                    continue
        except Exception as e:
            self.logger.debug(f"Failed to expand description: {e}")

        for i, selector in enumerate(DESCRIPTION_SELECTORS):
            try:
                desc_element = page.locator(selector)
                if await desc_element.count() > 0:
                    description = await desc_element.text_content()
                    if description and description.strip():
                        metadata['description'] = description.strip()
                        metadata['descriptionSource'] = description_sources[i]
                        self.logger.debug(f"Description extracted using method {i+1}: {description_sources[i]}")
                        return
            except Exception as e:
                self.logger.debug(f"Description method {i+1} failed: {e}")
                continue

        # No description found
        metadata['description'] = ''
        metadata['descriptionSource'] = 'none'
        self.logger.debug("No description found with any method")

    async def scrape_videos(self, videos: List[Dict], max_videos: int, average_pause: float) -> List[Dict]:
        """Scrape metadata for a list of videos."""
        self.load_progress()

        # Filter out already scraped videos
        videos_to_scrape = [v for v in videos if v['videoId'] not in self.scraped_videos]

        if not videos_to_scrape:
            self.logger.info("All videos have already been scraped")
            return self.scraped_metadata

        # Limit to max_videos
        videos_to_scrape = videos_to_scrape[:max_videos]

        self.logger.info(f"Starting to scrape {len(videos_to_scrape)} videos")
        self.logger.info(f"Average pause between videos: {average_pause} seconds")

        async with async_playwright() as p:
            # Launch browser with session persistence and muted audio
            browser = await p.firefox.launch(
                headless=self.headless,
                args=['--mute-audio', '--disable-audio-output']  # Always mute audio
            )

            # Load saved session if available
            session_file = Path("sessions/browser_context.json")
            if session_file.exists():
                try:
                    context = await browser.new_context(storage_state=str(session_file))
                    self.logger.info("Loaded saved browser session")
                except Exception as e:
                    self.logger.warning(f"Failed to load saved session: {e}")
                    context = await browser.new_context()
            else:
                context = await browser.new_context()

            page = await context.new_page()

            try:
                # Initial YouTube load - navigate to first video and wait for full initialization
                if videos_to_scrape:
                    first_video = videos_to_scrape[0]
                    first_video_url = self.clean_video_url(first_video['url'])

                    self.logger.info(f"Initial YouTube load: navigating to first video {first_video['videoId']}")
                    await page.goto(first_video_url, timeout=10000)

                    self.logger.info("Waiting 30 seconds for YouTube to fully load...")
                    await page.wait_for_timeout(30000)  # 30 seconds for YouTube to fully initialize
                    self.logger.info("Initial YouTube load complete")

                # Process videos
                for i, video_info in enumerate(videos_to_scrape):
                    video_id = video_info['videoId']

                    self.logger.info(f"Processing video {i+1}/{len(videos_to_scrape)}: {video_id}")

                    # Extract metadata
                    metadata = await self.extract_video_metadata(page, video_info)

                    # Add to results
                    self.scraped_metadata.append(metadata)
                    # Add to scraped videos list (preserve order, avoid duplicates)
                    if video_id not in self.scraped_videos:
                        self.scraped_videos.append(video_id)

                    # Save incremental metadata immediately for maximum safety
                    self.save_incremental_metadata(metadata)

                    # Save progress after EVERY video for safety
                    self.save_progress()

                    # Memory management for large datasets
                    self.cleanup_memory_periodically(i + 1)

                    self.logger.info(f"✅ Completed video {i+1}/{len(videos_to_scrape)}: {video_id}")

                    # Random pause between videos (except for the last one)
                    if i < len(videos_to_scrape) - 1:
                        pause_time = random.uniform(
                            average_pause * 0.5,
                            average_pause * 1.5
                        )
                        self.logger.debug(f"Pausing for {pause_time:.2f} seconds")
                        await asyncio.sleep(pause_time)

                # Final progress save
                self.save_progress()

            finally:
                await context.close()
                await browser.close()

        self.logger.info(f"Scraping completed. Total videos scraped: {len(self.scraped_metadata)}")
        return self.scraped_metadata

    def consolidate_incremental_data(self) -> List[Dict]:
        """Consolidate incremental metadata from JSONL file and memory."""
        all_metadata = []

        # Read from incremental JSONL file
        if self.incremental_file.exists():
            try:
                with open(self.incremental_file, 'r', encoding='utf-8') as f:
                    for line_num, line in enumerate(f, 1):
                        line = line.strip()
                        if line:
                            try:
                                metadata = json.loads(line)
                                all_metadata.append(metadata)
                            except json.JSONDecodeError as e:
                                self.logger.warning(f"Failed to parse line {line_num} in incremental file: {e}")

                self.logger.info(f"Loaded {len(all_metadata)} records from incremental file")
            except Exception as e:
                self.logger.warning(f"Failed to read incremental file: {e}")

        # Also include any metadata still in memory (shouldn't be much due to memory cleanup)
        all_metadata.extend(self.scraped_metadata)

        # Remove duplicates based on videoId (keep last occurrence)
        seen_ids = set()
        unique_metadata = []
        for metadata in reversed(all_metadata):  # Reverse to keep last occurrence
            video_id = metadata.get('videoId')
            if video_id and video_id not in seen_ids:
                seen_ids.add(video_id)
                unique_metadata.append(metadata)

        unique_metadata.reverse()  # Restore original order
        return unique_metadata

    def save_final_output(self, output_file: str) -> None:
        """Save the final scraped metadata to a JSON file."""
        try:
            # Consolidate all data including incremental data
            final_metadata = self.consolidate_incremental_data()

            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(final_metadata, f, indent=2, ensure_ascii=False)

            self.logger.info(f"Final output saved to {output_file}")
            print(f"✅ Scraped metadata saved to: {output_file}")
            print(f"📊 Total videos processed: {len(final_metadata)}")

            # Clean up incremental file
            self.cleanup_incremental_file()

        except Exception as e:
            self.logger.error(f"Failed to save final output: {e}")
            raise

    def cleanup_incremental_file(self) -> None:
        """Clean up incremental metadata file after successful consolidation."""
        try:
            if self.incremental_file.exists():
                self.incremental_file.unlink()
                self.logger.info("Cleaned up incremental file")
        except Exception as e:
            self.logger.warning(f"Failed to clean up incremental file: {e}")

    def handle_graceful_shutdown(self, output_file: str = None) -> None:
        """Handle graceful shutdown by consolidating data."""
        try:
            if output_file is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_file = f"scraped_metadata_interrupted_{timestamp}.json"

            print(f"\n⚠️  Graceful shutdown initiated...")
            print(f"💾 Consolidating scraped data...")

            final_metadata = self.consolidate_incremental_data()

            if final_metadata:
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(final_metadata, f, indent=2, ensure_ascii=False)

                print(f"✅ Scraped metadata saved to: {output_file}")
                print(f"📊 Total videos processed: {len(final_metadata)}")
            else:
                print("ℹ️  No metadata to save")

        except Exception as e:
            print(f"❌ Error during graceful shutdown: {e}")
            self.logger.error(f"Graceful shutdown failed: {e}")


async def main():
    """Main function to run the metadata scraper."""
    parser = argparse.ArgumentParser(description='YouTube Metadata Scraper')
    parser.add_argument(
        '--videos', '-v',
        type=int,
        default=DEFAULT_SCRAPE_COUNT,
        help=f'Number of videos to scrape (default: {DEFAULT_SCRAPE_COUNT})'
    )
    parser.add_argument(
        '--pause', '-p',
        type=float,
        default=DEFAULT_AVERAGE_PAUSE,
        help=f'Average pause between videos in seconds (default: {DEFAULT_AVERAGE_PAUSE})'
    )
    parser.add_argument(
        '--input', '-i',
        type=str,
        default='../youtube_liked.json',
        help='Path to youtube_liked.json file (default: ../youtube_liked.json)'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        help='Output file for scraped metadata (default: scraped_metadata_TIMESTAMP.json)'
    )
    parser.add_argument(
        '--headless',
        action='store_true',
        help='Run browser in headless mode'
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)

    # Generate output filename if not provided
    if args.output is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = f"scraped_metadata_{timestamp}.json"

    print("🚀 YouTube Metadata Scraper")
    print(f"📁 Input file: {args.input}")
    print(f"📊 Videos to scrape: {args.videos}")
    print(f"⏱️  Average pause: {args.pause} seconds")
    print(f"💾 Output file: {args.output}")
    print(f"🖥️  Headless mode: {args.headless}")
    print("⚠️  Press Ctrl+C to stop gracefully")
    print()

    # Initialize scraper for signal handling
    scraper = None

    def signal_handler(signum, frame):
        """Handle graceful shutdown on Ctrl+C."""
        _ = signum, frame  # Suppress unused parameter warnings
        if scraper:
            scraper.handle_graceful_shutdown(args.output)
        sys.exit(0)

    # Set up signal handler
    signal.signal(signal.SIGINT, signal_handler)

    try:
        # Initialize scraper
        scraper = YouTubeMetadataScraper(headless=args.headless)

        # Load video list
        videos = scraper.load_youtube_liked_json(args.input)

        if not videos:
            print("❌ No videos found in input file")
            return 1

        # Start scraping
        await scraper.scrape_videos(videos, args.videos, args.pause)

        # Save final output
        scraper.save_final_output(args.output)

        print("✅ Scraping completed successfully!")
        return 0

    except KeyboardInterrupt:
        # This should be handled by signal handler, but just in case
        if scraper:
            scraper.handle_graceful_shutdown(args.output)
        return 1

    except Exception as e:
        print(f"❌ Error: {e}")
        logging.error(f"Scraping failed: {e}")
        if scraper:
            scraper.handle_graceful_shutdown(args.output)
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
