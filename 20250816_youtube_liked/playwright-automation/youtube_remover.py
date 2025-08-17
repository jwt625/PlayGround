"""
YouTube video removal automation using Playwright.
Based on RFD-002 implementation specifications.
"""

import time
from datetime import datetime
from typing import Tuple, Optional
from playwright.sync_api import sync_playwright, Page, Locator

from config import (
    YOUTUBE_LIKED_URL, DEFAULT_REMOVAL_COUNT, HEADLESS_MODE,
    WAIT_BETWEEN_REMOVALS, MAX_RETRIES, VIDEO_SELECTOR,
    ACTION_MENU_SELECTOR, REMOVE_OPTION_SELECTOR, TITLE_SELECTOR,
    PAGE_LOAD_TIMEOUT, ELEMENT_WAIT_TIMEOUT, REMOVAL_WAIT_TIMEOUT
)
from utils.logging import setup_logger, log_removal_progress, log_error_with_context
from utils.auth import YouTubeAuth


class YouTubeVideoRemover:
    """
    Playwright-based YouTube video removal automation.
    """
    
    def __init__(self, headless: bool = HEADLESS_MODE):
        self.logger = setup_logger()
        self.headless = headless
        self.removed_count = 0
        self.removed_videos = []
        self.auth = YouTubeAuth()
        
    def safe_remove_video(self, page: Page, video_element: Locator) -> Tuple[bool, Optional[str]]:
        """
        Safely remove a single video with retry logic.
        
        Args:
            page: Playwright page instance
            video_element: Video element locator
            
        Returns:
            Tuple of (success, video_title)
        """
        for attempt in range(MAX_RETRIES):
            try:
                # Get title before removal
                title_element = video_element.locator(TITLE_SELECTOR)
                title = title_element.text_content(timeout=ELEMENT_WAIT_TIMEOUT)
                
                if not title:
                    title = "Unknown Title"
                
                self.logger.debug(f"Attempt {attempt + 1} to remove: {title}")
                
                # Click action menu button
                action_menu = video_element.locator(ACTION_MENU_SELECTOR)
                action_menu.wait_for(state="visible", timeout=ELEMENT_WAIT_TIMEOUT)
                action_menu.click()
                
                # Wait for popup menu and click remove option
                remove_button = page.locator(REMOVE_OPTION_SELECTOR)
                remove_button.wait_for(state="visible", timeout=ELEMENT_WAIT_TIMEOUT)
                remove_button.click()
                
                # Wait for removal to complete and verify the action worked
                page.wait_for_timeout(REMOVAL_WAIT_TIMEOUT)

                # Additional wait to ensure YouTube processes the removal
                page.wait_for_timeout(1000)

                self.logger.debug(f"Successfully removed: {title}")
                return True, title
                
            except Exception as e:
                log_error_with_context(
                    self.logger, e, 
                    f"removing video on attempt {attempt + 1}"
                )
                
                if attempt < MAX_RETRIES - 1:
                    self.logger.info(f"Retrying in 2 seconds...")
                    page.wait_for_timeout(2000)
                else:
                    self.logger.error(f"Failed to remove video after {MAX_RETRIES} attempts")
        
        return False, None
    
    def remove_top_videos(self, count: int = DEFAULT_REMOVAL_COUNT) -> int:
        """
        Remove the specified number of videos from the top of the liked videos list.

        Args:
            count: Number of videos to remove

        Returns:
            Number of videos actually removed
        """
        self.logger.info(f"Starting removal of {count} videos from YouTube liked list")

        with sync_playwright() as p:
            browser = p.firefox.launch(headless=self.headless)

            # Try to load saved session context
            session_info = self.auth.load_session_info()
            if session_info and self.auth.context_file.exists():
                self.logger.info("Loading browser with saved session...")
                context = browser.new_context(storage_state=str(self.auth.context_file))
            else:
                self.logger.info("Starting fresh browser session...")
                context = browser.new_context()

            page = context.new_page()

            try:
                # Handle authentication
                if not self.auth.handle_authentication(context, page):
                    self.logger.error("Authentication failed")
                    return 0

                # Verify we're on the liked videos page
                page.wait_for_selector(VIDEO_SELECTOR, timeout=PAGE_LOAD_TIMEOUT)
                self.logger.info("Ready to start video removal")
                
                # Get all videos once at the beginning
                page.wait_for_selector(VIDEO_SELECTOR, timeout=PAGE_LOAD_TIMEOUT)
                all_videos = page.locator(VIDEO_SELECTOR).all()
                total_videos_available = len(all_videos)

                self.logger.info(f"Found {total_videos_available} videos in playlist")

                if total_videos_available == 0:
                    self.logger.warning("No videos found in the playlist")
                    return 0

                # Limit count to available videos
                actual_count = min(count, total_videos_available)
                self.logger.info(f"Will remove {actual_count} videos")

                self.removed_count = 0
                consecutive_failures = 0
                max_consecutive_failures = 5

                # Go through the list sequentially
                videos_removed_in_current_batch = 0
                current_video_index = 0

                for removal_iteration in range(actual_count):
                    try:
                        # Refresh page every 100 videos if count > 100
                        if count > 100 and videos_removed_in_current_batch >= 100:
                            self.logger.info(f"Refreshing page after removing {videos_removed_in_current_batch} videos (total: {self.removed_count})...")
                            page.reload()
                            page.wait_for_selector(VIDEO_SELECTOR, timeout=PAGE_LOAD_TIMEOUT)

                            # Re-fetch video elements after refresh
                            all_videos = page.locator(VIDEO_SELECTOR).all()
                            self.logger.info(f"Page refreshed. Found {len(all_videos)} videos remaining")

                            # Reset batch counter and start from the top of the refreshed list
                            videos_removed_in_current_batch = 0
                            current_video_index = 0

                            if not all_videos:
                                self.logger.warning("No videos found after refresh")
                                break

                        # Target the current video (always from the top since removed videos disappear)
                        if current_video_index >= len(all_videos):
                            self.logger.warning(f"No more videos available at index {current_video_index}")
                            break

                        target_video = all_videos[current_video_index]
                        self.logger.debug(f"Targeting video at index {current_video_index} (removal #{self.removed_count + 1})")

                        # Wait for the video element to be stable and visible
                        target_video.wait_for(state="visible", timeout=500)

                        # Attempt to remove the video
                        success, title = self.safe_remove_video(page, target_video)

                        if success:
                            self.removed_count += 1
                            videos_removed_in_current_batch += 1
                            consecutive_failures = 0

                            # Log the removed video
                            video_info = {
                                "title": title,
                                "removed_at": datetime.now().isoformat(),
                                "position": self.removed_count
                            }
                            self.removed_videos.append(video_info)

                            log_removal_progress(self.logger, self.removed_count, actual_count, title)

                            # Wait between removals to avoid rate limiting
                            if WAIT_BETWEEN_REMOVALS > 0:
                                page.wait_for_timeout(WAIT_BETWEEN_REMOVALS)

                            # Don't increment current_video_index since the video was removed and list shifted
                        else:
                            consecutive_failures += 1
                            self.logger.warning(f"Failed to remove video at index {current_video_index}. Consecutive failures: {consecutive_failures}")

                            if consecutive_failures >= max_consecutive_failures:
                                self.logger.error(f"Too many consecutive failures ({consecutive_failures}). Stopping.")
                                break

                            # Move to next video if removal failed
                            current_video_index += 1

                    except Exception as e:
                        log_error_with_context(self.logger, e, f"removing video at index {current_video_index}")
                        consecutive_failures += 1

                        if consecutive_failures >= max_consecutive_failures:
                            self.logger.error("Too many consecutive failures. Stopping.")
                            break

                        # Wait before continuing and move to next video
                        page.wait_for_timeout(1000)
                        current_video_index += 1
                
            except Exception as e:
                log_error_with_context(self.logger, e, "during browser automation")

            finally:
                # Save removal log
                if hasattr(self, 'removed_videos') and self.removed_videos:
                    import json
                    from pathlib import Path

                    log_dir = Path("logs")
                    log_dir.mkdir(exist_ok=True)

                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    removal_log_file = log_dir / f"removed_videos_{timestamp}.json"

                    log_data = {
                        "removal_session": {
                            "started_at": datetime.now().isoformat(),
                            "total_removed": len(self.removed_videos),
                            "target_count": count
                        },
                        "removed_videos": self.removed_videos
                    }

                    with open(removal_log_file, 'w', encoding='utf-8') as f:
                        json.dump(log_data, f, indent=2, ensure_ascii=False)

                    self.logger.info(f"Removal log saved to: {removal_log_file}")

                context.close()
                browser.close()

        self.logger.info(f"Removal completed. Successfully removed {self.removed_count} videos")
        return self.removed_count
    
    def integrated_removal_workflow(self, videos_to_remove: int = DEFAULT_REMOVAL_COUNT) -> bool:
        """
        Complete workflow with backup verification and video removal.
        
        Args:
            videos_to_remove: Number of videos to remove
            
        Returns:
            True if workflow completed successfully
        """
        self.logger.info("Starting integrated removal workflow")
        
        # Initialize removal tracking
        self.removed_videos = []
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.logger.info(f"Starting video removal session at {timestamp}")
        
        # Remove videos with Playwright
        removed_count = self.remove_top_videos(videos_to_remove)
        
        # Log final results
        success = removed_count == videos_to_remove
        if success:
            self.logger.info(f"✅ Workflow completed successfully! Removed {removed_count} videos")
        else:
            self.logger.warning(f"⚠️ Workflow partially completed. Removed {removed_count}/{videos_to_remove} videos")
        
        return success

    def clear_saved_session(self):
        """
        Clear saved authentication session.
        Useful for troubleshooting or switching accounts.
        """
        self.auth.clear_session()
        self.logger.info("Saved session cleared")


def main():
    """
    Main entry point for the YouTube video remover.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Remove videos from YouTube liked list")
    parser.add_argument("--count", type=int, default=DEFAULT_REMOVAL_COUNT,
                       help=f"Number of videos to remove (default: {DEFAULT_REMOVAL_COUNT})")
    parser.add_argument("--headless", action="store_true",
                       help="Run in headless mode")
    parser.add_argument("--clear-session", action="store_true",
                       help="Clear saved login session and exit")
    parser.add_argument("--force-login", action="store_true",
                       help="Force new login (ignore saved session)")
    
    args = parser.parse_args()

    remover = YouTubeVideoRemover(headless=args.headless)

    # Handle session management commands
    if args.clear_session:
        remover.clear_saved_session()
        print("Saved session cleared")
        return

    if args.force_login:
        remover.clear_saved_session()
        print("Forcing new login...")

    # Run the removal workflow
    success = remover.integrated_removal_workflow(args.count)
    if not success:
        exit(1)


if __name__ == "__main__":
    main()
