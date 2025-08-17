"""
Authentication utilities for YouTube automation with session persistence.
"""

import json
import logging
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, Any
from playwright.sync_api import Page, BrowserContext


class YouTubeAuth:
    """
    Handles YouTube authentication with session persistence.
    """
    
    def __init__(self, session_dir: str = "sessions"):
        self.logger = logging.getLogger("youtube_remover")
        self.session_dir = Path(session_dir)
        self.session_dir.mkdir(exist_ok=True)
        self.session_file = self.session_dir / "youtube_session.json"
        self.context_dir = self.session_dir / "browser_context"
        
    def save_session_info(self, context: BrowserContext) -> bool:
        """
        Save session information and browser context.
        
        Args:
            context: Browser context to save
            
        Returns:
            True if session saved successfully
        """
        try:
            # Save browser context (cookies, localStorage, etc.)
            context.storage_state(path=str(self.context_dir))
            
            # Save session metadata
            session_info = {
                "saved_at": datetime.now().isoformat(),
                "expires_at": (datetime.now() + timedelta(days=30)).isoformat(),
                "context_path": str(self.context_dir)
            }
            
            with open(self.session_file, 'w') as f:
                json.dump(session_info, f, indent=2)
            
            self.logger.info("Session saved successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save session: {e}")
            return False
    
    def load_session_info(self) -> Optional[Dict[str, Any]]:
        """
        Load session information.
        
        Returns:
            Session info dict or None if not available/expired
        """
        try:
            if not self.session_file.exists():
                self.logger.debug("No saved session found")
                return None
            
            with open(self.session_file, 'r') as f:
                session_info = json.load(f)
            
            # Check if session is expired
            expires_at = datetime.fromisoformat(session_info["expires_at"])
            if datetime.now() > expires_at:
                self.logger.info("Saved session has expired")
                self.clear_session()
                return None
            
            # Check if context file exists
            context_path = Path(session_info["context_path"])
            if not context_path.exists():
                self.logger.warning("Session context file not found")
                return None
            
            self.logger.info("Valid saved session found")
            return session_info
            
        except Exception as e:
            self.logger.error(f"Failed to load session: {e}")
            return None
    
    def clear_session(self):
        """Clear saved session data."""
        try:
            if self.session_file.exists():
                self.session_file.unlink()
            if self.context_dir.exists():
                # Remove context directory and contents
                import shutil
                shutil.rmtree(self.context_dir)
            self.logger.info("Session data cleared")
        except Exception as e:
            self.logger.error(f"Failed to clear session: {e}")
    
    def verify_logged_in(self, page: Page) -> bool:
        """
        Verify that user is logged in to YouTube.
        
        Args:
            page: Playwright page instance
            
        Returns:
            True if user appears to be logged in
        """
        try:
            # Look for user avatar or account menu
            # YouTube shows these when logged in
            selectors_to_check = [
                'button[aria-label*="Account menu"]',
                'img[alt*="Avatar"]',
                '#avatar-btn',
                'ytd-topbar-menu-button-renderer #avatar-btn'
            ]
            
            for selector in selectors_to_check:
                try:
                    element = page.locator(selector)
                    if element.count() > 0:
                        self.logger.debug(f"Login verified with selector: {selector}")
                        return True
                except:
                    continue
            
            # Additional check: look for "Sign in" button (indicates not logged in)
            sign_in_selectors = [
                'a[aria-label="Sign in"]',
                'ytd-button-renderer:has-text("Sign in")',
                'paper-button:has-text("SIGN IN")'
            ]
            
            for selector in sign_in_selectors:
                try:
                    element = page.locator(selector)
                    if element.count() > 0:
                        self.logger.debug("Sign in button found - user not logged in")
                        return False
                except:
                    continue
            
            self.logger.warning("Could not definitively verify login status")
            return False
            
        except Exception as e:
            self.logger.error(f"Error verifying login status: {e}")
            return False
    
    def navigate_to_liked_videos(self, page: Page) -> bool:
        """
        Navigate to the liked videos page and verify we're there.
        
        Args:
            page: Playwright page instance
            
        Returns:
            True if successfully navigated to liked videos
        """
        try:
            self.logger.info("Navigating to liked videos...")
            page.goto("https://www.youtube.com/playlist?list=LL", timeout=30000)
            
            # Wait for page to load and verify we're on the right page
            page.wait_for_selector("ytd-playlist-header-renderer", timeout=15000)
            
            # Check if we can see playlist videos or if we need to sign in
            if page.locator('ytd-playlist-video-renderer').count() > 0:
                self.logger.info("Successfully navigated to liked videos")
                return True
            elif page.locator('text="Sign in"').count() > 0:
                self.logger.warning("Liked videos page requires sign in")
                return False
            else:
                # Might be empty playlist or loading
                self.logger.info("On liked videos page (may be empty)")
                return True
                
        except Exception as e:
            self.logger.error(f"Failed to navigate to liked videos: {e}")
            return False
    
    def handle_authentication(self, context: BrowserContext, page: Page) -> bool:
        """
        Complete authentication flow with session persistence.
        
        Args:
            context: Browser context
            page: Playwright page instance
            
        Returns:
            True if authentication successful
        """
        self.logger.info("Starting authentication flow...")
        
        # Try to load saved session
        session_info = self.load_session_info()
        if session_info and self.context_dir.exists():
            self.logger.info("Attempting to use saved session...")
            try:
                # The context should already be loaded with the saved state
                # Just verify we're still logged in
                page.goto("https://www.youtube.com", timeout=30000)
                page.wait_for_timeout(2000)  # Give page time to load
                
                if self.verify_logged_in(page):
                    self.logger.info("✅ Saved session is still valid")
                    return self.navigate_to_liked_videos(page)
                else:
                    self.logger.info("Saved session is no longer valid")
                    self.clear_session()
            except Exception as e:
                self.logger.warning(f"Error using saved session: {e}")
                self.clear_session()
        
        # Manual login flow
        self.logger.info("Starting manual login flow...")
        print("\nYouTube login required:")
        print("1. Browser will open to YouTube")
        print("2. Log in to your account")
        print("3. Return here and press Enter")
        
        try:
            # Navigate to YouTube
            page.goto("https://www.youtube.com", timeout=30000)
            
            # Wait for user to log in
            input("\nPress Enter after logging in to YouTube...")

            # Verify login
            if not self.verify_logged_in(page):
                print("Login verification failed. Please ensure you're logged in.")
                return False

            print("Login verified successfully!")

            # Save session for future use
            if self.save_session_info(context):
                print("Session saved for future use")
            
            # Navigate to liked videos
            return self.navigate_to_liked_videos(page)
            
        except Exception as e:
            self.logger.error(f"Authentication flow failed: {e}")
            print(f"Authentication failed: {e}")
            return False
