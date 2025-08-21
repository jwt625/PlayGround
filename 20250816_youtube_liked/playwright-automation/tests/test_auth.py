"""
Test script for YouTube authentication and session management.
"""

from playwright.sync_api import sync_playwright
from utils.auth import YouTubeAuth
from utils.logging import setup_logger

def test_authentication():
    """Test the authentication flow."""
    logger = setup_logger()
    auth = YouTubeAuth()
    
    print("=== YouTube Authentication Test ===")
    print("This will test the login flow and session persistence.")
    print()
    
    with sync_playwright() as p:
        # Start with fresh browser
        browser = p.firefox.launch(headless=False)
        context = browser.new_context()
        page = context.new_page()
        
        try:
            # Test authentication
            print("Testing authentication flow...")
            if auth.handle_authentication(context, page):
                print("Authentication successful!")

                # Test that we can access liked videos
                print("Testing access to liked videos...")
                if auth.navigate_to_liked_videos(page):
                    print("Successfully accessed liked videos page")
                else:
                    print("Failed to access liked videos page")

                # Test session saving
                print("Testing session persistence...")
                if auth.save_session_info(context):
                    print("Session saved successfully")
                else:
                    print("Failed to save session")

            else:
                print("Authentication failed")
                
        except Exception as e:
            logger.error(f"Test failed: {e}")
            print(f"Test failed: {e}")
            
        finally:
            input("\nPress Enter to close browser...")
            context.close()
            browser.close()

def test_session_reuse():
    """Test reusing a saved session."""
    logger = setup_logger()
    auth = YouTubeAuth()
    
    print("\n=== Session Reuse Test ===")
    
    # Check if we have a saved session
    session_info = auth.load_session_info()
    if not session_info:
        print("No saved session found. Run authentication test first.")
        return

    print("Testing saved session...")
    
    with sync_playwright() as p:
        browser = p.firefox.launch(headless=False)

        # Load context with saved session
        context = browser.new_context(storage_state=str(auth.context_file))
        page = context.new_page()
        
        try:
            # Navigate to YouTube and check if we're logged in
            page.goto("https://www.youtube.com", timeout=30000)
            page.wait_for_timeout(2000)
            
            if auth.verify_logged_in(page):
                print("Saved session is valid - still logged in!")

                # Test accessing liked videos
                if auth.navigate_to_liked_videos(page):
                    print("Successfully accessed liked videos with saved session")
                else:
                    print("Failed to access liked videos")
            else:
                print("Saved session is no longer valid")
                
        except Exception as e:
            logger.error(f"Session reuse test failed: {e}")
            print(f"Session reuse test failed: {e}")
            
        finally:
            input("\nPress Enter to close browser...")
            context.close()
            browser.close()

def main():
    """Run authentication tests."""
    print("Choose a test to run:")
    print("1. Test authentication flow (login and save session)")
    print("2. Test session reuse (use saved session)")
    print("3. Clear saved session")
    print()
    
    choice = input("Enter choice (1-3): ").strip()
    
    if choice == "1":
        test_authentication()
    elif choice == "2":
        test_session_reuse()
    elif choice == "3":
        auth = YouTubeAuth()
        auth.clear_session()
        print("Saved session cleared")
    else:
        print("Invalid choice")

if __name__ == "__main__":
    main()
