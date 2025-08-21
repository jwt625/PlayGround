"""
Test script to verify Playwright setup and basic functionality.
"""

from playwright.sync_api import sync_playwright
import sys
import os

def test_playwright_setup():
    """Test basic Playwright functionality."""
    print("Testing Playwright setup...")
    
    try:
        with sync_playwright() as p:
            print("Playwright imported successfully")

            # Test browser launch
            browser = p.chromium.launch(headless=True)
            print("Chromium browser launched successfully")

            # Test page creation
            page = browser.new_page()
            print("Page created successfully")

            # Test navigation to a simple page
            page.goto("https://www.google.com", timeout=30000)
            print("Navigation to Google successful")

            # Test basic selector
            title = page.title()
            print(f"Page title retrieved: {title}")

            # Close page first, then browser
            page.close()
            browser.close()
            print("Browser closed successfully")

        print("\nAll tests passed! Playwright setup is working correctly.")
        return True

    except Exception as e:
        print(f"Test failed: {e}")
        return False

def test_imports():
    """Test that all required modules can be imported."""
    print("Testing module imports...")
    
    try:
        import config
        print("config module imported")

        from utils.logging import setup_logger
        print("utils.logging imported")

        from utils.verification import verify_recent_backup
        print("utils.verification imported")

        print("All modules imported successfully")
        return True

    except Exception as e:
        print(f"Import test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("=== Playwright YouTube Remover Setup Test ===\n")
    
    # Test imports first
    import_success = test_imports()
    print()
    
    # Test Playwright functionality
    playwright_success = test_playwright_setup()
    
    print("\n=== Test Summary ===")
    if import_success and playwright_success:
        print("All tests passed! The setup is ready for use.")
        sys.exit(0)
    else:
        print("Some tests failed. Please check the setup.")
        sys.exit(1)

if __name__ == "__main__":
    main()
