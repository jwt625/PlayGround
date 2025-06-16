import asyncio
import os
import time
from pathlib import Path
from playwright.async_api import async_playwright

class GoogleDrivePlaywright:
    def __init__(self, download_dir=None):
        self.download_dir = download_dir or os.path.join(os.getcwd(), 'playwright_downloads')
        Path(self.download_dir).mkdir(exist_ok=True)
        self.page = None
        self.context = None
        self.browser = None
    
    async def setup_browser(self, headless=False):
        """Initialize Playwright browser with download settings"""
        playwright = await async_playwright().start()
        
        # Launch browser - Chrome is less detectable than Chromium
        self.browser = await playwright.chromium.launch(
            headless=headless,
            args=[
                '--no-first-run',
                '--disable-blink-features=AutomationControlled',
                '--disable-web-security',
                '--disable-features=VizDisplayCompositor'
            ]
        )
        
        # Create context with download path
        self.context = await self.browser.new_context(
            accept_downloads=True,
            viewport={'width': 1920, 'height': 1080},
            user_agent='Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        )
        
        # Create page
        self.page = await self.context.new_page()
        
        # Inject script to remove automation markers
        await self.page.add_init_script("""
            Object.defineProperty(navigator, 'webdriver', {
                get: () => undefined,
            });
            
            window.chrome = {
                runtime: {},
            };
            
            Object.defineProperty(navigator, 'plugins', {
                get: () => [1, 2, 3, 4, 5],
            });
        """)
    
    async def login_to_drive(self):
        """Navigate to Google Drive and handle login"""
        print("🚀 Opening Google Drive...")
        await self.page.goto('https://drive.google.com', wait_until='networkidle')
        
        # Wait for user to manually authenticate
        print("📝 Please log in to Google Drive in the browser window...")
        print("✅ Press Enter here when you're logged in and can see your Drive files...")
        
        # Keep browser alive while user logs in
        await asyncio.sleep(1)
        input()  # Wait for user confirmation
        
        # Verify we're on Drive
        try:
            await self.page.wait_for_selector('[data-target="drive"]', timeout=10000)
            print("✅ Successfully logged in to Google Drive!")
        except:
            print("⚠️  Could not confirm Google Drive login. Continuing anyway...")
    
    async def wait_random(self, min_sec=1, max_sec=3):
        """Add random delay to mimic human behavior"""
        import random
        delay = random.uniform(min_sec, max_sec)
        await asyncio.sleep(delay)
    
    async def select_all_files(self):
        """Select all files using keyboard shortcut"""
        print("📁 Selecting all files...")
        await self.page.keyboard.press('Control+a')
        await self.wait_random(1, 2)
        print("✅ All files selected")
    
    async def download_selected(self):
        """Download selected files"""
        print("⬇️  Initiating download...")
        
        # Method 1: Try keyboard shortcut
        try:
            await self.page.keyboard.press('Control+Shift+s')
            await self.wait_random(2, 4)
            print("✅ Download started via keyboard shortcut")
            return True
        except Exception as e:
            print(f"⚠️  Keyboard shortcut failed: {e}")
        
        # Method 2: Try right-click menu
        try:
            # Right-click on selected area
            files_container = await self.page.query_selector('[data-target="drive"]')
            if files_container:
                await files_container.click(button='right')
                await self.wait_random(0.5, 1)
                
                # Look for download option
                download_option = await self.page.query_selector('text="Download"')
                if download_option:
                    await download_option.click()
                    await self.wait_random(2, 4)
                    print("✅ Download started via context menu")
                    return True
        except Exception as e:
            print(f"⚠️  Context menu failed: {e}")
        
        print("❌ Could not initiate download")
        return False
    
    async def navigate_to_folder(self, folder_name):
        """Navigate to a specific folder"""
        print(f"📂 Looking for folder: {folder_name}")
        
        try:
            # Look for folder by name
            folder_selector = f'[data-tooltip="{folder_name}"]'
            folder_element = await self.page.wait_for_selector(folder_selector, timeout=5000)
            
            if folder_element:
                await folder_element.dblclick()
                await self.wait_random(2, 3)
                print(f"✅ Navigated to folder: {folder_name}")
                return True
        except:
            # Try alternative selector
            try:
                folder_element = await self.page.query_selector(f'text="{folder_name}"')
                if folder_element:
                    await folder_element.dblclick()
                    await self.wait_random(2, 3)
                    print(f"✅ Navigated to folder: {folder_name}")
                    return True
            except Exception as e:
                print(f"❌ Could not find folder '{folder_name}': {e}")
        
        return False
    
    async def go_back(self):
        """Go back to parent folder"""
        try:
            # Try back button
            back_button = await self.page.query_selector('[data-tooltip="Back"]')
            if back_button:
                await back_button.click()
                await self.wait_random(1, 2)
                print("⬅️  Went back to parent folder")
                return True
            
            # Alternative: use keyboard
            await self.page.keyboard.press('Alt+ArrowLeft')
            await self.wait_random(1, 2)
            print("⬅️  Went back using keyboard")
            return True
        except Exception as e:
            print(f"❌ Could not go back: {e}")
            return False
    
    async def download_folder_contents(self, folder_names=None):
        """Download contents of specified folders or all files"""
        if not folder_names:
            # Download everything in current view
            print("📥 Downloading all files in current view...")
            await self.select_all_files()
            await self.download_selected()
        else:
            # Download specific folders
            for folder_name in folder_names:
                print(f"\n🎯 Processing folder: {folder_name}")
                
                if await self.navigate_to_folder(folder_name):
                    await self.select_all_files()
                    await self.download_selected()
                    await self.go_back()
                else:
                    print(f"⚠️  Skipping folder: {folder_name}")
                
                await self.wait_random(2, 4)  # Pause between folders
    
    async def handle_downloads(self):
        """Monitor and handle downloads"""
        print("📋 Monitoring downloads...")
        
        # Set up download handler
        async def handle_download(download):
            filename = download.suggested_filename
            download_path = os.path.join(self.download_dir, filename)
            await download.save_as(download_path)
            print(f"💾 Downloaded: {filename}")
        
        self.page.on("download", handle_download)
    
    async def close(self):
        """Clean up browser resources"""
        if self.context:
            await self.context.close()
        if self.browser:
            await self.browser.close()

async def main():
    downloader = GoogleDrivePlaywright(download_dir="./playwright_downloads")
    
    try:
        # Setup browser (set headless=True to run in background)
        await downloader.setup_browser(headless=False)
        
        # Setup download monitoring
        await downloader.handle_downloads()
        
        # Login to Google Drive
        await downloader.login_to_drive()
        
        # Option 1: Download everything
        # await downloader.download_folder_contents()
        
        # Option 2: Download specific folders
        folders_to_download = ["Documents", "Photos", "Projects"]  # Customize as needed
        await downloader.download_folder_contents(folders_to_download)
        
        print("\n🎉 Download process completed!")
        print("⏳ Waiting 30 seconds for downloads to finish...")
        await asyncio.sleep(30)
        
    except Exception as e:
        print(f"❌ An error occurred: {e}")
    finally:
        input("Press Enter to close browser...")
        await downloader.close()

if __name__ == "__main__":
    asyncio.run(main())