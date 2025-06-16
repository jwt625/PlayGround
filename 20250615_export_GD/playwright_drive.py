import asyncio
import os
import time
from pathlib import Path
from playwright.async_api import async_playwright
import random
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class GoogleDrivePlaywright:
    def __init__(self, download_dir=None):
        self.download_dir = download_dir or os.path.join(os.getcwd(), 'playwright_downloads')
        Path(self.download_dir).mkdir(exist_ok=True)
        self.page = None
        self.context = None
        self.browser = None
        self.current_folder_path = ""
        self.download_count = 0
        self.rate_limit_delay = 2  # Base delay between downloads
    
    def get_chrome_profile_path(self):
        """Find the best Chrome profile to use"""
        base_path = "/Users/wentaojiang/Library/Application Support/Google/Chrome"
        profiles = ["Default", "Profile 1", "Profile 3"]
        
        # Get email addresses from environment
        profile_emails = {
            "Default": os.getenv("PROFILE_DEFAULT_EMAIL", "Unknown"),
            "Profile 1": os.getenv("PROFILE_1_EMAIL", "Unknown"),
            "Profile 3": os.getenv("PROFILE_3_EMAIL", "Unknown")
        }
        
        print("🔍 Checking Chrome profiles for Google Drive access...")
        for profile in profiles:
            profile_path = f"{base_path}/{profile}"
            if os.path.exists(profile_path):
                print(f"✅ Found profile: {profile}")
        
        # Get default choice from environment
        default_choice = os.getenv("DEFAULT_PROFILE_CHOICE", "2")
        
        # Let user choose or use default
        print("\n📋 Available profiles:")
        print(f"1. Default ({profile_emails['Default']})")
        print(f"2. Profile 1 ({profile_emails['Profile 1']})") 
        print(f"3. Profile 3 ({profile_emails['Profile 3']})")
        choice = input(f"Which profile has your Google Drive access? (1-3, default={default_choice}): ").strip()
        
        if not choice:
            choice = default_choice
        
        profile_map = {"1": "Default", "2": "Profile 1", "3": "Profile 3"}
        selected_profile = profile_map.get(choice, "Profile 1")
        
        print(f"🎯 Using profile: {selected_profile} ({profile_emails[selected_profile]})")
        return f"{base_path}/{selected_profile}"
    
    async def setup_browser(self, headless=False):
        """Initialize Playwright browser with maximum stealth and persistent context"""
        playwright = await async_playwright().start()
        
        # Get Chrome profile path
        profile_path = self.get_chrome_profile_path()
        print(f"🚀 Using Chrome profile: {profile_path}")
        
        # Check if Chrome is running and warn user
        import subprocess
        chrome_processes = subprocess.run(['pgrep', '-f', 'Google Chrome'], capture_output=True, text=True)
        if chrome_processes.stdout.strip():
            print("⚠️  WARNING: Chrome is currently running!")
            print("📌 For best results with persistent context:")
            print("   1. Close ALL Chrome windows")
            print("   2. Wait a few seconds")
            print("   3. Run this script again")
            response = input("Continue anyway? (y/N): ").strip().lower()
            if response != 'y':
                print("❌ Exiting. Please close Chrome and try again.")
                return None
        
        # Use persistent context with your existing Chrome profile
        self.context = await playwright.chromium.launch_persistent_context(
            user_data_dir=profile_path,
            headless=headless,
            accept_downloads=True,
            viewport={'width': 1920, 'height': 1080},
            
            # Maximum stealth arguments
            args=[
                '--no-first-run',
                '--disable-blink-features=AutomationControlled',
                '--disable-web-security',
                '--disable-features=VizDisplayCompositor',
                '--disable-background-timer-throttling',
                '--disable-backgrounding-occluded-windows',
                '--disable-renderer-backgrounding',
                '--disable-field-trial-config',
                '--disable-ipc-flooding-protection',
                '--disable-hang-monitor',
                '--disable-prompt-on-repost',
                '--disable-sync',
                '--disable-domain-reliability',
                '--disable-background-networking',
                '--disable-default-apps',
                '--disable-extensions-except',
                '--disable-component-extensions-with-background-pages',
                '--disable-dev-shm-usage',
                '--no-sandbox',
                '--disable-setuid-sandbox',
                '--disable-gpu-sandbox',
                '--disable-software-rasterizer',
                '--disable-background-timer-throttling',
                '--disable-backgrounding-occluded-windows',
                '--disable-renderer-backgrounding',
                '--disable-features=TranslateUI',
                '--disable-client-side-phishing-detection',
                '--disable-component-update',
                '--disable-permissions-api',
                '--disable-plugins-discovery',
                '--disable-preconnect',
                '--disable-print-preview'
            ],
            
            # Realistic user agent
            user_agent='Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36'
        )
        
        # Get the page from persistent context
        pages = self.context.pages
        if pages:
            self.page = pages[0]
        else:
            self.page = await self.context.new_page()
        
        # Inject maximum stealth scripts
        await self.page.add_init_script("""
            // Remove webdriver property
            Object.defineProperty(navigator, 'webdriver', {
                get: () => undefined,
            });
            
            // Mock chrome object
            window.chrome = {
                runtime: {},
                loadTimes: function() {
                    return {
                        requestTime: Date.now() * 0.001,
                        startLoadTime: Date.now() * 0.001,
                        commitLoadTime: Date.now() * 0.001,
                        finishDocumentLoadTime: Date.now() * 0.001,
                        finishLoadTime: Date.now() * 0.001,
                        firstPaintTime: Date.now() * 0.001,
                        firstPaintAfterLoadTime: 0,
                        navigationType: 'navigate',
                        wasFetchedViaSpdy: false,
                        wasNpnNegotiated: false,
                        npnNegotiatedProtocol: 'unknown',
                        wasAlternateProtocolAvailable: false,
                        connectionInfo: 'http/1.1'
                    };
                },
                csi: function() {
                    return {
                        onloadT: Date.now(),
                        startE: Date.now(),
                        tran: 15
                    };
                }
            };
            
            // Mock plugins
            Object.defineProperty(navigator, 'plugins', {
                get: () => [
                    {
                        0: {type: "application/x-google-chrome-pdf", suffixes: "pdf", description: "Portable Document Format", enabledPlugin: Plugin},
                        description: "Portable Document Format",
                        filename: "internal-pdf-viewer",
                        length: 1,
                        name: "Chrome PDF Plugin"
                    },
                    {
                        0: {type: "application/pdf", suffixes: "pdf", description: "", enabledPlugin: Plugin},
                        description: "",
                        filename: "mhjfbmdgcfjbbpaeojofohoefgiehjai",
                        length: 1,
                        name: "Chrome PDF Viewer"
                    }
                ]
            });
            
            // Mock languages
            Object.defineProperty(navigator, 'languages', {
                get: () => ['en-US', 'en']
            });
            
            // Mock permissions
            const originalQuery = window.navigator.permissions.query;
            window.navigator.permissions.query = (parameters) => (
                parameters.name === 'notifications' ?
                    Promise.resolve({ state: Notification.permission }) :
                    originalQuery(parameters)
            );
            
            // Mock media devices
            Object.defineProperty(navigator, 'mediaDevices', {
                get: () => ({
                    enumerateDevices: () => Promise.resolve([])
                })
            });
            
            // Remove automation traces
            delete window.cdc_adoQpoasnfa76pfcZLmcfl_Array;
            delete window.cdc_adoQpoasnfa76pfcZLmcfl_Promise;
            delete window.cdc_adoQpoasnfa76pfcZLmcfl_Symbol;
        """)
    
    async def login_to_drive(self):
        """Navigate to Google Drive and handle login"""
        print("🚀 Opening Google Drive...")
        await self.page.goto('https://drive.google.com', wait_until='networkidle')
        
        # Wait for user to manually authenticate
        print("📝 Please log in to Google Drive in the browser window...")
        print("🗂️  Navigate to the folder you want to download...")
        print("✅ Press Enter here when you're ready to start downloading...")
        
        # Keep browser alive while user logs in and navigates
        await asyncio.sleep(1)
        input()  # Wait for user confirmation
        
        # Verify we're on Drive
        try:
            await self.page.wait_for_selector('[data-target="drive"]', timeout=10000)
            print("✅ Ready to start downloading!")
        except:
            print("⚠️  Could not confirm Google Drive login. Continuing anyway...")
    
    async def wait_random(self, min_sec=1, max_sec=3):
        """Add random delay to mimic human behavior"""
        delay = random.uniform(min_sec, max_sec)
        await asyncio.sleep(delay)
    
    async def apply_rate_limit(self):
        """Apply progressive rate limiting to avoid detection"""
        # Increase delay after every 10 downloads
        progressive_delay = self.rate_limit_delay + (self.download_count // 10) * 0.5
        delay = random.uniform(progressive_delay, progressive_delay + 1)
        print(f"⏱️  Rate limit delay: {delay:.1f}s (downloaded: {self.download_count})")
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
    
    async def get_current_folder_items(self):
        """Get list of files and folders in current directory"""
        await self.wait_random(1, 2)
        
        # Get all file/folder items
        items = []
        try:
            # Wait for grid to load
            await self.page.wait_for_selector('[data-target="drive"]', timeout=10000)
            
            # Get file elements - this selector might need adjustment based on actual Drive UI
            file_elements = await self.page.query_selector_all('[data-tooltip]:not([data-tooltip=""])')
            
            for element in file_elements:
                try:
                    name = await element.get_attribute('data-tooltip')
                    # Check if it's a folder by looking for folder icon or other indicators
                    parent = await element.query_selector('..')  # Get parent element
                    is_folder = False
                    if parent:
                        folder_icon = await parent.query_selector('[data-target="folder"]')
                        is_folder = folder_icon is not None
                    
                    if name and name not in ['Back', 'Search', 'New']:  # Filter out UI elements
                        items.append({'name': name, 'is_folder': is_folder, 'element': element})
                except:
                    continue
                    
        except Exception as e:
            print(f"⚠️  Error getting folder items: {e}")
        
        return items
    
    async def download_individual_file(self, file_element, file_name):
        """Download a single file"""
        try:
            print(f"📄 Downloading file: {file_name}")
            
            # Click on the file to select it
            await file_element.click()
            await self.wait_random(0.5, 1)
            
            # Try download shortcut
            await self.page.keyboard.press('Control+Shift+s')
            await self.wait_random(1, 2)
            
            # Apply rate limiting
            await self.apply_rate_limit()
            self.download_count += 1
            
            print(f"✅ Initiated download: {file_name}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to download {file_name}: {e}")
            return False
    
    async def download_folder_recursively(self, base_path=""):
        """Recursively download all files in current folder and subfolders"""
        print(f"🗂️  Processing folder: {base_path or 'Root'}")
        
        # Get all items in current folder
        items = await self.get_current_folder_items()
        
        if not items:
            print("📭 No items found in this folder")
            return
        
        print(f"📊 Found {len(items)} items")
        
        # First, download all files (non-folders)
        files = [item for item in items if not item['is_folder']]
        folders = [item for item in items if item['is_folder']]
        
        print(f"📄 Files to download: {len(files)}")
        print(f"📁 Folders to process: {len(folders)}")
        
        # Download files
        for file_item in files:
            await self.download_individual_file(file_item['element'], file_item['name'])
        
        # Process subfolders recursively
        for folder_item in folders:
            folder_name = folder_item['name']
            print(f"\n🔄 Entering folder: {folder_name}")
            
            # Double-click to enter folder
            await folder_item['element'].dblclick()
            await self.wait_random(2, 3)
            
            # Recursively process subfolder
            new_path = f"{base_path}/{folder_name}" if base_path else folder_name
            await self.download_folder_recursively(new_path)
            
            # Go back to parent folder
            print(f"⬅️  Returning from: {folder_name}")
            await self.go_back()
            await self.wait_random(1, 2)
    
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
        
        # Login to Google Drive and let user navigate to desired folder
        await downloader.login_to_drive()
        
        # Start recursive download from current folder
        print("\n🚀 Starting recursive download...")
        await downloader.download_folder_recursively()
        
        print(f"\n🎉 Download process completed!")
        print(f"📊 Total files downloaded: {downloader.download_count}")
        print("⏳ Waiting 30 seconds for downloads to finish...")
        await asyncio.sleep(30)
        
    except Exception as e:
        print(f"❌ An error occurred: {e}")
        import traceback
        traceback.print_exc()
    finally:
        input("Press Enter to close browser...")
        await downloader.close()

if __name__ == "__main__":
    asyncio.run(main())