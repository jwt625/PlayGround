import asyncio
import os
import time
from pathlib import Path
from playwright.async_api import async_playwright
import random
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class GoogleDriveFirefox:
    def __init__(self, download_dir=None):
        self.download_dir = download_dir or os.path.join(os.getcwd(), 'firefox_downloads')
        Path(self.download_dir).mkdir(exist_ok=True)
        self.page = None
        self.context = None
        self.browser = None
        self.current_folder_path = ""
        self.download_count = 0
        self.rate_limit_delay = 2  # Base delay between downloads
    
    def get_firefox_profile_path(self):
        """Find Firefox profile path"""
        home = os.path.expanduser("~")
        firefox_path = f"{home}/Library/Application Support/Firefox/Profiles"
        
        print("🔍 Looking for Firefox profiles...")
        
        if not os.path.exists(firefox_path):
            print("❌ Firefox not found. Using fresh profile.")
            return None
        
        # Find default profile
        try:
            profiles = [d for d in os.listdir(firefox_path) if os.path.isdir(os.path.join(firefox_path, d))]
            if profiles:
                # Use the first profile (usually default)
                default_profile = profiles[0]
                profile_path = os.path.join(firefox_path, default_profile)
                print(f"✅ Found Firefox profile: {default_profile}")
                return profile_path
        except Exception as e:
            print(f"⚠️  Error finding Firefox profile: {e}")
        
        return None
    
    async def setup_browser(self, headless=False):
        """Initialize Firefox browser with maximum stealth"""
        playwright = await async_playwright().start()
        
        # Get Firefox profile path
        profile_path = self.get_firefox_profile_path()
        
        if profile_path:
            print(f"🦊 Using Firefox profile: {profile_path}")
            
            # Use persistent context with existing Firefox profile
            self.context = await playwright.firefox.launch_persistent_context(
                user_data_dir=profile_path,
                headless=headless,
                accept_downloads=True,
                viewport={'width': 1920, 'height': 1080},
                
                # Firefox args for stealth
                args=[
                    '--disable-blink-features=AutomationControlled',
                    '--disable-web-security',
                    '--disable-features=VizDisplayCompositor',
                    '--no-first-run',
                    '--disable-backgrounding-occluded-windows',
                    '--disable-renderer-backgrounding',
                ],
                
                # Firefox user agent
                user_agent='Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:122.0) Gecko/20100101 Firefox/122.0'
            )
            
            # Get page from persistent context - find active Drive tab
            pages = self.context.pages
            drive_page = None
            
            # Look for existing Drive tab
            for page in pages:
                try:
                    url = page.url
                    if 'drive.google.com' in url:
                        drive_page = page
                        print(f"✅ Found existing Drive tab: {url}")
                        break
                except:
                    continue
            
            if drive_page:
                self.page = drive_page
            elif pages:
                self.page = pages[0]
            else:
                self.page = await self.context.new_page()
        else:
            # Launch fresh Firefox browser
            print("🦊 Launching fresh Firefox browser...")
            self.browser = await playwright.firefox.launch(
                headless=headless,
                args=[
                    '--disable-blink-features=AutomationControlled',
                    '--disable-web-security',
                    '--no-first-run',
                ]
            )
            
            # Create new context
            self.context = await self.browser.new_context(
                accept_downloads=True,
                viewport={'width': 1920, 'height': 1080},
                user_agent='Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:122.0) Gecko/20100101 Firefox/122.0'
            )
            
            self.page = await self.context.new_page()
        
        # Firefox-specific stealth injection
        await self.page.add_init_script("""
            // Remove webdriver property
            Object.defineProperty(navigator, 'webdriver', {
                get: () => undefined,
            });
            
            // Mock plugins for Firefox
            Object.defineProperty(navigator, 'plugins', {
                get: () => [
                    {
                        description: "Portable Document Format",
                        filename: "libpdf.so",
                        length: 1,
                        name: "PDF.js"
                    }
                ]
            });
            
            // Mock languages
            Object.defineProperty(navigator, 'languages', {
                get: () => ['en-US', 'en']
            });
            
            // Remove automation traces
            delete window.cdc_adoQpoasnfa76pfcZLmcfl_Array;
            delete window.cdc_adoQpoasnfa76pfcZLmcfl_Promise;
            delete window.cdc_adoQpoasnfa76pfcZLmcfl_Symbol;
            
            // Override permissions API
            if (navigator.permissions && navigator.permissions.query) {
                const originalQuery = navigator.permissions.query;
                navigator.permissions.query = (parameters) => (
                    parameters.name === 'notifications' ?
                        Promise.resolve({ state: 'default' }) :
                        originalQuery(parameters)
                );
            }
        """)
    
    async def login_to_drive(self):
        """Navigate to Google Drive and handle login"""
        # Check if already on Drive
        current_url = self.page.url
        if 'drive.google.com' not in current_url:
            print("🚀 Navigating to Google Drive...")
            await self.page.goto('https://drive.google.com', wait_until='networkidle')
        else:
            print(f"✅ Already on Drive: {current_url}")
        
        # Wait for user to manually authenticate and navigate
        print("📝 Make sure you're logged in to Google Drive...")
        print("🗂️  Navigate to the folder you want to download...")
        print("✅ Press Enter here when you're ready to start downloading...")
        
        # Keep browser alive while user logs in and navigates
        await asyncio.sleep(1)
        input()  # Wait for user confirmation
        
        # Switch to the active tab (in case user switched tabs)
        await self.page.bring_to_front()
        
        # Verify we're on Drive
        try:
            # More flexible Drive detection
            await self.page.wait_for_selector('div[role="main"], [data-target="drive"], [aria-label*="Drive"]', timeout=5000)
            print("✅ Ready to start downloading!")
        except:
            print("⚠️  Could not confirm Google Drive interface. Continuing anyway...")
    
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
    
    async def get_current_folder_items(self):
        """Get list of files and folders in current directory"""
        await self.wait_random(1, 2)
        
        items = []
        try:
            # Wait for the main content area to load
            await self.page.wait_for_selector('div[role="main"]', timeout=10000)
            print("✅ Found main Drive interface")
            
            # Target the specific file grid area, not the entire page
            content_selectors = [
                # The main file grid container
                'div[role="grid"] div[role="gridcell"]',
                # Files area with specific data attributes 
                'div[data-id][role="gridcell"]',
                # Alternative grid cells in main content
                'div[role="main"] div[role="gridcell"]'
            ]
            
            file_elements = []
            for selector in content_selectors:
                file_elements = await self.page.query_selector_all(selector)
                if file_elements:
                    print(f"✅ Found {len(file_elements)} grid items with: {selector}")
                    break
            
            if not file_elements:
                print("⚠️  No grid items found, trying fallback...")
                return items
            
            for i, element in enumerate(file_elements):
                try:
                    # Debug info for each element
                    data_id = await element.get_attribute('data-id')
                    aria_label = await element.get_attribute('aria-label')
                    data_tooltip = await element.get_attribute('data-tooltip')
                    text_content = await element.inner_text()
                    
                    print(f"\n🔍 DEBUG Element {i+1}:")
                    print(f"   data-id: {data_id}")
                    print(f"   aria-label: {aria_label}")
                    print(f"   data-tooltip: {data_tooltip}")
                    print(f"   text_content: {repr(text_content[:100])}")
                    
                    # Try multiple ways to get the file/folder name
                    name = None
                    is_folder = False
                    
                    # Method 1: Look for the filename in aria-label
                    if aria_label:
                        # Parse aria-label format: "filename, Type, Owner, Date"
                        parts = aria_label.split(',')
                        if len(parts) > 0:
                            name = parts[0].strip()
                            # Check if it's a folder from the aria-label
                            if 'folder' in aria_label.lower() or 'google drive folder' in aria_label.lower():
                                is_folder = True
                    
                    # Method 2: Look for data-tooltip as backup
                    if not name and data_tooltip:
                        name = data_tooltip.strip()
                    
                    # Method 3: Look for text content as last resort
                    if not name and text_content:
                        lines = text_content.strip().split('\n')
                        name = lines[0] if lines else None
                        if name:
                            name = name.strip()
                    
                    print(f"   extracted_name: {repr(name)}")
                    print(f"   is_folder: {is_folder}")
                    
                    if not name or len(name.strip()) == 0:
                        print("   ❌ SKIPPED: No name found")
                        continue
                    
                    name = name.strip()
                    
                    # Additional folder detection from element content
                    if not is_folder:
                        element_html = await element.inner_html()
                        folder_indicators = ['folder-icon', 'folder', 'directory']
                        for indicator in folder_indicators:
                            if indicator in element_html.lower():
                                is_folder = True
                                print(f"   🗂️  Found folder indicator: {indicator}")
                                break
                    
                    # Filter out obvious UI elements (less aggressive)
                    ui_keywords = [
                        'advanced search', 'settings', 'support', 'activity',
                        'list layout', 'grid layout', 'hide details',
                        'clear selection', 'more actions', 'reverse sort',
                        'zoom out', 'zoom in', 'edit description'
                    ]
                    
                    # Check if it's a UI element
                    is_ui_element = any(keyword in name.lower() for keyword in ui_keywords)
                    print(f"   is_ui_element: {is_ui_element}")
                    
                    if is_ui_element:
                        print("   ❌ SKIPPED: UI element")
                        continue
                    
                    # Skip if name is too long (likely metadata) but be less restrictive
                    if len(name) > 200:
                        print("   ❌ SKIPPED: Name too long")
                        continue
                    
                    items.append({
                        'name': name, 
                        'is_folder': is_folder, 
                        'element': element,
                        'data_id': data_id
                    })
                    print(f"   ✅ ADDED: {name} {'(folder)' if is_folder else '(file)'}")
                        
                except Exception as e:
                    print(f"   ⚠️  Error processing element {i+1}: {e}")
                    continue
                    
        except Exception as e:
            print(f"⚠️  Error getting folder items: {e}")
        
        print(f"📊 Total valid items found: {len(items)}")
        return items
    
    async def download_individual_file(self, file_element, file_name):
        """Download a single file"""
        try:
            print(f"📄 Downloading file: {file_name}")
            
            # Click on the file to select it
            await file_element.click()
            await self.wait_random(0.5, 1)
            
            # Try Firefox keyboard shortcut for download
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
            await self.page.keyboard.press('Alt+Left')
            await self.wait_random(1, 2)
            print("⬅️  Went back using keyboard")
            return True
        except Exception as e:
            print(f"❌ Could not go back: {e}")
            return False
    
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
    downloader = GoogleDriveFirefox(download_dir="./firefox_downloads")
    
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