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
        self.base_download_dir = download_dir or os.path.join(os.getcwd(), 'firefox_downloads')
        Path(self.base_download_dir).mkdir(exist_ok=True)
        self.page = None
        self.context = None
        self.browser = None
        self.current_folder_path = ""
        self.download_count = 0
        self.rate_limit_delay = 1  # Base delay between downloads (reduced from 2s)
        self.folder_structure = {}  # Track folder hierarchy
        self.current_local_path = ""  # Track current local download path
        self.url_stack = []  # Track folder URLs for navigation
    
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
    
    async def folder_navigation_pause(self):
        """Add random pause between 10s to 5min for folder navigation rate limiting"""
        pause_duration = random.uniform(10, 120)  # 10 seconds to 2 minutes
        print(f"🔄 Folder navigation rate limit: pausing for {pause_duration:.1f} seconds ({pause_duration/60:.1f} minutes)")
        await asyncio.sleep(pause_duration)
    
    async def apply_rate_limit(self):
        """Apply progressive rate limiting to avoid detection"""
        # Increase delay after every 15 downloads (less frequent increases)
        progressive_delay = self.rate_limit_delay + (self.download_count // 15) * 0.3
        delay = random.uniform(progressive_delay, progressive_delay + 0.5)  # Smaller random range
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
    
    async def close_any_open_menus(self):
        """Close any open context menus or popups by pressing Escape key"""
        try:
            # Simple and reliable: just press Escape key to close any open menus
            await self.page.keyboard.press('Escape')
            await self.wait_random(0.3, 0.5)  # Short wait
        except Exception as e:
            print(f"⚠️  Error closing menus: {e}")

    async def handle_download_popup(self):
        """Handle download popups by clicking 'Download anyway' - only popup confirmations"""
        try:
            # Only look for popup-specific buttons, not general download buttons
            popup_selectors = [
                'text="Download anyway"',
                'text="download anyway"',
                'text="Download Anyway"',
                'button:has-text("Download anyway")',
                'button:has-text("download anyway")',
                'text="仍要下载"',  # Chinese "download anyway"
                'text="继续下载"',  # Chinese "continue download"
                'text="Télécharger quand même"'  # French "download anyway"
            ]
            
            for selector in popup_selectors:
                try:
                    popup_button = await self.page.query_selector(selector)
                    if popup_button:
                        print(f"🔄 Found popup confirmation with: {selector}")
                        await popup_button.click(timeout=5000)
                        await self.wait_random(1, 2)
                        print(f"✅ Clicked popup confirmation")
                        return True
                except Exception as e:
                    continue
            
            return False
            
        except Exception as e:
            print(f"⚠️  Error handling download popup: {e}")
            return False

    async def wait_for_download_completion(self, expected_file_name, files_before):
        """Wait for download to complete by monitoring the download folder"""
        download_folder = os.path.join(self.base_download_dir, self.current_local_path) if self.current_local_path else self.base_download_dir
        
        print(f"🔄 Waiting for download completion of: {expected_file_name}")
        print(f"🔄 Monitoring folder: {download_folder}")
        
        # Create download folder if it doesn't exist
        os.makedirs(download_folder, exist_ok=True)
        
        max_wait_time = 300  # Maximum wait time in seconds
        check_interval = 10  # Check every 10 seconds
        elapsed_time = 0
        popup_handled = False
        
        while elapsed_time < max_wait_time:
            try:
                # Check for download popups periodically (but not too often)
                if elapsed_time >= 6 and elapsed_time <= 30 and not popup_handled:  # Check between 6-30 seconds
                    if elapsed_time % 6 == 0:  # Every 6 seconds within this window
                        if await self.handle_download_popup():
                            popup_handled = True
                            print(f"💾 Download popup handled, continuing download wait...")
                
                if os.path.exists(download_folder):
                    current_files = os.listdir(download_folder)
                    files_after = len(current_files)
                    
                    # Check if file count increased
                    if files_after > files_before:
                        print(f"✅ Download completed: File count increased from {files_before} to {files_after}")
                        
                        # Try to identify which file was downloaded
                        if files_after == files_before + 1:
                            # Find the new file
                            for file in current_files:
                                file_path = os.path.join(download_folder, file)
                                # Check if file was created recently (within last 2 minutes)
                                if os.path.getctime(file_path) > (time.time() - 120):
                                    print(f"✅ New file detected: {file}")
                                    break
                        
                        return True
                    
                    # Alternative check: look for file with similar name
                    expected_base = os.path.splitext(expected_file_name)[0].lower()
                    for file in current_files:
                        file_base = os.path.splitext(file)[0].lower()
                        # Check if file name is similar (handles Google Drive renaming)
                        if expected_base in file_base or file_base in expected_base:
                            file_path = os.path.join(download_folder, file)
                            # Check if file was created recently
                            if os.path.getctime(file_path) > (time.time() - 120):
                                print(f"✅ Download completed: Found matching file: {file}")
                                return True
                
                # Wait before next check
                await asyncio.sleep(check_interval)
                elapsed_time += check_interval
                
                if elapsed_time % 10 == 0:  # Progress update every 10 seconds
                    print(f"🔄 Still waiting for download... ({elapsed_time}/{max_wait_time}s)")
                
            except Exception as e:
                print(f"⚠️  Error checking download folder: {e}")
                await asyncio.sleep(check_interval)
                elapsed_time += check_interval
        
        print(f"❌ Download timeout after {max_wait_time}s for: {expected_file_name}")
        return False

    async def download_individual_file(self, file_element, file_name):
        """Download a single file with verification"""
        try:
            print(f"📄 Downloading file: {file_name}")
            
            # First, clear any existing context menus
            await self.close_any_open_menus()
            await self.wait_random(0.2, 0.4)  # Reduced wait time
            
            # Check file count before download for verification
            download_folder = os.path.join(self.base_download_dir, self.current_local_path) if self.current_local_path else self.base_download_dir
            if os.path.exists(download_folder):
                files_before = len(os.listdir(download_folder))
                print(f"🔄 Files in download folder before: {files_before}")
            else:
                files_before = 0
                print(f"🔄 Download folder doesn't exist yet: {download_folder}")
            
            expected_download_path = os.path.join(download_folder, file_name)
            print(f"🔄 Expected download location: {expected_download_path}")
            
            # Right-click on the file to open context menu
            print(f"🔄 Right-clicking on file element...")
            await file_element.click(button='right')
            await self.wait_random(0.8, 1.2)  # Reduced wait time for menu to appear
            
            # Look for download option in context menu (prioritize working selector)
            download_selectors = [
                '[aria-label*="Download"]',  # This one works reliably
                '[data-tooltip*="Download"]',
                'text="Download"',  # Keep as fallback but lower priority
                'text="下载"',  # Chinese
                'text="Télécharger"'  # French
            ]
            
            download_clicked = False
            for selector in download_selectors:
                try:
                    download_option = await self.page.query_selector(selector)
                    if download_option:
                        print(f"🔄 Found download option with: {selector}")
                        # Try clicking with 5 second timeout
                        await download_option.click(timeout=5000)
                        download_clicked = True
                        print(f"✅ Clicked download from context menu")
                        break
                except Exception as e:
                    print(f"⚠️  Download selector {selector} failed: {e}")
                    # Try alternative clicking method if regular click fails
                    try:
                        if download_option:
                            print(f"🔄 Trying force click for: {selector}")
                            await download_option.click(force=True, timeout=5000)
                            download_clicked = True
                            print(f"✅ Force clicked download from context menu")
                            break
                    except Exception as e2:
                        print(f"⚠️  Force click also failed: {e2}")
                        continue
            
            # Always close context menu after attempting download
            await self.close_any_open_menus()
            await self.wait_random(0.2, 0.4)  # Reduced wait time
            
            if not download_clicked:
                # Fallback: try keyboard shortcut
                print("🔄 Trying keyboard shortcut fallback...")
                await file_element.click()  # Select the file first
                await self.wait_random(0.3, 0.5)  # Reduced wait time
                await self.page.keyboard.press('Control+Shift+s')
                await self.wait_random(0.5, 0.8)  # Reduced wait time
                
            print(f"✅ Initiated download: {file_name}")
            
            # Immediately check for any download popup
            await self.wait_random(1, 2)  # Give popup time to appear
            await self.handle_download_popup()
            
            # Wait for download to complete with proper verification
            download_success = await self.wait_for_download_completion(file_name, files_before)
            
            if download_success:
                # Apply rate limiting only after successful download
                await self.apply_rate_limit()
                self.download_count += 1
                return True
            else:
                print(f"⚠️  Download verification failed for: {file_name}")
                return False
            
        except Exception as e:
            print(f"❌ Failed to download {file_name}: {e}")
            # Always ensure menus are closed even on error
            await self.close_any_open_menus()
            return False
    
    def parse_selection(self, selection_input):
        """Parse user selection input like '4, 5, 7~14, 20' into a list of indices"""
        indices = []
        parts = selection_input.split(',')
        
        for part in parts:
            part = part.strip()
            if '~' in part:
                # Handle range like "7~14"
                start, end = part.split('~')
                start_idx = int(start.strip()) - 1  # Convert to 0-based
                end_idx = int(end.strip()) - 1      # Convert to 0-based
                indices.extend(range(start_idx, end_idx + 1))
            else:
                # Handle single number like "4"
                idx = int(part.strip()) - 1  # Convert to 0-based
                indices.append(idx)
        
        return sorted(list(set(indices)))  # Remove duplicates and sort

    async def download_folder_recursively(self, base_path="", is_root=True):
        """Recursively download selected files in current folder and subfolders"""
        print(f"🗂️  Processing folder: {base_path or 'Root'}")
        
        # Update current local path for downloads
        self.current_local_path = base_path
        
        # Get all items in current folder (this will be our baseline)
        items = await self.get_current_folder_items()
        
        if not items:
            print("📭 No items found in this folder")
            return
        
        print(f"📊 Found {len(items)} items")
        
        # Only show selection interface for root folder
        if is_root:
            # Display numbered list of all items
            print("\n" + "="*60)
            print("📋 ITEMS IN CURRENT FOLDER:")
            print("="*60)
            for i, item in enumerate(items, 1):
                item_type = "📁" if item['is_folder'] else "📄"
                print(f"{i:3d}. {item_type} {item['name']}")
            
            print("\n" + "="*60)
            print("📝 SELECT ITEMS TO DOWNLOAD:")
            print("Format: 4, 5, 7~14, 20 (individual numbers, ranges with ~)")
            print("Enter 'all' to download everything, or 'skip' to skip this folder")
            selection = input("Your selection: ").strip()
            
            if selection.lower() == 'skip':
                print("⏭️  Skipping this folder")
                return
            
            selected_indices = []
            if selection.lower() == 'all':
                selected_indices = list(range(len(items)))
            else:
                try:
                    selected_indices = self.parse_selection(selection)
                except (ValueError, IndexError) as e:
                    print(f"❌ Invalid selection format: {e}")
                    print("📝 Please use format like: 4, 5, 7~14, 20")
                    return
            
            # Filter selected items
            selected_items = []
            for idx in selected_indices:
                if 0 <= idx < len(items):
                    selected_items.append(items[idx])
                else:
                    print(f"⚠️  Index {idx+1} is out of range (max: {len(items)})")
            
            if not selected_items:
                print("❌ No valid items selected")
                return
            
            print(f"\n📦 Selected {len(selected_items)} items for download:")
            for item in selected_items:
                item_type = "📁" if item['is_folder'] else "📄"
                print(f"   {item_type} {item['name']}")
        else:
            # For non-root folders, download everything
            selected_items = items
            print(f"📦 Downloading all {len(selected_items)} items in subfolder")
        
        # Separate selected files and folders
        files = [item for item in selected_items if not item['is_folder']]
        folders = [item for item in selected_items if item['is_folder']]
        
        print(f"\n📄 Files to download: {len(files)}")
        print(f"📁 Folders to process: {len(folders)}")
        
        # Download selected files
        for i, file_item in enumerate(files):
            print(f"\n📥 Downloading file {i+1}/{len(files)}: {file_item['name']}")
            
            # Ensure clean state before each download
            await self.close_any_open_menus()
            
            # Download the file with built-in verification
            download_success = await self.download_individual_file(file_item['element'], file_item['name'])
            
            if download_success:
                print(f"✅ Successfully downloaded: {file_item['name']}")
            else:
                print(f"❌ Failed to download: {file_item['name']}")
            
            # Small delay and thorough cleanup between files
            await self.wait_random(0.3, 0.5)  # Reduced wait time between files
            await self.close_any_open_menus()
        
        # Remember current folder structure and URL before entering subfolders
        current_folder_items = [item['name'] for item in items]
        current_folder_url = self.page.url
        print(f"🔄 Current folder URL: {current_folder_url}")
        
        # Process subfolders recursively
        for i, folder_item in enumerate(folders):
            folder_name = folder_item['name']
            print(f"\n🔄 Entering folder {i+1}/{len(folders)}: {folder_name}")
            
            navigation_successful = False
            
            try:
                # Save current URL before navigation
                current_url = self.page.url
                print(f"🔄 Current URL before navigation: {current_url}")
                
                # Check if element is still attached
                try:
                    element_text = await folder_item['element'].inner_text()
                    print(f"🔄 Element text: {element_text[:50]}...")
                    print(f"🔄 Element attached: ✅")
                except Exception as e:
                    print(f"❌ Element detached: {e}")
                    print("🔄 Need to refresh folder items...")
                    fresh_items = await self.get_current_folder_items()
                    fresh_folders = [item for item in fresh_items if item['is_folder']]
                    # Find the current folder in fresh items
                    folder_item = None
                    for fresh_folder in fresh_folders:
                        if fresh_folder['name'] == folder_name:
                            folder_item = fresh_folder
                            print(f"✅ Found fresh element for: {folder_name}")
                            break
                    
                    if not folder_item:
                        print(f"❌ Could not find fresh element for: {folder_name}")
                        continue
                
                # Navigate to folder by clicking on the name/element
                print(f"🔄 Double-clicking on folder element...")
                await folder_item['element'].dblclick()
                print(f"🔄 Double-click completed, waiting...")
                await self.wait_random(3, 4)  # Give more time for navigation
                
                # Simple cleanup of any stuck context menus after folder navigation
                await self.close_any_open_menus()
                
                # Check if URL changed (most reliable way to detect navigation)
                new_url = self.page.url
                print(f"🔄 URL after navigation: {new_url}")
                print(f"🔄 URL comparison: {current_url} -> {new_url}")
                print(f"🔄 URLs are {'different' if new_url != current_url else 'SAME'}")
                
                if new_url != current_url:
                    print(f"✅ Navigation successful - URL changed")
                    print(f"   From: {current_url}")
                    print(f"   To:   {new_url}")
                    # Push current URL to stack for going back
                    self.url_stack.append(current_url)
                    print(f"🔄 URL stack now has {len(self.url_stack)} items")
                    navigation_successful = True
                else:
                    print(f"❌ Navigation failed - URL unchanged")
                    print(f"🔄 Waiting additional time and checking again...")
                    await self.wait_random(2, 3)
                    new_url = self.page.url
                    print(f"🔄 URL after additional wait: {new_url}")
                    if new_url != current_url:
                        print(f"✅ Navigation successful after delay - URL changed")
                        self.url_stack.append(current_url)
                        print(f"🔄 URL stack now has {len(self.url_stack)} items")
                        navigation_successful = True
                    else:
                        print(f"❌ Navigation definitely failed - URL still unchanged")
                        print(f"   Expected change from: {current_url}")
                        print(f"   But still at:        {new_url}")
                        navigation_successful = False
                
                if navigation_successful:
                    # Add rate limiting pause after successful folder navigation
                    await self.folder_navigation_pause()
                    
                    # Recursively process subfolder with updated path
                    new_path = os.path.join(base_path, folder_name) if base_path else folder_name
                    
                    # Create local folder structure to mirror Google Drive
                    local_folder_path = os.path.join(self.base_download_dir, new_path)
                    os.makedirs(local_folder_path, exist_ok=True)
                    print(f"📁 Created local folder: {local_folder_path}")
                    
                    await self.download_folder_recursively(new_path, is_root=False)
                
            except Exception as e:
                print(f"❌ Failed to enter folder {folder_name}: {e}")
                
            # If navigation failed, try alternative method
            if not navigation_successful:
                print("🔄 Trying alternative navigation method...")
                try:
                    # Refresh the page elements since they might be stale
                    fresh_items = await self.get_current_folder_items()
                    fresh_folder = None
                    for item in fresh_items:
                        if item['name'] == folder_name and item['is_folder']:
                            fresh_folder = item
                            break
                    
                    if fresh_folder:
                        await fresh_folder['element'].click()
                        await self.wait_random(1, 2)
                        await fresh_folder['element'].click()  # Double-click manually
                        await self.wait_random(3, 4)
                        
                        # Verify alternative method worked by checking URL
                        await self.wait_random(2, 3)
                        alt_url = self.page.url
                        print(f"🔄 Alternative method URL: {alt_url}")
                        
                        if alt_url != current_url:
                            print(f"✅ Alternative navigation successful - URL changed")
                            # Don't forget to save the URL for going back
                            if len(self.url_stack) == 0 or self.url_stack[-1] != alt_url:
                                self.url_stack.append(current_url)
                            navigation_successful = True
                            new_path = os.path.join(base_path, folder_name) if base_path else folder_name
                            
                            # Create local folder structure
                            local_folder_path = os.path.join(self.base_download_dir, new_path)
                            os.makedirs(local_folder_path, exist_ok=True)
                            print(f"📁 Created local folder: {local_folder_path}")
                            
                            await self.download_folder_recursively(new_path, is_root=False)
                        else:
                            print(f"❌ Alternative method also failed - URL unchanged")
                    else:
                        print(f"❌ Could not find folder {folder_name} after refresh")
                        continue
                        
                except Exception as e2:
                    print(f"❌ Alternative navigation also failed: {e2}")
                    continue
            
            # Go back to parent folder only if we successfully navigated
            if navigation_successful:
                print(f"⬅️  Returning from: {folder_name}")
                if not await self.go_back():
                    print("❌ Could not go back, stopping recursion")
                    break
                
                # Add rate limiting pause after returning from subfolder
                await self.folder_navigation_pause()
                
                # Wait for page to load after going back
                await self.wait_random(2, 3)
                
                # Verify we're back by checking URL
                back_url = self.page.url
                print(f"🔄 Back navigation URL: {back_url}")
                
                if back_url == current_folder_url:
                    print(f"✅ Successfully returned to parent folder")
                else:
                    print(f"⚠️  Back navigation URL different from expected")
                    print(f"   Expected: {current_folder_url}")
                    print(f"   Got: {back_url}")
                
                # Reset local path back to current level
                self.current_local_path = base_path
                
                # IMPORTANT: Refresh folder items after going back to fix element detachment
                print("🔄 Refreshing folder items after going back...")
                items = await self.get_current_folder_items()
                folders = [item for item in items if item['is_folder']]
    
    async def go_back(self):
        """Go back to parent folder using URL stack"""
        print("🔄 Attempting to go back...")
        
        current_url = self.page.url
        print(f"🔄 Current URL: {current_url}")
        
        # Check if we have a URL to go back to
        if not self.url_stack:
            print("❌ No parent URL in stack - cannot go back")
            return False
        
        target_url = self.url_stack[-1]  # Get the last URL (parent folder)
        print(f"🎯 Target URL: {target_url}")
        
        # Method 1: Direct navigation to parent URL with multiple wait strategies
        try:
            print("🔄 Trying direct navigation to parent URL...")
            # Try with domcontentloaded first (faster)
            await self.page.goto(target_url, wait_until='domcontentloaded', timeout=8000)
            await self.wait_random(2, 3)
            
            final_url = self.page.url
            if final_url == target_url or target_url in final_url:
                print("⬅️  Went back using direct navigation (domcontentloaded)")
                self.url_stack.pop()  # Remove the URL we just went back to
                return True
            else:
                print(f"🔄 URL reached but different: {final_url}, trying networkidle...")
                # Try waiting for networkidle if we're close
                try:
                    await self.page.wait_for_load_state('networkidle', timeout=5000)
                    final_url = self.page.url
                    if final_url == target_url or target_url in final_url:
                        print("⬅️  Went back using direct navigation (networkidle)")
                        self.url_stack.pop()
                        return True
                except:
                    print(f"⚠️  NetworkIdle timeout but got URL: {final_url}")
                    # Check if we're close enough to the target
                    if target_url in final_url or final_url in target_url:
                        print("⬅️  Went back using direct navigation (close enough)")
                        self.url_stack.pop()
                        return True
        except Exception as e:
            print(f"⚠️  Direct navigation failed: {e}")
        
        # Method 2: Try browser back
        try:
            print("🔄 Trying browser back...")
            await self.page.go_back(wait_until='domcontentloaded', timeout=8000)
            await self.wait_random(2, 3)
            
            final_url = self.page.url
            if final_url != current_url:
                print("⬅️  Went back using browser history")
                # Check if we reached the expected URL (be more flexible)
                if final_url == target_url or target_url in final_url or final_url in target_url:
                    self.url_stack.pop()
                return True
            else:
                print("⚠️  Browser back didn't change URL")
        except Exception as e:
            print(f"⚠️  Browser back failed: {e}")
            # Try browser back without waiting for domcontentloaded
            try:
                print("🔄 Trying browser back without wait condition...")
                await self.page.go_back(timeout=5000)
                await self.wait_random(3, 4)  # Give more time
                
                final_url = self.page.url
                if final_url != current_url:
                    print("⬅️  Went back using browser history (no wait)")
                    if final_url == target_url or target_url in final_url or final_url in target_url:
                        self.url_stack.pop()
                    return True
            except Exception as e2:
                print(f"⚠️  Browser back (no wait) also failed: {e2}")
        
        # Method 3: Try keyboard shortcuts
        try:
            print("🔄 Trying keyboard shortcuts...")
            await self.page.keyboard.press('Alt+ArrowLeft')
            await self.wait_random(2, 3)
            
            final_url = self.page.url
            if final_url != current_url:
                print("⬅️  Went back using keyboard shortcut")
                if final_url == target_url or target_url in final_url:
                    self.url_stack.pop()
                return True
            else:
                print("⚠️  Keyboard shortcut didn't change URL")
        except Exception as e:
            print(f"⚠️  Keyboard shortcut failed: {e}")
        
        print("❌ All back navigation methods failed")
        print(f"   Current URL: {self.page.url}")
        print(f"   Target URL: {target_url}")
        return False
    
    def get_unique_filename(self, folder_path, filename):
        """Generate a unique filename by adding (2), (3), etc. if file exists"""
        original_path = os.path.join(folder_path, filename)
        
        # If file doesn't exist, use original name
        if not os.path.exists(original_path):
            return filename
        
        # Split filename into name and extension
        name, ext = os.path.splitext(filename)
        counter = 2
        
        # Keep trying until we find a unique name
        while True:
            new_filename = f"{name} ({counter}){ext}"
            new_path = os.path.join(folder_path, new_filename)
            
            if not os.path.exists(new_path):
                print(f"🔄 File exists, renamed to: {new_filename}")
                return new_filename
            
            counter += 1
            
            # Safety check to prevent infinite loop
            if counter > 1000:
                import time
                timestamp = int(time.time())
                new_filename = f"{name}_{timestamp}{ext}"
                print(f"🔄 Too many duplicates, using timestamp: {new_filename}")
                return new_filename

    async def handle_downloads(self):
        """Monitor and handle downloads with folder structure preservation"""
        print("📋 Monitoring downloads...")
        
        # Set up download handler
        async def handle_download(download):
            filename = download.suggested_filename
            
            # Create the full local path preserving folder structure
            local_folder_path = os.path.join(self.base_download_dir, self.current_local_path)
            os.makedirs(local_folder_path, exist_ok=True)
            
            # Handle duplicate files by generating unique filename
            unique_filename = self.get_unique_filename(local_folder_path, filename)
            download_path = os.path.join(local_folder_path, unique_filename)
            
            await download.save_as(download_path)
            if unique_filename != filename:
                print(f"💾 Downloaded: {filename} → {self.current_local_path}/{unique_filename} (renamed)")
            else:
                print(f"💾 Downloaded: {filename} → {self.current_local_path}/{filename}")
        
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
        
        # Continuous operation mode
        while True:
            try:
                # Login to Google Drive and let user navigate to desired folder
                await downloader.login_to_drive()
                
                # Start recursive download from current folder
                print("\n🚀 Starting recursive download...")
                await downloader.download_folder_recursively()
                
                print(f"\n🎉 Current folder download completed!")
                print(f"📊 Total files downloaded this session: {downloader.download_count}")
                print(f"📁 Downloads saved to: {downloader.base_download_dir}")
                
                # Ask user if they want to continue with another folder
                print("\n" + "="*60)
                print("📋 DOWNLOAD COMPLETE - Ready for next folder")
                print("="*60)
                print("🗂️  Navigate to the next folder you want to download")
                print("✅ Press Enter when ready to download the next folder")
                print("❌ Type 'quit' or 'exit' to stop the script")
                
                user_choice = input("Continue? (Enter to continue, 'quit' to exit): ").strip().lower()
                
                if user_choice in ['quit', 'exit', 'q', 'stop']:
                    print("👋 Stopping download script...")
                    break
                
                print(f"\n🔄 Ready for next download session...")
                
            except KeyboardInterrupt:
                print("\n\n⚠️  Script interrupted by user (Ctrl+C)")
                break
            except Exception as e:
                print(f"❌ An error occurred in this session: {e}")
                import traceback
                traceback.print_exc()
                
                # Ask if user wants to continue despite the error
                continue_choice = input("\n❓ Error occurred. Continue with next folder? (y/N): ").strip().lower()
                if continue_choice not in ['y', 'yes']:
                    break
        
        print(f"\n📊 Final Statistics:")
        print(f"   Total files downloaded: {downloader.download_count}")
        print(f"   Downloads location: {downloader.base_download_dir}")
        print("🎉 Thank you for using the Google Drive downloader!")
        
    except Exception as e:
        print(f"❌ Fatal error occurred: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Don't automatically close the browser
        input("\nPress Enter to close browser and exit...")
        await downloader.close()

if __name__ == "__main__":
    asyncio.run(main())