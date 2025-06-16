import time
import os
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.keys import Keys

class GoogleDriveDownloader:
    def __init__(self, download_dir=None):
        self.download_dir = download_dir or os.path.join(os.getcwd(), 'downloads')
        os.makedirs(self.download_dir, exist_ok=True)
        self.driver = self._setup_driver()
        self.wait = WebDriverWait(self.driver, 10)
    
    def _setup_driver(self):
        chrome_options = Options()
        chrome_options.add_experimental_option("prefs", {
            "download.default_directory": self.download_dir,
            "download.prompt_for_download": False,
            "download.directory_upgrade": True,
            "safebrowsing.enabled": True
        })
        # Remove headless mode for authentication
        # chrome_options.add_argument("--headless")
        return webdriver.Chrome(options=chrome_options)
    
    def login_to_google(self):
        """Navigate to Google Drive and handle authentication"""
        print("Opening Google Drive...")
        self.driver.get("https://drive.google.com")
        
        # Wait for user to manually log in
        print("Please log in to Google Drive in the browser window...")
        print("Press Enter here when you're logged in and can see your Drive files...")
        input()
        
        # Verify we're on the Drive page
        try:
            self.wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "[data-target='drive']")))
            print("Successfully logged in to Google Drive!")
        except:
            print("Could not confirm Google Drive login. Please check manually.")
    
    def select_all_files(self):
        """Select all files in current folder"""
        try:
            # Use Ctrl+A to select all
            body = self.driver.find_element(By.TAG_NAME, "body")
            body.send_keys(Keys.CONTROL + "a")
            time.sleep(2)
            print("Selected all files")
        except Exception as e:
            print(f"Error selecting files: {e}")
    
    def download_selected_files(self):
        """Download currently selected files"""
        try:
            # Right-click to open context menu
            files_area = self.wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "[data-target='drive']")))
            
            # Use keyboard shortcut for download
            body = self.driver.find_element(By.TAG_NAME, "body")
            body.send_keys(Keys.CONTROL + Keys.SHIFT + "s")  # Download shortcut
            
            print("Download initiated. Check your downloads folder.")
            time.sleep(5)  # Wait for download to start
        except Exception as e:
            print(f"Error downloading: {e}")
    
    def navigate_to_folder(self, folder_name):
        """Navigate to a specific folder by name"""
        try:
            # Find folder by name and double-click
            folder_element = self.wait.until(
                EC.element_to_be_clickable((By.XPATH, f"//div[@data-tooltip='{folder_name}']"))
            )
            folder_element.click()
            folder_element.click()  # Double-click
            time.sleep(3)
            print(f"Navigated to folder: {folder_name}")
        except Exception as e:
            print(f"Could not find folder '{folder_name}': {e}")
    
    def download_folder_contents(self, folder_names=None):
        """Download contents of specified folders or all files"""
        try:
            if folder_names:
                for folder_name in folder_names:
                    print(f"Processing folder: {folder_name}")
                    self.navigate_to_folder(folder_name)
                    self.select_all_files()
                    self.download_selected_files()
                    
                    # Go back to parent folder
                    back_button = self.driver.find_element(By.CSS_SELECTOR, "[data-tooltip='Back']")
                    back_button.click()
                    time.sleep(2)
            else:
                # Download all files in current view
                self.select_all_files()
                self.download_selected_files()
        except Exception as e:
            print(f"Error during download process: {e}")
    
    def close(self):
        """Close the browser"""
        self.driver.quit()

def main():
    # Initialize downloader
    downloader = GoogleDriveDownloader(download_dir="./drive_downloads")
    
    try:
        # Login to Google Drive
        downloader.login_to_google()
        
        # Option 1: Download all files in root
        # downloader.download_folder_contents()
        
        # Option 2: Download specific folders
        folders_to_download = ["Documents", "Photos", "Projects"]  # Adjust as needed
        downloader.download_folder_contents(folders_to_download)
        
        print("Download process completed!")
        
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        input("Press Enter to close browser...")
        downloader.close()

if __name__ == "__main__":
    main()