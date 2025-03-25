#%%

import requests
import time
from datetime import datetime
import os

# Replace with the actual IP address of your ESP32-S3 camera board
CAMERA_URL = "http://192.168.12.135/capture"

# Folder to save images (will be created if it doesn't exist)
SAVE_FOLDER = "./captured_images/"

def capture_image(max_retries=3):
    try:
        print(f"Attempting to connect to {CAMERA_URL}")
        
        # First capture to clear the buffer
        print("Making first request to clear buffer...")
        first_response = requests.get(CAMERA_URL, timeout=10)
        print(f"First request status code: {first_response.status_code}")
        
        # Longer delay between captures
        time.sleep(2)  # Increased from 0.5 to 2 seconds
        
        # Second capture with retries
        for attempt in range(max_retries):
            print(f"Making capture attempt {attempt + 1} of {max_retries}...")
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            response = requests.get(CAMERA_URL, headers=headers, timeout=10)
            print(f"Request status code: {response.status_code}")
            print(f"Response headers: {response.headers}")
            
            if response.status_code == 200:
                content_length = len(response.content)
                print(f"Received image data (size: {content_length} bytes)")
                
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"{SAVE_FOLDER}image_{timestamp}.jpg"
                
                os.makedirs(os.path.dirname(filename), exist_ok=True)
                
                with open(filename, "wb") as f:
                    f.write(response.content)
                print(f"Successfully saved image: {filename}")
                return True  # Successful capture
            else:
                print(f"Attempt {attempt + 1} failed with status code {response.status_code}")
                print(f"Response text: {response.text}")
                if attempt < max_retries - 1:
                    print(f"Waiting 5 seconds before retry...")
                    time.sleep(5)  # Wait 5 seconds between retries
        
        return False  # All attempts failed

    except Exception as e:
        print(f"Error during capture: {str(e)}")
        print(f"Error type: {type(e).__name__}")
        return False

def main():
    print("Starting image capture script...")
    print(f"Images will be saved to: {os.path.abspath(SAVE_FOLDER)}")
    
    consecutive_failures = 0
    while True:
        success = capture_image()
        
        if success:
            consecutive_failures = 0
            print("\nWaiting 60 seconds before next capture...")
            time.sleep(60)
        else:
            consecutive_failures += 1
            # Add exponential backoff for consecutive failures
            wait_time = min(60 * (2 ** consecutive_failures), 300)  # Max 5 minutes
            print(f"\nCapture failed. Waiting {wait_time} seconds before next attempt...")
            time.sleep(wait_time)

if __name__ == "__main__":
    main()
