#%%

import requests
import time
from datetime import datetime
import os

# Replace with the actual IP address of your ESP32-S3 camera board
CAMERA_URL = "http://192.168.12.135/capture"

# Folder to save images (will be created if it doesn't exist)
SAVE_FOLDER = "./captured_images/"

def capture_image():
    try:
        # First capture to clear the buffer
        requests.get(CAMERA_URL, timeout=10)
        time.sleep(0.5)  # Short delay between captures
        
        # Second capture for the actual image
        response = requests.get(CAMERA_URL, timeout=10)
        if response.status_code == 200:
            # Create a timestamped filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{SAVE_FOLDER}image_{timestamp}.jpg"
            
            # Ensure the save directory exists
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            
            with open(filename, "wb") as f:
                f.write(response.content)
            print(f"Saved image: {filename}")
        else:
            print(f"Error: Received status code {response.status_code}")
    except Exception as e:
        print("Error capturing image:", e)

def main():
    while True:
        capture_image()
        # Wait for 60 seconds
        time.sleep(60)

if __name__ == "__main__":
    main()
