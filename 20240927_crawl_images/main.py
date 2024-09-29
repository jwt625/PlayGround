
#%%
import requests
import os
import time

#%%

# https://cdn.caeonline.com/images/jeol_jbx-6300_61255119.jpg

# Base URL (without the last number)
base_url = "https://cdn.caeonline.com/images/jeol_jbx-6300_"

# Starting number in the address
# start_number = 61255119 - 400
start_number = 61289927 - 200
num_images = 400

# Directory to save images
save_dir = "downloaded_images"
os.makedirs(save_dir, exist_ok=True)

# Function to check if a URL is valid and download the image
def download_image(image_url, save_path):
    try:
        response = requests.get(image_url)
        # Check if the request was successful (status code 200)
        if response.status_code == 200:
            with open(save_path, 'wb') as file:
                file.write(response.content)
            print(f"Downloaded {image_url}")
        else:
            print(f"Failed to download {image_url}: Status code {response.status_code}")
    except Exception as e:
        print(f"Error downloading {image_url}: {e}")

# Loop through the range of image numbers
for i in range(num_images):
    image_number = start_number + i
    image_url = f"{base_url}{image_number}.jpg"
    save_path = os.path.join(save_dir, f"image_{image_number}.jpg")
    
    download_image(image_url, save_path)
    # Add a 1-second pause between requests
    time.sleep(0.5)

# %%
