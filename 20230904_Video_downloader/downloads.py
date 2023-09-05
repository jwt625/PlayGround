#%%
import requests
import os


#%%
def download_mp4(url, output_folder, fn):
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()

        filename = os.path.join(output_folder, os.path.basename(url)+fn)

        with open(filename, 'wb') as file:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    file.write(chunk)

        print(f"Downloaded: {filename}")
    except requests.exceptions.RequestException as e:
        print(f"Error downloading {url}: {e}")

input_file = "urls.txt"  # Replace with the path to your text file containing URLs
output_folder = "downloaded_mp4"  # Replace with the desired output folder

# Create the output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

try:
    with open(input_file, 'r') as file:
        ind = 0
        for line in file:
            parts = line.strip().split(' ')
            if len(parts) >= 2:
                url = parts[0]
                download_mp4(url, output_folder, f"_{ind}.mp4")
            ind = ind+1
except FileNotFoundError:
    print(f"File not found: {input_file}")


