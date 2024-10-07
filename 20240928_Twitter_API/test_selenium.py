


#%%
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
import time
import csv
import json
import os
import hashlib
import requests
import re

#%%
driver = webdriver.Chrome()  # Or whichever browser you're using

# login manually

#%%

def get_original_media_url(url):
    # Remove the size specification from the URL
    return re.sub(r'&name=\w+', '', url)

def download_media(url, folder):
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        # Create a unique filename based on the URL
        filename = hashlib.md5(url.encode()).hexdigest() + '.jpg'
        filepath = os.path.join(folder, filename)
        with open(filepath, 'wb') as f:
            for chunk in response.iter_content(1024):
                f.write(chunk)
        return filepath
    return None

def scrape_bookmarks(driver, media_folder, N = 1):
    bookmarks = []
    last_height = driver.execute_script("return document.body.scrollHeight")

    for ii in range(N):
        # Scroll down to bottom
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(2)

        # Calculate new scroll height and compare with last scroll height
        new_height = driver.execute_script("return document.body.scrollHeight")
        # if new_height == last_height:
        #     break
        # last_height = new_height

        tweet_elements = driver.find_elements(By.CSS_SELECTOR, 'article[data-testid="tweet"]')

        for tweet in tweet_elements:
            tweet_data = {}

            # Extract tweet text
            try:
                tweet_text = tweet.find_element(By.CSS_SELECTOR, 'div[data-testid="tweetText"]').text
                tweet_data['text'] = tweet_text
            except:
                tweet_data['text'] = ''

            # Extract and process media
            try:
                media_elements = tweet.find_elements(By.CSS_SELECTOR, 'div[data-testid="tweetPhoto"] img')
                tweet_data['media'] = []
                for elem in media_elements:
                    media_url = elem.get_attribute('src')
                    original_url = get_original_media_url(media_url)
                    
                    # Download and save the media
                    local_path = download_media(original_url, media_folder)
                    
                    if local_path:
                        tweet_data['media'].append({
                            'url': original_url,
                            'local_path': os.path.relpath(local_path, start=os.getcwd())
                        })
            except Exception as e:
                print(f"Error processing media: {str(e)}")

            # Extract tweet URL
            try:
                tweet_time = tweet.find_element(By.CSS_SELECTOR, 'time')
                tweet_url = tweet_time.find_element(By.XPATH, './..').get_attribute('href')
                tweet_data['url'] = tweet_url
            except:
                tweet_data['url'] = ''

            if tweet_data not in bookmarks:
                bookmarks.append(tweet_data)
        print(f"Scrolled {ii+1}/{N}...")

    return bookmarks

#%%

media_folder = 'media'
os.makedirs(media_folder, exist_ok=True)

# Scrape bookmarks
bookmarked_tweets = scrape_bookmarks(driver, media_folder, N = 2)

# Save to JSON file
with open('bookmarked_tweets.json', 'w', encoding='utf-8') as f:
    json.dump(bookmarked_tweets, f, ensure_ascii=False, indent=4)

print(f"Scraped {len(bookmarked_tweets)} bookmarked tweets.")







