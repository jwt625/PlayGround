


#%%
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import csv


#%%
def login_to_twitter(driver, username, password):
    driver.get("https://twitter.com/login")
    # Add code to input username and password and submit the form
    # Use WebDriverWait to wait for elements to be clickable

def scrape_tweets(driver):
    driver.get(f"https://twitter.com/{username}")
    tweets = []
    # Add code to scroll and collect tweets
    return tweets

def scrape_bookmarks(driver):
    driver.get("https://twitter.com/i/bookmarks")
    bookmarks = []
    # Add code to scroll and collect bookmarks
    return bookmarks

def save_to_csv(data, filename):
    with open(filename, 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerows(data)

# Main execution
username = "your_username"
password = "your_password"

driver = webdriver.Chrome()  # Or whichever browser you're using

try:
    login_to_twitter(driver, username, password)
    
    tweets = scrape_tweets(driver)
    save_to_csv(tweets, 'tweets.csv')
    
    bookmarks = scrape_bookmarks(driver)
    save_to_csv(bookmarks, 'bookmarks.csv')

finally:
    driver.quit()


#%%

def scrape_bookmarks(driver, N = 10):
    bookmarks = []
    last_height = driver.execute_script("return document.body.scrollHeight")

    for ii in range(N):
        # Scroll down to bottom
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")

        # Wait to load page
        time.sleep(2)

        # Calculate new scroll height and compare with last scroll height
        new_height = driver.execute_script("return document.body.scrollHeight")
        if new_height == last_height:
            break
        last_height = new_height

        # Find all tweet elements
        tweet_elements = driver.find_elements(By.CSS_SELECTOR, 'article[data-testid="tweet"]')

        for tweet in tweet_elements:
            tweet_data = {}

            # Extract tweet text
            try:
                tweet_text = tweet.find_element(By.CSS_SELECTOR, 'div[data-testid="tweetText"]').text
                tweet_data['text'] = tweet_text
            except:
                tweet_data['text'] = ''

            # Extract media (if any)
            try:
                media_elements = tweet.find_elements(By.CSS_SELECTOR, 'div[data-testid="tweetPhoto"]')
                tweet_data['media'] = [elem.find_element(By.TAG_NAME, 'img').get_attribute('src') for elem in media_elements]
            except:
                tweet_data['media'] = []

            # Extract tweet URL
            try:
                tweet_time = tweet.find_element(By.CSS_SELECTOR, 'time')
                tweet_url = tweet_time.find_element(By.XPATH, './..').get_attribute('href')
                tweet_data['url'] = tweet_url
            except:
                tweet_data['url'] = ''

            if tweet_data not in bookmarks:
                bookmarks.append(tweet_data)
        print(f"{ii}/{N}")
    return bookmarks


#%%

import json
# Scrape bookmarks
bookmarked_tweets = scrape_bookmarks(driver)

# Save to JSON file
with open('bookmarked_tweets.json', 'w', encoding='utf-8') as f:
    json.dump(bookmarked_tweets, f, ensure_ascii=False, indent=4)

print(f"Scraped {len(bookmarked_tweets)} bookmarked tweets.")


