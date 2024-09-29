

#%%
import tweepy
import json
import webbrowser

#%%
# Load bearer token from token.json
with open('token.json') as f:
    data = json.load(f)
    bearer_token = data["bearer_token"]
    user_id = data["user_id"]

client = tweepy.Client(bearer_token=bearer_token)

# Get the user's liked tweets (used as a workaround for bookmarks)
liked_tweets = client.get_liked_tweets(id=user_id, max_results=1)
for tweet in liked_tweets.data:
    print(tweet.text)

# %% test getting user id
with open('token.json') as f:
    data = json.load(f)
    bearer_token = data["bearer_token"]

client = tweepy.Client(bearer_token=bearer_token)

user = client.get_user(username="jwt0625")
user_id = user.data.id
print(f"Your user ID is: {user_id}")





#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# %% test claud's code

# load tokens
import json
with open('token.json') as f:
    data = json.load(f)
    bearer_token = data["bearer_token"]
    user_id = data["user_id"]
    client_id = data["client_id"]
    client_secret = data["client_secret"]
    access_token = data["access_token"]

import requests
from requests_oauthlib import OAuth2Session
from oauthlib.oauth2 import BackendApplicationClient

# Your Twitter API credentials
# client_id = "YOUR_CLIENT_ID"
# client_secret = "YOUR_CLIENT_SECRET"
# access_token = "YOUR_ACCESS_TOKEN"
# user_id = "YOUR_USER_ID"

# Create OAuth 2 session
client = BackendApplicationClient(client_id=client_id)
oauth = OAuth2Session(client=client)

# Twitter API v2 bookmarks endpoint
url = f"https://api.twitter.com/2/users/{user_id}/bookmarks"

# Parameters for the request
params = {
    "max_results": 2,  # Retrieve the last 2 bookmarks
    "tweet.fields": "created_at,author_id,text",  # Additional tweet fields
    "expansions": "author_id",  # Expand author information
    "user.fields": "username,name"  # Additional user fields
}

# Headers for the request
headers = {
    "Authorization": f"Bearer {access_token}",
    "User-Agent": "BookmarksRetrievalScript"
}

# Make the request
response = requests.get(url, headers=headers, params=params)

# Check if the request was successful
if response.status_code != 200:
    raise Exception(f"Request returned an error: {response.status_code} {response.text}")

# Parse the JSON response
json_response = response.json()

# Print the formatted JSON response
import json
print(json.dumps(json_response, indent=2))

# Process and display the bookmarks in a more readable format
if "data" in json_response:
    print("\nYour last 2 bookmarks:")
    for tweet in json_response["data"]:
        author = next((user for user in json_response["includes"]["users"] if user["id"] == tweet["author_id"]), None)
        print(f"\nTweet by {author['name']} (@{author['username']}):")
        print(tweet["text"])
        print(f"Created at: {tweet['created_at']}")
        print(f"Tweet ID: {tweet['id']}")
else:
    print("No bookmarks found.")


    
# %% version 3
import requests

# Your Twitter API credentials
# access_token = "YOUR_OAUTH2_ACCESS_TOKEN"
# user_id = "YOUR_USER_ID"

# Twitter API v2 bookmarks endpoint
url = f"https://api.twitter.com/2/users/{user_id}/bookmarks"

# Parameters for the request
params = {
    "max_results": 2,  # Retrieve the last 2 bookmarks
    "tweet.fields": "created_at,author_id,text",  # Additional tweet fields
    "expansions": "author_id",  # Expand author information
    "user.fields": "username,name"  # Additional user fields
}

# Headers for the request
headers = {
    "Authorization": f"Bearer {access_token}",
    "User-Agent": "BookmarksRetrievalScript"
}

# Make the request
response = requests.get(url, headers=headers, params=params)

# Check if the request was successful
if response.status_code != 200:
    raise Exception(f"Request returned an error: {response.status_code} {response.text}")

# Parse the JSON response
json_response = response.json()

# Print the formatted JSON response
import json
print(json.dumps(json_response, indent=2))

# Process and display the bookmarks in a more readable format
if "data" in json_response:
    print("\nYour last 2 bookmarks:")
    for tweet in json_response["data"]:
        author = next((user for user in json_response["includes"]["users"] if user["id"] == tweet["author_id"]), None)
        print(f"\nTweet by {author['name']} (@{author['username']}):")
        print(tweet["text"])
        print(f"Created at: {tweet['created_at']}")
        print(f"Tweet ID: {tweet['id']}")
else:
    print("No bookmarks found.")