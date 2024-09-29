

#%%
import tweepy
import json

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

# %%
