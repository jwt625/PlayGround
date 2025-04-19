


#%%
import requests
import json

# Load configuration from config.json
with open('config.json', 'r') as f:
    config = json.load(f)

API_URL = config['api_url']
API_KEY = config['api_key']

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
}

data = {
    "model": "deepseek-chat",  # Specify the model you want to use
    "messages": [
        {"role": "user", "content": "Hello! Can you explain how APIs work?"}
    ],
    "temperature": 0.7,  # Controls randomness (0.0 to 1.0)
    "max_tokens": 500,   # Limit response length
}

response = requests.post(API_URL, headers=headers, json=data)

if response.status_code == 200:
    result = response.json()
    reply = result["choices"][0]["message"]["content"]
    print("AI Response:", reply)
else:
    print("Error:", response.status_code, response.text)


    
#%%
from openai import OpenAI

# Configure with DeepSeek's API endpoint and key
client = OpenAI(
    api_key=API_KEY,  # Replace with your key
    base_url="https://api.deepseek.com/v1"  # DeepSeek-compatible endpoint
)

response = client.chat.completions.create(
    model="deepseek-chat",  # Now maps to DeepSeek-V3 as of 2025-03-25:cite[3]:cite[4]
    messages=[
        {"role": "system", "content": "You are a coding assistant"},
        {"role": "user", "content": "Write a Python function to reverse a string"}
    ],
    temperature=0.5,
    max_tokens=256
)

print(response.choices[0].message.content)

# %%
