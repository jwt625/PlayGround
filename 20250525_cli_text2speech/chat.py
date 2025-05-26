#!/usr/bin/env python3
from openai import OpenAI
import os
import sys
import json

# Set API credentials and endpoint from environment variables
openai_api_key = os.getenv("LLM_API_KEY")
openai_api_base = os.getenv("LLM_API_BASE", "https://api.lambda.ai/v1")

# Initialize the OpenAI client
client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

# Choose the model
model = "llama-4-maverick-17b-128e-instruct-fp8"

# Read prompt from args or stdin
if len(sys.argv) > 1:
    user_prompt = " ".join(sys.argv[1:])
else:
    user_prompt = sys.stdin.read()

# Create a chat completion request with a system prompt
chat_completion = client.chat.completions.create(
    messages=[
        {"role": "system", "content": "You are an expert conversationalist who responds to the best of your ability."},
        {"role": "user", "content": user_prompt}
    ],
    model=model,
)

# Print only the assistant's reply
print(chat_completion.choices[0].message.content.strip())
