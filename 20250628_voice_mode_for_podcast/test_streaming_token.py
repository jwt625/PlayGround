#!/usr/bin/env python3
"""
Test generating AssemblyAI streaming token
"""

import os
import requests

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("✅ Loaded .env file")
except ImportError:
    print("⚠️ python-dotenv not available")

# Get API key
api_key = os.getenv('API_KEY')
if not api_key:
    print("❌ No API_KEY found in environment")
    exit(1)

print(f"🔑 Using API key: {api_key[:10]}...")

def test_streaming_token():
    """Test generating a streaming token"""
    url = "https://api.assemblyai.com/v2/realtime/token"
    headers = {
        "Authorization": api_key,
        "Content-Type": "application/json"
    }
    data = {
        "expires_in": 3600  # 1 hour
    }
    
    try:
        print(f"📡 Requesting streaming token from {url}")
        response = requests.post(url, headers=headers, json=data)
        
        print(f"📊 Status: {response.status_code}")
        print(f"📄 Response: {response.text}")
        
        if response.status_code == 200:
            token_data = response.json()
            token = token_data.get('token')
            print(f"✅ Got streaming token: {token[:20]}...")
            return token
        else:
            print(f"❌ Failed to get token: {response.status_code} - {response.text}")
            return None
            
    except Exception as e:
        print(f"💥 Error: {e}")
        return None

if __name__ == "__main__":
    test_streaming_token()
