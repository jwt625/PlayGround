#!/usr/bin/env python3
"""
Test AssemblyAI WebSocket streaming API directly
Based on the actual API documentation
"""

import os
import json
import asyncio
import websockets
import pyaudio
import threading
import time
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

# Audio settings
SAMPLE_RATE = 16000
CHUNK_SIZE = 1024  # ~64ms chunks
FORMAT = pyaudio.paInt16
CHANNELS = 1

class AssemblyAIWebSocketStreamer:
    def __init__(self):
        self.websocket = None
        self.audio = pyaudio.PyAudio()
        self.stream = None
        self.running = False
        self.token = None

    def generate_token(self):
        """Generate a temporary streaming token"""
        url = "https://streaming.assemblyai.com/v3/token"
        headers = {
            "Authorization": api_key
        }
        params = {
            "expires_in_seconds": 500  # 10 minutes (maximum allowed)
        }

        try:
            print(f"🔑 Generating streaming token...")
            response = requests.get(url, headers=headers, params=params)

            if response.status_code == 200:
                token_data = response.json()
                self.token = token_data.get('token')
                print(f"✅ Got streaming token: {self.token[:20]}...")
                return True
            else:
                print(f"❌ Failed to get token: {response.status_code} - {response.text}")
                return False

        except Exception as e:
            print(f"💥 Error generating token: {e}")
            return False

    async def connect(self):
        """Connect to AssemblyAI WebSocket"""
        if not self.token:
            print("❌ No token available")
            return False

        uri = f"wss://streaming.assemblyai.com/v3/ws?sample_rate={SAMPLE_RATE}&encoding=pcm_s16le&format_turns=true"

        try:
            print(f"🔗 Connecting to {uri}")
            # Use the temporary token for authentication
            self.websocket = await websockets.connect(
                uri,
                additional_headers={"Authorization": f"Bearer {self.token}"}
            )
            print("✅ Connected to AssemblyAI WebSocket")
            return True
        except Exception as e:
            print(f"❌ Failed to connect: {e}")
            return False
    
    async def listen_for_messages(self):
        """Listen for messages from AssemblyAI"""
        try:
            async for message in self.websocket:
                data = json.loads(message)
                msg_type = data.get('type')
                
                if msg_type == 'Begin':
                    print(f"🌊 Session started: {data.get('id')}")
                elif msg_type == 'Turn':
                    text = data.get('text', '')
                    is_final = data.get('end_of_turn', False)
                    confidence = data.get('confidence', 0.0)
                    
                    if text:
                        status = "FINAL" if is_final else "PARTIAL"
                        print(f"📝 {status}: {text} (confidence: {confidence:.2f})")
                elif msg_type == 'Termination':
                    print(f"🛑 Session terminated: {data}")
                    break
                else:
                    print(f"📨 Unknown message: {data}")
                    
        except websockets.exceptions.ConnectionClosed:
            print("🔌 WebSocket connection closed")
        except Exception as e:
            print(f"❌ Error listening for messages: {e}")
    
    def start_audio_stream(self):
        """Start capturing audio from microphone"""
        try:
            self.stream = self.audio.open(
                format=FORMAT,
                channels=CHANNELS,
                rate=SAMPLE_RATE,
                input=True,
                frames_per_buffer=CHUNK_SIZE
            )
            print("🎤 Started microphone stream")
            return True
        except Exception as e:
            print(f"❌ Failed to start audio stream: {e}")
            return False
    
    async def send_audio_loop(self):
        """Send audio data to WebSocket"""
        print("🎙️ Starting audio send loop...")
        
        while self.running and self.websocket:
            try:
                # Read audio data
                audio_data = self.stream.read(CHUNK_SIZE, exception_on_overflow=False)
                
                # Send to WebSocket
                await self.websocket.send(audio_data)
                
                # Small delay to prevent overwhelming
                await asyncio.sleep(0.01)
                
            except Exception as e:
                print(f"❌ Error sending audio: {e}")
                break
    
    async def run(self):
        """Main run loop"""
        # Generate token first
        if not self.generate_token():
            return

        # Connect to WebSocket
        if not await self.connect():
            return
        
        # Start audio stream
        if not self.start_audio_stream():
            return
        
        self.running = True
        
        try:
            # Start both listening and sending tasks
            listen_task = asyncio.create_task(self.listen_for_messages())
            send_task = asyncio.create_task(self.send_audio_loop())
            
            print("🎯 Streaming started! Speak into your microphone...")
            print("Press Ctrl+C to stop")
            
            # Wait for either task to complete
            await asyncio.gather(listen_task, send_task, return_exceptions=True)
            
        except KeyboardInterrupt:
            print("\n⏹️ Stopping...")
        finally:
            await self.cleanup()
    
    async def cleanup(self):
        """Clean up resources"""
        self.running = False
        
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
            print("🔇 Stopped audio stream")
        
        if self.websocket:
            # Send termination message
            try:
                termination_msg = {"type": "SessionTermination"}
                await self.websocket.send(json.dumps(termination_msg))
                await self.websocket.close()
                print("🔌 Closed WebSocket")
            except:
                pass
        
        self.audio.terminate()
        print("✅ Cleanup complete")

async def main():
    streamer = AssemblyAIWebSocketStreamer()
    await streamer.run()

if __name__ == "__main__":
    asyncio.run(main())
