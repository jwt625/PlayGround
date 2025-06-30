#!/usr/bin/env python3
"""
Minimal AssemblyAI streaming demo using direct API endpoints
No SDK - just raw WebSocket API like the reference demo
"""

import os
import asyncio
import websockets
import json
import pyaudio
import requests

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("✅ Loaded .env file")
except ImportError:
    print("⚠️ python-dotenv not available")

# Get API key
ASSEMBLYAI_API_KEY = os.getenv('API_KEY')
if not ASSEMBLYAI_API_KEY:
    print("❌ No API_KEY found in environment")
    exit(1)

print(f"🔑 Using API key: {ASSEMBLYAI_API_KEY[:10]}...")

# Audio settings
SAMPLE_RATE = 16000
CHUNK_SIZE = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1

async def get_streaming_token():
    """Get a temporary streaming token"""
    url = "https://api.assemblyai.com/v2/realtime/token"
    headers = {"authorization": ASSEMBLYAI_API_KEY}

    response = requests.post(url, headers=headers, json={"expires_in": 3600})

    if response.status_code == 200:
        token = response.json()["token"]
        print(f"✅ Got streaming token: {token[:20]}...")
        return token
    else:
        print(f"❌ Failed to get token: {response.status_code} - {response.text}")
        return None

class StreamingTranscriber:
    def __init__(self):
        self.websocket = None
        self.audio = pyaudio.PyAudio()
        self.stream = None
        self.running = False

    async def connect(self, token):
        """Connect to AssemblyAI WebSocket"""
        uri = f"wss://api.assemblyai.com/v2/realtime/ws?sample_rate={SAMPLE_RATE}&token={token}"

        try:
            print(f"🔗 Connecting to AssemblyAI WebSocket...")
            self.websocket = await websockets.connect(uri)
            print("✅ Connected to AssemblyAI streaming")
            return True
        except Exception as e:
            print(f"❌ Failed to connect: {e}")
            return False

    async def listen_for_transcripts(self):
        """Listen for transcript messages"""
        try:
            async for message in self.websocket:
                data = json.loads(message)
                message_type = data.get("message_type")

                if message_type == "SessionBegins":
                    print(f"🌊 Session started: {data.get('session_id')}")
                elif message_type == "PartialTranscript":
                    text = data.get("text", "")
                    if text:
                        print(f"📝 PARTIAL: {text}")
                elif message_type == "FinalTranscript":
                    text = data.get("text", "")
                    confidence = data.get("confidence", 0)
                    if text:
                        print(f"📝 FINAL: {text} (confidence: {confidence:.2f})")
                elif message_type == "SessionTerminated":
                    print("🛑 Session terminated")
                    break
                else:
                    print(f"📨 Unknown message: {data}")

        except websockets.exceptions.ConnectionClosed:
            print("🔌 WebSocket connection closed")
        except Exception as e:
            print(f"❌ Error listening: {e}")

    async def send_audio_loop(self):
        """Send audio data to WebSocket"""
        print("🎙️ Starting audio capture...")

        # Open audio stream
        self.stream = self.audio.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=SAMPLE_RATE,
            input=True,
            frames_per_buffer=CHUNK_SIZE
        )

        self.running = True

        while self.running and self.websocket:
            try:
                # Read audio data
                audio_data = self.stream.read(CHUNK_SIZE, exception_on_overflow=False)

                # Send to WebSocket
                await self.websocket.send(audio_data)

                # Small delay
                await asyncio.sleep(0.01)

            except Exception as e:
                print(f"❌ Error sending audio: {e}")
                break

    async def start_streaming(self):
        """Start streaming transcription"""
        try:
            # Start both listening and sending tasks
            listen_task = asyncio.create_task(self.listen_for_transcripts())
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
            try:
                # Send termination message
                await self.websocket.send(json.dumps({"terminate_session": True}))
                await self.websocket.close()
                print("🔌 Closed WebSocket")
            except:
                pass

        self.audio.terminate()
        print("✅ Cleanup complete")

async def main():
    print("🎯 Minimal AssemblyAI Streaming Demo")
    print("Using direct WebSocket API endpoints")
    print("=" * 50)

    # Get streaming token
    token = await get_streaming_token()
    if not token:
        return

    # Create transcriber
    transcriber = StreamingTranscriber()

    # Connect to WebSocket
    if await transcriber.connect(token):
        # Start streaming
        await transcriber.start_streaming()
    else:
        print("❌ Failed to connect to streaming service")

if __name__ == "__main__":
    asyncio.run(main())
