#!/usr/bin/env python3
"""
Minimal AssemblyAI streaming demo
Continuously captures mic input and sends to AssemblyAI for transcription
"""

import os
import asyncio
import requests
import pyaudio
import wave
import tempfile
import threading
import time
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

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
CHUNK_DURATION = 3  # seconds per chunk

# Exact function from the reference
async def transcribe_audio_to_text(audio_bytes):
    headers = {
        "authorization": ASSEMBLYAI_API_KEY,
        "content-type": "application/octet-stream"
    }

    # Step 1: Upload audio file
    upload_response = requests.post(
        "https://api.assemblyai.com/v2/upload",
        headers=headers,
        data=audio_bytes
    )
    audio_url = upload_response.json()["upload_url"]

    # Step 2: Start transcription job
    transcript_response = requests.post(
        "https://api.assemblyai.com/v2/transcript",
        json={"audio_url": audio_url},
        headers={"authorization": ASSEMBLYAI_API_KEY}
    )
    transcript_id = transcript_response.json()["id"]

    # Step 3: Poll for completion
    polling_url = f"https://api.assemblyai.com/v2/transcript/{transcript_id}"
    while True:
        polling_response = requests.get(polling_url, headers={"authorization": ASSEMBLYAI_API_KEY})
        status = polling_response.json()["status"]

        if status == "completed":
            return polling_response.json()["text"]
        elif status == "error":
            print("Transcription failed:", polling_response.json()["error"])
            return None
        await asyncio.sleep(1)

def create_wav_bytes(audio_data):
    """Convert raw audio data to WAV format bytes"""
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
        temp_path = temp_file.name

    # Write WAV file
    with wave.open(temp_path, 'wb') as wav_file:
        wav_file.setnchannels(CHANNELS)
        wav_file.setsampwidth(2)  # 16-bit = 2 bytes
        wav_file.setframerate(SAMPLE_RATE)
        wav_file.writeframes(audio_data)

    # Read WAV file as bytes
    with open(temp_path, 'rb') as f:
        wav_bytes = f.read()

    # Clean up
    os.unlink(temp_path)
    return wav_bytes

class StreamingTranscriber:
    def __init__(self):
        self.audio = pyaudio.PyAudio()
        self.stream = None
        self.running = False

    def start_streaming(self):
        """Start streaming microphone input to AssemblyAI"""
        print("🎤 Starting microphone streaming...")
        print("Speak into your microphone. Press Ctrl+C to stop.")

        try:
            # Open audio stream
            self.stream = self.audio.open(
                format=FORMAT,
                channels=CHANNELS,
                rate=SAMPLE_RATE,
                input=True,
                frames_per_buffer=CHUNK_SIZE
            )

            self.running = True
            chunk_count = 0
            frames = []

            chunks_per_segment = int(SAMPLE_RATE / CHUNK_SIZE * CHUNK_DURATION)

            while self.running:
                try:
                    # Read audio chunk
                    data = self.stream.read(CHUNK_SIZE, exception_on_overflow=False)
                    frames.append(data)
                    chunk_count += 1

                    # Process every N chunks (e.g., every 3 seconds)
                    if chunk_count >= chunks_per_segment:
                        print(f"\n📦 Processing {CHUNK_DURATION}s audio chunk...")

                        # Combine frames into audio data
                        audio_data = b''.join(frames)

                        # Convert to WAV bytes
                        wav_bytes = create_wav_bytes(audio_data)

                        # Send to AssemblyAI (async)
                        asyncio.create_task(self.process_audio_chunk(wav_bytes))

                        # Reset for next chunk
                        frames = []
                        chunk_count = 0

                except Exception as e:
                    print(f"❌ Audio read error: {e}")
                    break

        except KeyboardInterrupt:
            print("\n⏹️ Stopping...")
        finally:
            self.stop_streaming()

    async def process_audio_chunk(self, wav_bytes):
        """Process audio chunk with AssemblyAI"""
        try:
            print(f"📤 Sending {len(wav_bytes)} bytes to AssemblyAI...")
            result = await transcribe_audio_to_text(wav_bytes)

            if result and result.strip():
                print(f"📝 TRANSCRIPT: '{result}'")
            else:
                print("🔇 (silence)")

        except Exception as e:
            print(f"❌ Transcription error: {e}")

    def stop_streaming(self):
        """Stop streaming"""
        self.running = False
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        self.audio.terminate()
        print("✅ Streaming stopped")

async def main():
    print("🎯 Minimal AssemblyAI Streaming Demo")
    print("Simulates real-time streaming using file-based API")
    print("=" * 50)

    transcriber = StreamingTranscriber()

    # Run streaming in a separate thread so async works
    def run_streaming():
        transcriber.start_streaming()

    streaming_thread = threading.Thread(target=run_streaming)
    streaming_thread.daemon = True
    streaming_thread.start()

    # Keep the async event loop running
    try:
        while streaming_thread.is_alive():
            await asyncio.sleep(0.1)
    except KeyboardInterrupt:
        print("\n⏹️ Stopping...")
        transcriber.stop_streaming()

if __name__ == "__main__":
    asyncio.run(main())
