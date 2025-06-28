#!/usr/bin/env python3
"""
Minimal AssemblyAI streaming test
Just test the streaming functionality in isolation
"""

import os
import logging
import assemblyai as aai

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("✅ Loaded .env file")
except ImportError:
    print("⚠️ python-dotenv not available")

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get API key
api_key = os.getenv('API_KEY')
if not api_key:
    print("❌ No API_KEY found in environment")
    exit(1)

print(f"🔑 Using API key: {api_key[:10]}...")

# Set AssemblyAI API key
aai.settings.api_key = api_key

def on_open(session_opened: aai.RealtimeSessionOpened):
    """Handle session start"""
    print(f"🌊 Session started: {session_opened.session_id}")

def on_data(transcript: aai.RealtimeTranscript):
    """Handle transcript data"""
    if not transcript.text:
        return
    
    if isinstance(transcript, aai.RealtimeFinalTranscript):
        print(f"📝 FINAL: {transcript.text}")
    else:
        print(f"📝 PARTIAL: {transcript.text}")

def on_error(error: aai.RealtimeError):
    """Handle errors"""
    print(f"❌ Error: {error}")

def on_close():
    """Handle session close"""
    print("🛑 Session closed")

def main():
    print("🎤 Starting AssemblyAI streaming test...")
    print("Speak into your microphone. Press Ctrl+C to stop.")
    
    try:
        # Create realtime transcriber
        transcriber = aai.RealtimeTranscriber(
            sample_rate=16000,
            on_data=on_data,
            on_error=on_error,
            on_open=on_open,
            on_close=on_close
        )
        
        # Connect
        transcriber.connect()
        print("🔗 Connected to AssemblyAI")
        
        # Start streaming from microphone
        print("🎙️ Starting microphone stream...")
        microphone_stream = aai.extras.MicrophoneStream(sample_rate=16000)
        transcriber.stream(microphone_stream)
        
    except KeyboardInterrupt:
        print("\n⏹️ Stopping...")
    except Exception as e:
        print(f"💥 Error: {e}")
    finally:
        try:
            transcriber.close()
        except:
            pass
        print("✅ Done")

if __name__ == "__main__":
    main()
