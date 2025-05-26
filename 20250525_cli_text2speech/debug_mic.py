#!/usr/bin/env python3

import pyaudio
import webrtcvad
import wave
import tempfile
import time
import sys

def test_microphone_access():
    """Test if we can access the microphone"""
    print("=== Testing Microphone Access ===")
    try:
        audio = pyaudio.PyAudio()
        print("✓ PyAudio initialized successfully")
        
        # List available input devices
        print("\nAvailable input devices:")
        for i in range(audio.get_device_count()):
            info = audio.get_device_info_by_index(i)
            if info['maxInputChannels'] > 0:
                print(f"  Device {i}: {info['name']} (channels: {info['maxInputChannels']})")
        
        audio.terminate()
        return True
    except Exception as e:
        print(f"✗ Error accessing microphone: {e}")
        return False

def test_audio_recording():
    """Test basic audio recording"""
    print("\n=== Testing Audio Recording ===")
    try:
        FORMAT = pyaudio.paInt16
        CHANNELS = 1
        RATE = 16000
        CHUNK = 1024
        RECORD_SECONDS = 3
        
        audio = pyaudio.PyAudio()
        
        print(f"Recording for {RECORD_SECONDS} seconds...")
        print("Say something now!")
        
        stream = audio.open(format=FORMAT,
                          channels=CHANNELS,
                          rate=RATE,
                          input=True,
                          frames_per_buffer=CHUNK)
        
        frames = []
        for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
            data = stream.read(CHUNK, exception_on_overflow=False)
            frames.append(data)
            if i % 10 == 0:  # Print progress
                print(".", end="", flush=True)
        
        print("\n✓ Recording completed")
        
        stream.stop_stream()
        stream.close()
        audio.terminate()
        
        # Save to temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        with wave.open(temp_file.name, 'wb') as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(audio.get_sample_size(FORMAT))
            wf.setframerate(RATE)
            wf.writeframes(b''.join(frames))
        
        print(f"✓ Audio saved to: {temp_file.name}")
        return temp_file.name
        
    except Exception as e:
        print(f"✗ Error recording audio: {e}")
        return None

def test_vad():
    """Test Voice Activity Detection"""
    print("\n=== Testing Voice Activity Detection ===")
    try:
        vad = webrtcvad.Vad(3)  # Aggressive mode
        print("✓ WebRTC VAD initialized successfully")
        
        # Test with some sample data
        FORMAT = pyaudio.paInt16
        CHANNELS = 1
        RATE = 16000
        FRAME_DURATION = 30  # ms
        FRAME_SIZE = int(RATE * FRAME_DURATION / 1000)
        
        audio = pyaudio.PyAudio()
        stream = audio.open(format=FORMAT,
                          channels=CHANNELS,
                          rate=RATE,
                          input=True,
                          frames_per_buffer=FRAME_SIZE)
        
        print("Testing VAD for 10 seconds...")
        print("Speak to see voice activity detection in action:")
        
        for i in range(333):  # ~10 seconds at 30ms frames
            frame = stream.read(FRAME_SIZE, exception_on_overflow=False)
            is_speech = vad.is_speech(frame, RATE)
            
            if i % 33 == 0:  # Print every second
                status = "SPEECH" if is_speech else "SILENCE"
                print(f"Frame {i//33 + 1}s: {status}")
        
        stream.stop_stream()
        stream.close()
        audio.terminate()
        
        print("✓ VAD test completed")
        return True
        
    except Exception as e:
        print(f"✗ Error testing VAD: {e}")
        return False

def test_whisper_import():
    """Test if Whisper can be imported and used"""
    print("\n=== Testing Whisper Import ===")
    try:
        import whisper
        print("✓ Whisper imported successfully")
        
        # Try to load the smallest model
        print("Loading tiny model...")
        model = whisper.load_model("tiny")
        print("✓ Whisper model loaded successfully")
        return True
        
    except Exception as e:
        print(f"✗ Error with Whisper: {e}")
        return False

def test_full_pipeline(audio_file):
    """Test the full pipeline with recorded audio"""
    if not audio_file:
        print("\n=== Skipping Full Pipeline Test (no audio file) ===")
        return
        
    print("\n=== Testing Full Pipeline ===")
    try:
        import whisper
        
        print("Loading Whisper model...")
        model = whisper.load_model("tiny")  # Use tiny for faster testing
        
        print("Transcribing recorded audio...")
        result = model.transcribe(audio_file)
        
        print(f"✓ Transcription result: '{result['text']}'")
        
        # Clean up
        import os
        os.remove(audio_file)
        print("✓ Temporary file cleaned up")
        
    except Exception as e:
        print(f"✗ Error in full pipeline: {e}")

def main():
    print("🎤 Microphone and Audio Debug Script")
    print("=" * 50)
    
    # Run all tests
    mic_ok = test_microphone_access()
    if not mic_ok:
        print("\n❌ Microphone access failed. Check permissions and hardware.")
        sys.exit(1)
    
    audio_file = test_audio_recording()
    vad_ok = test_vad()
    whisper_ok = test_whisper_import()
    
    if audio_file and whisper_ok:
        test_full_pipeline(audio_file)
    
    print("\n" + "=" * 50)
    if mic_ok and vad_ok and whisper_ok:
        print("✅ All tests passed! The wspr tool should work.")
    else:
        print("❌ Some tests failed. Check the errors above.")

if __name__ == '__main__':
    main() 