#!/usr/bin/env python3

import pyaudio
import webrtcvad
import numpy as np
import time

def test_vad_sensitivity():
    """Test VAD with different sensitivity levels"""
    print("=== Testing VAD Sensitivity Levels ===")
    
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
    
    # Test all VAD modes (0=least aggressive, 3=most aggressive)
    for vad_mode in range(4):
        print(f"\n--- Testing VAD Mode {vad_mode} ---")
        vad = webrtcvad.Vad(vad_mode)
        
        print("Speak now for 5 seconds...")
        speech_detected = 0
        total_frames = 0
        
        for i in range(167):  # ~5 seconds at 30ms frames
            frame = stream.read(FRAME_SIZE, exception_on_overflow=False)
            is_speech = vad.is_speech(frame, RATE)
            
            if is_speech:
                speech_detected += 1
            total_frames += 1
            
            if i % 33 == 0:  # Print every second
                status = "SPEECH" if is_speech else "SILENCE"
                print(f"  {i//33 + 1}s: {status}")
        
        speech_percentage = (speech_detected / total_frames) * 100
        print(f"  Speech detected: {speech_detected}/{total_frames} frames ({speech_percentage:.1f}%)")
    
    stream.stop_stream()
    stream.close()
    audio.terminate()

def test_audio_levels():
    """Test raw audio levels to see if microphone is picking up sound"""
    print("\n=== Testing Audio Levels ===")
    
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 16000
    CHUNK = 1024
    
    audio = pyaudio.PyAudio()
    stream = audio.open(format=FORMAT,
                      channels=CHANNELS,
                      rate=RATE,
                      input=True,
                      frames_per_buffer=CHUNK)
    
    print("Monitoring audio levels for 10 seconds...")
    print("Speak to see if microphone is picking up sound:")
    print("(Looking for RMS values > 100 for speech)")
    
    for i in range(100):  # 10 seconds at ~10Hz
        data = stream.read(CHUNK, exception_on_overflow=False)
        
        # Convert to numpy array and calculate RMS
        audio_data = np.frombuffer(data, dtype=np.int16)
        rms = np.sqrt(np.mean(audio_data**2))
        
        # Create a simple level meter
        level_bars = int(rms / 100)
        level_display = "█" * min(level_bars, 50)
        
        print(f"\rRMS: {rms:6.0f} |{level_display:<50}|", end="", flush=True)
        time.sleep(0.1)
    
    print("\n")
    stream.stop_stream()
    stream.close()
    audio.terminate()

def test_microphone_permissions():
    """Test if we have proper microphone permissions"""
    print("\n=== Testing Microphone Permissions ===")
    
    try:
        audio = pyaudio.PyAudio()
        
        # Try to open a stream with default input device
        stream = audio.open(format=pyaudio.paInt16,
                          channels=1,
                          rate=16000,
                          input=True,
                          frames_per_buffer=1024)
        
        # Try to read a frame
        data = stream.read(1024, exception_on_overflow=False)
        
        stream.stop_stream()
        stream.close()
        audio.terminate()
        
        print("✓ Microphone permissions OK")
        return True
        
    except Exception as e:
        print(f"✗ Microphone permission error: {e}")
        print("  You may need to grant microphone access in System Preferences > Security & Privacy > Privacy > Microphone")
        return False

def main():
    print("🔍 Detailed VAD and Audio Level Debug")
    print("=" * 60)
    
    # Test permissions first
    if not test_microphone_permissions():
        return
    
    # Test audio levels
    test_audio_levels()
    
    # Test VAD sensitivity
    test_vad_sensitivity()
    
    print("\n" + "=" * 60)
    print("Debug complete!")
    print("\nTips:")
    print("- If RMS levels are very low (< 100), check microphone volume/gain")
    print("- If no VAD mode detects speech, the audio might be too quiet")
    print("- Try speaking louder or closer to the microphone")
    print("- Check if AirPods microphone is working in other apps")

if __name__ == '__main__':
    main() 