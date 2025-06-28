#!/usr/bin/env python3
"""
Audio Capture Testing Script
Critical Phase 1.2 - Test both microphone input and system audio output capture
"""

import pyaudio
import wave
import time
import sys
import os
from datetime import datetime

class AudioTester:
    def __init__(self):
        self.audio = pyaudio.PyAudio()
        self.sample_rate = 16000  # Whisper-compatible sample rate
        self.channels = 1  # Mono for Whisper
        self.chunk_size = 1024
        self.format = pyaudio.paInt16
        
    def list_audio_devices(self):
        """List all available audio input and output devices"""
        print("=" * 60)
        print("AVAILABLE AUDIO DEVICES")
        print("=" * 60)
        
        device_count = self.audio.get_device_count()
        input_devices = []
        output_devices = []
        
        for i in range(device_count):
            device_info = self.audio.get_device_info_by_index(i)
            print(f"Device {i}: {device_info['name']}")
            print(f"  Max Input Channels: {device_info['maxInputChannels']}")
            print(f"  Max Output Channels: {device_info['maxOutputChannels']}")
            print(f"  Default Sample Rate: {device_info['defaultSampleRate']}")
            print(f"  Host API: {self.audio.get_host_api_info_by_index(device_info['hostApi'])['name']}")
            print()
            
            if device_info['maxInputChannels'] > 0:
                input_devices.append((i, device_info['name']))
            if device_info['maxOutputChannels'] > 0:
                output_devices.append((i, device_info['name']))
        
        print("INPUT DEVICES:")
        for idx, name in input_devices:
            print(f"  {idx}: {name}")
        
        print("\nOUTPUT DEVICES:")
        for idx, name in output_devices:
            print(f"  {idx}: {name}")
        
        return input_devices, output_devices
    
    def test_microphone_capture(self, device_id=None, duration=10):
        """Test microphone input capture"""
        print(f"\n🎤 TESTING MICROPHONE CAPTURE (Device: {device_id or 'default'})")
        print(f"Recording for {duration} seconds...")
        print("Speak into your microphone now!")
        
        try:
            # Open stream
            stream = self.audio.open(
                format=self.format,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                input_device_index=device_id,
                frames_per_buffer=self.chunk_size
            )
            
            frames = []
            start_time = time.time()
            
            # Record audio
            for _ in range(int(self.sample_rate / self.chunk_size * duration)):
                data = stream.read(self.chunk_size)
                frames.append(data)
                
                # Show progress
                elapsed = time.time() - start_time
                print(f"\rRecording... {elapsed:.1f}s / {duration}s", end="", flush=True)
            
            print("\n✅ Recording complete!")
            
            # Stop and close stream
            stream.stop_stream()
            stream.close()
            
            # Save to file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"audio_samples/mic_test_{timestamp}.wav"
            
            with wave.open(filename, 'wb') as wf:
                wf.setnchannels(self.channels)
                wf.setsampwidth(self.audio.get_sample_size(self.format))
                wf.setframerate(self.sample_rate)
                wf.writeframes(b''.join(frames))
            
            print(f"✅ Saved microphone test to: {filename}")
            print(f"   Sample rate: {self.sample_rate} Hz")
            print(f"   Channels: {self.channels}")
            print(f"   Duration: {duration} seconds")
            
            return filename
            
        except Exception as e:
            print(f"❌ Microphone test failed: {e}")
            return None
    
    def test_system_audio_capture(self, device_id=None, duration=10):
        """Test system audio output capture"""
        print(f"\n🔊 TESTING SYSTEM AUDIO CAPTURE (Device: {device_id or 'default'})")
        print(f"Recording for {duration} seconds...")
        print("Play some audio on your system now (music, video, etc.)!")
        print("Note: You may need to configure a virtual audio device for this to work.")
        
        try:
            # Open stream
            stream = self.audio.open(
                format=self.format,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                input_device_index=device_id,
                frames_per_buffer=self.chunk_size
            )
            
            frames = []
            start_time = time.time()
            
            # Record audio
            for _ in range(int(self.sample_rate / self.chunk_size * duration)):
                data = stream.read(self.chunk_size)
                frames.append(data)
                
                # Show progress
                elapsed = time.time() - start_time
                print(f"\rRecording... {elapsed:.1f}s / {duration}s", end="", flush=True)
            
            print("\n✅ Recording complete!")
            
            # Stop and close stream
            stream.stop_stream()
            stream.close()
            
            # Save to file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"audio_samples/system_test_{timestamp}.wav"
            
            with wave.open(filename, 'wb') as wf:
                wf.setnchannels(self.channels)
                wf.setsampwidth(self.audio.get_sample_size(self.format))
                wf.setframerate(self.sample_rate)
                wf.writeframes(b''.join(frames))
            
            print(f"✅ Saved system audio test to: {filename}")
            print(f"   Sample rate: {self.sample_rate} Hz")
            print(f"   Channels: {self.channels}")
            print(f"   Duration: {duration} seconds")
            
            return filename
            
        except Exception as e:
            print(f"❌ System audio test failed: {e}")
            return None
    
    def test_dual_capture(self, mic_device=None, system_device=None, duration=10):
        """Test capturing both microphone and system audio simultaneously"""
        print(f"\n🎤🔊 TESTING DUAL AUDIO CAPTURE")
        print(f"Recording both mic and system audio for {duration} seconds...")
        print("Speak AND play audio simultaneously!")
        
        # This is a simplified version - in practice, you'd need separate threads
        # or a more sophisticated approach to handle dual streams
        print("⚠️  Note: This is a basic test. Full dual capture requires more complex implementation.")
        
        # For now, just test them sequentially
        print("\n1. Testing microphone first...")
        mic_file = self.test_microphone_capture(mic_device, duration//2)
        
        print("\n2. Testing system audio...")
        system_file = self.test_system_audio_capture(system_device, duration//2)
        
        return mic_file, system_file
    
    def cleanup(self):
        """Clean up PyAudio"""
        self.audio.terminate()

def main():
    print("🎯 CHATGPT VOICE MODE TRANSCRIPT RECORDER")
    print("Phase 1.2: Critical Audio Capture Testing")
    print("=" * 60)

    # Create audio_samples directory if it doesn't exist
    os.makedirs("audio_samples", exist_ok=True)

    tester = AudioTester()

    try:
        # List all available devices
        input_devices, output_devices = tester.list_audio_devices()

        # Check for critical issues
        if not input_devices:
            print("\n🚨 CRITICAL ISSUE: NO INPUT DEVICES DETECTED!")
            print("=" * 60)
            print("This means no microphone is available for audio capture.")
            print("Please check the AUDIO_SETUP.md file for detailed setup instructions.")
            print("\nCommon solutions:")
            print("1. Connect a USB microphone or headset")
            print("2. Grant microphone permissions in System Preferences")
            print("3. Check System Preferences → Sound → Input")
            print("\nYou can still test system audio capture if available.")
            print("=" * 60)

        print("\n" + "=" * 60)
        print("AUDIO TESTING MENU")
        print("=" * 60)
        if input_devices:
            print("1. Test microphone capture ✅")
        else:
            print("1. Test microphone capture ❌ (No input devices)")
        print("2. Test system audio capture")
        print("3. Test dual capture (mic + system)")
        print("4. List devices again")
        print("5. Check system audio permissions")
        print("6. Exit")
        
        while True:
            choice = input("\nEnter your choice (1-6): ").strip()

            if choice == "1":
                if not input_devices:
                    print("❌ No input devices available. Please connect a microphone first.")
                    print("See AUDIO_SETUP.md for detailed instructions.")
                else:
                    device_id = input("Enter microphone device ID (or press Enter for default): ").strip()
                    device_id = int(device_id) if device_id.isdigit() else None
                    duration = input("Enter recording duration in seconds (default 10): ").strip()
                    duration = int(duration) if duration.isdigit() else 10
                    tester.test_microphone_capture(device_id, duration)

            elif choice == "2":
                device_id = input("Enter system audio device ID (or press Enter for default): ").strip()
                device_id = int(device_id) if device_id.isdigit() else None
                duration = input("Enter recording duration in seconds (default 10): ").strip()
                duration = int(duration) if duration.isdigit() else 10
                tester.test_system_audio_capture(device_id, duration)

            elif choice == "3":
                if not input_devices:
                    print("❌ No input devices available for dual capture.")
                    print("Please connect a microphone first.")
                else:
                    mic_id = input("Enter microphone device ID (or press Enter for default): ").strip()
                    mic_id = int(mic_id) if mic_id.isdigit() else None
                    sys_id = input("Enter system audio device ID (or press Enter for default): ").strip()
                    sys_id = int(sys_id) if sys_id.isdigit() else None
                    duration = input("Enter recording duration in seconds (default 10): ").strip()
                    duration = int(duration) if duration.isdigit() else 10
                    tester.test_dual_capture(mic_id, sys_id, duration)

            elif choice == "4":
                input_devices, output_devices = tester.list_audio_devices()

            elif choice == "5":
                print("\n🔍 CHECKING SYSTEM AUDIO PERMISSIONS...")
                print("To check microphone permissions:")
                print("1. Open System Preferences")
                print("2. Go to Security & Privacy → Privacy")
                print("3. Select 'Microphone' from the left sidebar")
                print("4. Ensure 'Terminal' is checked")
                print("5. Restart Terminal if you made changes")

            elif choice == "6":
                break

            else:
                print("Invalid choice. Please enter 1-6.")
    
    except KeyboardInterrupt:
        print("\n\nTesting interrupted by user.")
    
    finally:
        tester.cleanup()
        print("\n✅ Audio testing complete!")
        print("Check the audio_samples/ directory for recorded files.")

if __name__ == "__main__":
    main()
