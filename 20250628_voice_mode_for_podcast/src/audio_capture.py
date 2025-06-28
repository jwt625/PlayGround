#!/usr/bin/env python3
"""
Audio Capture Module
Handles microphone and system audio capture for the transcript recorder
"""

import pyaudio
import wave
import threading
import time
import numpy as np
from collections import deque
import os

class AudioCapture:
    def __init__(self, sample_rate=16000, channels=1, chunk_size=1024):
        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk_size = chunk_size
        self.format = pyaudio.paInt16
        
        self.audio = pyaudio.PyAudio()
        self.is_recording = False
        self.recording_thread = None
        self.callback = None
        
        # Audio buffers
        self.mic_buffer = deque(maxlen=200)  # Keep last 200 chunks (~12 seconds)
        self.system_buffer = deque(maxlen=200)
        
        # Device IDs
        self.mic_device_id = None
        self.system_device_id = None
        
        # Streams
        self.mic_stream = None
        self.system_stream = None
    
    def list_devices(self):
        """List available audio input and output devices"""
        device_count = self.audio.get_device_count()
        input_devices = []
        output_devices = []
        
        for i in range(device_count):
            try:
                device_info = self.audio.get_device_info_by_index(i)
                if device_info['maxInputChannels'] > 0:
                    input_devices.append((i, device_info['name']))
                if device_info['maxOutputChannels'] > 0:
                    output_devices.append((i, device_info['name']))
            except Exception as e:
                print(f"Error getting device {i}: {e}")
        
        return input_devices, output_devices
    
    def set_devices(self, mic_device_id=None, system_device_id=None):
        """Set the device IDs for microphone and system audio"""
        self.mic_device_id = mic_device_id
        self.system_device_id = system_device_id
    
    def start_recording(self, session_id, callback=None):
        """Start recording audio from microphone and system"""
        if self.is_recording:
            raise Exception("Already recording")
        
        self.callback = callback
        self.is_recording = True
        
        # Create session directory
        session_dir = f"audio_samples/{session_id}"
        os.makedirs(session_dir, exist_ok=True)
        self._current_session_dir = session_dir
        print(f"📁 Created session directory: {session_dir}")
        
        # Start recording thread
        self.recording_thread = threading.Thread(
            target=self._recording_loop,
            args=(session_dir,),
            daemon=True
        )
        self.recording_thread.start()
    
    def stop_recording(self):
        """Stop recording"""
        self.is_recording = False

        if self.recording_thread:
            self.recording_thread.join(timeout=5.0)

        # Save final audio buffers
        if hasattr(self, '_current_session_dir') and self._current_session_dir:
            print("💾 Saving final audio buffers...")
            self._save_audio_buffers(self._current_session_dir, 9999)

        self._close_streams()
    
    def _recording_loop(self, session_dir):
        """Main recording loop"""
        try:
            # Initialize streams
            self._init_streams()
            
            chunk_count = 0
            start_time = time.time()
            
            while self.is_recording:
                try:
                    # Read microphone data
                    mic_data = None
                    if self.mic_stream:
                        try:
                            mic_data = self.mic_stream.read(
                                self.chunk_size, 
                                exception_on_overflow=False
                            )
                            self.mic_buffer.append(mic_data)
                            
                            # Calculate and emit audio level (every 10th chunk for performance)
                            if self.callback and chunk_count % 10 == 0:
                                mic_level = self.get_audio_level(mic_data)
                                self.callback(mic_data, source='microphone', audio_level=mic_level)

                            # Process microphone audio for transcription
                            if self.callback and chunk_count % 50 == 0:  # Process every 50th chunk (~3 seconds)
                                audio_chunk = b''.join(list(self.mic_buffer)[-50:])
                                self.callback(audio_chunk, source='microphone', is_transcription=True)
                        
                        except Exception as e:
                            print(f"Microphone read error: {e}")
                    
                    # Read system audio data
                    system_data = None
                    if self.system_stream:
                        try:
                            system_data = self.system_stream.read(
                                self.chunk_size,
                                exception_on_overflow=False
                            )
                            self.system_buffer.append(system_data)
                            
                            # Calculate and emit audio level (every 10th chunk for performance)
                            if self.callback and chunk_count % 10 == 0:
                                system_level = self.get_audio_level(system_data)
                                self.callback(system_data, source='system', audio_level=system_level)

                            # Process system audio for transcription
                            if self.callback and chunk_count % 50 == 0:  # Process every 50th chunk (~3 seconds)
                                audio_chunk = b''.join(list(self.system_buffer)[-50:])
                                self.callback(audio_chunk, source='system', is_transcription=True)
                        
                        except Exception as e:
                            print(f"System audio read error: {e}")
                    
                    chunk_count += 1
                    
                    # Save audio periodically (every 10 seconds for testing)
                    save_interval = self.sample_rate // self.chunk_size * 10  # 10 seconds
                    if chunk_count % save_interval == 0:
                        print(f"💾 Saving audio buffers at chunk {chunk_count}")
                        self._save_audio_buffers(session_dir, chunk_count)
                    
                    # Small delay to prevent excessive CPU usage
                    time.sleep(0.001)
                
                except Exception as e:
                    print(f"Recording loop error: {e}")
                    break
        
        except Exception as e:
            print(f"Recording initialization error: {e}")
        
        finally:
            self._close_streams()
    
    def _init_streams(self):
        """Initialize audio streams"""
        input_devices, _ = self.list_devices()
        
        if not input_devices:
            raise Exception("No input devices available")
        
        # Initialize microphone stream
        try:
            self.mic_stream = self.audio.open(
                format=self.format,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                input_device_index=self.mic_device_id,
                frames_per_buffer=self.chunk_size,
                stream_callback=None
            )
            print(f"✅ Microphone stream initialized (device: {self.mic_device_id or 'default'})")
        
        except Exception as e:
            print(f"❌ Failed to initialize microphone stream: {e}")
            self.mic_stream = None
        
        # Initialize system audio stream (if device is available)
        if self.system_device_id is not None:
            try:
                self.system_stream = self.audio.open(
                    format=self.format,
                    channels=self.channels,
                    rate=self.sample_rate,
                    input=True,
                    input_device_index=self.system_device_id,
                    frames_per_buffer=self.chunk_size,
                    stream_callback=None
                )
                print(f"✅ System audio stream initialized (device: {self.system_device_id})")
            
            except Exception as e:
                print(f"❌ Failed to initialize system audio stream: {e}")
                self.system_stream = None
        else:
            print("⚠️  No system audio device configured")
    
    def _close_streams(self):
        """Close audio streams"""
        if self.mic_stream:
            try:
                self.mic_stream.stop_stream()
                self.mic_stream.close()
            except:
                pass
            self.mic_stream = None
        
        if self.system_stream:
            try:
                self.system_stream.stop_stream()
                self.system_stream.close()
            except:
                pass
            self.system_stream = None
    
    def _save_audio_buffers(self, session_dir, chunk_count):
        """Save current audio buffers to files"""
        timestamp = int(time.time())
        
        # Save microphone buffer
        if self.mic_buffer:
            mic_filename = f"{session_dir}/mic_{timestamp}_{chunk_count}.wav"
            self._save_buffer_to_file(self.mic_buffer, mic_filename)
        
        # Save system buffer
        if self.system_buffer:
            system_filename = f"{session_dir}/system_{timestamp}_{chunk_count}.wav"
            self._save_buffer_to_file(self.system_buffer, system_filename)
    
    def _save_buffer_to_file(self, buffer, filename):
        """Save audio buffer to WAV file"""
        try:
            with wave.open(filename, 'wb') as wf:
                wf.setnchannels(self.channels)
                wf.setsampwidth(self.audio.get_sample_size(self.format))
                wf.setframerate(self.sample_rate)
                wf.writeframes(b''.join(buffer))
        except Exception as e:
            print(f"Error saving audio file {filename}: {e}")
    
    def get_audio_level(self, audio_data):
        """Calculate audio level from raw audio data"""
        if not audio_data:
            return 0.0
        
        try:
            # Convert bytes to numpy array
            audio_array = np.frombuffer(audio_data, dtype=np.int16)
            
            # Calculate RMS (Root Mean Square) level
            rms = np.sqrt(np.mean(audio_array.astype(np.float32) ** 2))
            
            # Normalize to 0-1 range (assuming 16-bit audio)
            normalized_level = min(rms / 32767.0, 1.0)
            
            return normalized_level
        
        except Exception as e:
            print(f"Error calculating audio level: {e}")
            return 0.0
    
    def cleanup(self):
        """Clean up resources"""
        self.stop_recording()
        if self.audio:
            self.audio.terminate()

# Test function
def test_audio_capture():
    """Test the audio capture functionality"""
    print("Testing AudioCapture...")
    
    capture = AudioCapture()
    
    try:
        input_devices, output_devices = capture.list_devices()
        print(f"Found {len(input_devices)} input devices")
        print(f"Found {len(output_devices)} output devices")
        
        if input_devices:
            print("✅ AudioCapture module is working")
        else:
            print("❌ No input devices found")
    
    finally:
        capture.cleanup()

if __name__ == "__main__":
    test_audio_capture()
