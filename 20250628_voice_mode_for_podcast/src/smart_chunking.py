#!/usr/bin/env python3
"""
Smart Audio Chunking with Voice Activity Detection (VAD)
Prevents word cutting by detecting natural speech pauses
"""

import time
from collections import deque

class SmartAudioChunker:
    def __init__(self, 
                 silence_threshold=0.01,      # Audio level below this = silence
                 min_silence_duration=0.8,    # Minimum silence to trigger processing (800ms)
                 min_speech_duration=1.0,     # Minimum speech before considering processing
                 max_chunk_duration=15.0,     # Maximum chunk length (fallback)
                 sample_rate=16000,
                 chunk_size=1024):
        
        # VAD parameters
        self.silence_threshold = silence_threshold
        self.min_silence_duration = min_silence_duration
        self.min_speech_duration = min_speech_duration
        self.max_chunk_duration = max_chunk_duration
        
        # Audio parameters
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.chunk_duration = chunk_size / sample_rate  # ~0.064 seconds per chunk
        
        # State tracking
        self.speech_buffer = deque()
        self.current_silence_duration = 0.0
        self.current_speech_duration = 0.0
        self.is_in_speech = False
        self.last_process_time = time.time()
        
        print(f"🎤 Smart chunker initialized:")
        print(f"   Silence threshold: {silence_threshold}")
        print(f"   Min silence duration: {min_silence_duration}s")
        print(f"   Min speech duration: {min_speech_duration}s")
        print(f"   Max chunk duration: {max_chunk_duration}s")
    
    def add_audio_chunk(self, audio_data, audio_level):
        """
        Add audio chunk and return transcript data if ready to process
        Returns: (should_process, audio_chunk_for_transcription)
        """
        # Add to buffer
        self.speech_buffer.append(audio_data)
        
        # Determine if this chunk contains speech
        is_speech = audio_level > self.silence_threshold
        
        # Update state durations
        if is_speech:
            self.current_speech_duration += self.chunk_duration
            self.current_silence_duration = 0.0
            if not self.is_in_speech:
                self.is_in_speech = True
                print(f"🗣️  Speech started (level: {audio_level:.3f})")
        else:
            self.current_silence_duration += self.chunk_duration
            if self.is_in_speech and self.current_silence_duration > 0.2:  # 200ms of silence
                print(f"🤫 Silence detected (duration: {self.current_silence_duration:.1f}s)")
        
        # Check if we should process
        should_process = self._should_process_now()
        
        if should_process:
            # Get audio data for transcription
            audio_chunk = b''.join(self.speech_buffer)
            
            # Log processing decision
            buffer_duration = len(self.speech_buffer) * self.chunk_duration
            print(f"🎯 Processing chunk: {buffer_duration:.1f}s audio, "
                  f"speech: {self.current_speech_duration:.1f}s, "
                  f"silence: {self.current_silence_duration:.1f}s")
            
            # Reset for next chunk
            self._reset_state()
            
            return True, audio_chunk
        
        return False, None
    
    def _should_process_now(self):
        """Determine if we should process the current buffer"""
        
        # Get current buffer duration
        buffer_duration = len(self.speech_buffer) * self.chunk_duration
        
        # Condition 1: Natural pause detected
        if (self.is_in_speech and 
            self.current_speech_duration >= self.min_speech_duration and
            self.current_silence_duration >= self.min_silence_duration):
            print(f"✅ Natural pause detected")
            return True
        
        # Condition 2: Buffer getting too long (fallback)
        if buffer_duration >= self.max_chunk_duration:
            print(f"⏰ Max duration reached ({buffer_duration:.1f}s)")
            return True
        
        # Condition 3: Long silence after any speech (cleanup)
        if (self.current_speech_duration > 0 and 
            self.current_silence_duration > self.min_silence_duration * 2):
            print(f"🧹 Long silence cleanup")
            return True
        
        return False
    
    def _reset_state(self):
        """Reset state after processing"""
        self.speech_buffer.clear()
        self.current_silence_duration = 0.0
        self.current_speech_duration = 0.0
        self.is_in_speech = False
        self.last_process_time = time.time()
    
    def force_process(self):
        """Force process current buffer (for end of recording)"""
        if len(self.speech_buffer) > 0:
            audio_chunk = b''.join(self.speech_buffer)
            buffer_duration = len(self.speech_buffer) * self.chunk_duration
            print(f"🔚 Force processing final chunk: {buffer_duration:.1f}s")
            self._reset_state()
            return audio_chunk
        return None
    
    def get_stats(self):
        """Get current chunker statistics"""
        buffer_duration = len(self.speech_buffer) * self.chunk_duration
        return {
            'buffer_duration': buffer_duration,
            'speech_duration': self.current_speech_duration,
            'silence_duration': self.current_silence_duration,
            'is_in_speech': self.is_in_speech,
            'buffer_chunks': len(self.speech_buffer)
        }

# Test function
def test_smart_chunker():
    """Test the smart chunker with simulated audio levels"""
    chunker = SmartAudioChunker(
        silence_threshold=0.02,
        min_silence_duration=0.5,
        min_speech_duration=1.0
    )
    
    # Simulate audio pattern: silence -> speech -> silence -> speech -> silence
    test_pattern = [
        (0.001, 10),  # 10 chunks of silence
        (0.05, 30),   # 30 chunks of speech
        (0.001, 8),   # 8 chunks of silence (should trigger processing)
        (0.08, 25),   # 25 chunks of speech
        (0.001, 12),  # 12 chunks of silence (should trigger processing)
    ]
    
    chunk_count = 0
    for audio_level, num_chunks in test_pattern:
        for i in range(num_chunks):
            # Simulate audio data
            fake_audio = b'x' * 1024
            
            should_process, audio_chunk = chunker.add_audio_chunk(fake_audio, audio_level)
            
            if should_process:
                print(f"📝 Would transcribe chunk of {len(audio_chunk)} bytes")
            
            chunk_count += 1
    
    # Force process remaining
    final_chunk = chunker.force_process()
    if final_chunk:
        print(f"📝 Final chunk: {len(final_chunk)} bytes")
    
    print(f"✅ Test complete: processed {chunk_count} chunks")

if __name__ == "__main__":
    test_smart_chunker()
