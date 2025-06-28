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
        
        # Continuous audio buffer (never cleared)
        self.continuous_buffer = deque()
        self.last_processed_chunk_index = 0  # Track what we've already sent to Whisper

        # VAD state tracking
        self.current_silence_duration = 0.0
        self.current_speech_duration = 0.0
        self.is_in_speech = False
        self.last_process_time = time.time()

        # Chunk tracking
        self.total_chunks_received = 0

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
        # Always add to continuous buffer (NEVER cleared)
        self.continuous_buffer.append(audio_data)
        self.total_chunks_received += 1
        
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
            # Extract NEW audio chunks that haven't been sent to Whisper yet
            new_audio_chunk = self._extract_new_audio_for_processing()

            if new_audio_chunk:
                # Log processing decision
                new_chunks_count = self.total_chunks_received - self.last_processed_chunk_index
                duration = new_chunks_count * self.chunk_duration
                print(f"🎯 Processing NEW audio: {duration:.1f}s ({new_chunks_count} chunks), "
                      f"speech: {self.current_speech_duration:.1f}s, "
                      f"silence: {self.current_silence_duration:.1f}s")

                # Update tracking - mark these chunks as processed
                self.last_processed_chunk_index = self.total_chunks_received

                # Reset VAD counters for next detection cycle
                self._reset_vad_counters()

                return True, new_audio_chunk
        
        return False, None
    
    def _should_process_now(self):
        """Determine if we should process the current buffer"""

        # Get current unprocessed buffer duration
        unprocessed_chunks = self.total_chunks_received - self.last_processed_chunk_index
        buffer_duration = unprocessed_chunks * self.chunk_duration
        
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
    
    def _extract_new_audio_for_processing(self):
        """Extract only NEW audio chunks that haven't been sent to Whisper"""
        if self.last_processed_chunk_index >= self.total_chunks_received:
            return None  # No new audio

        # Get only the NEW chunks since last processing
        new_chunks = list(self.continuous_buffer)[self.last_processed_chunk_index:]

        if not new_chunks:
            return None

        # Combine new chunks into audio data
        new_audio_chunk = b''.join(new_chunks)
        return new_audio_chunk

    def _reset_vad_counters(self):
        """Reset only VAD counters, keep continuous buffer intact"""
        self.current_silence_duration = 0.0
        self.current_speech_duration = 0.0
        self.is_in_speech = False
        self.last_process_time = time.time()

    def _reset_state(self):
        """Legacy method - now just calls _reset_vad_counters"""
        self._reset_vad_counters()
    
    def force_process(self):
        """Force process any remaining NEW audio (for end of recording)"""
        # Get any unprocessed audio chunks
        final_chunk = self._extract_new_audio_for_processing()

        if final_chunk:
            new_chunks_count = self.total_chunks_received - self.last_processed_chunk_index
            duration = new_chunks_count * self.chunk_duration
            print(f"🔚 Force processing final chunk: {duration:.1f}s ({new_chunks_count} chunks)")

            # Mark all chunks as processed
            self.last_processed_chunk_index = self.total_chunks_received
            return final_chunk

        return None
    
    def get_stats(self):
        """Get current chunker statistics"""
        total_duration = self.total_chunks_received * self.chunk_duration
        unprocessed_chunks = self.total_chunks_received - self.last_processed_chunk_index
        unprocessed_duration = unprocessed_chunks * self.chunk_duration

        return {
            'total_duration': total_duration,
            'total_chunks': self.total_chunks_received,
            'processed_chunks': self.last_processed_chunk_index,
            'unprocessed_chunks': unprocessed_chunks,
            'unprocessed_duration': unprocessed_duration,
            'speech_duration': self.current_speech_duration,
            'silence_duration': self.current_silence_duration,
            'is_in_speech': self.is_in_speech
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
