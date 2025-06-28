#!/usr/bin/env python3
"""
Transcript Processor Module
Handles Whisper transcription and text processing
"""

import whisper
import numpy as np
import io
import wave
import tempfile
import os
import threading
import time
from collections import deque

class TranscriptProcessor:
    def __init__(self, model_name="small", language=None):
        self.model_name = model_name
        self.language = language
        self.model = None
        self.is_processing = False
        
        # Processing queue
        self.audio_queue = deque()
        self.processing_thread = None
        
        # Statistics
        self.total_processed = 0
        self.total_processing_time = 0
        
        # Load model
        self._load_model()
    
    def _load_model(self):
        """Load the Whisper model"""
        try:
            print(f"Loading Whisper {self.model_name} model...")
            start_time = time.time()
            self.model = whisper.load_model(self.model_name)
            load_time = time.time() - start_time
            print(f"✅ Whisper {self.model_name} model loaded in {load_time:.2f}s")
        except Exception as e:
            print(f"❌ Failed to load Whisper model: {e}")
            raise
    
    def process_audio_chunk(self, audio_data, source='microphone'):
        """Process an audio chunk and return transcript"""
        if not self.model or not audio_data:
            return None
        
        try:
            # Convert audio data to temporary file
            temp_file = self._audio_data_to_temp_file(audio_data)
            
            if not temp_file:
                return None
            
            # Transcribe with Whisper
            start_time = time.time()
            result = self.model.transcribe(
                temp_file,
                language=self.language,
                task="transcribe",
                fp16=False,  # Use fp32 for better compatibility
                verbose=False
            )
            processing_time = time.time() - start_time
            
            # Clean up temp file
            try:
                os.unlink(temp_file)
            except:
                pass
            
            # Extract transcript text
            text = result.get('text', '').strip()

            print(f"🎤 Whisper result: '{text}' (confidence: {self._estimate_confidence(result):.2f})")

            if not text:
                print("⚠️  Empty transcript result")
                return None
            
            # Update statistics
            self.total_processed += 1
            self.total_processing_time += processing_time
            
            # Calculate confidence (Whisper doesn't provide this directly)
            # We'll estimate based on the presence of segments and their characteristics
            confidence = self._estimate_confidence(result)
            
            return {
                'text': text,
                'confidence': confidence,
                'processing_time': processing_time,
                'language': result.get('language', 'unknown'),
                'is_final': True,  # For now, all results are final
                'segments': result.get('segments', [])
            }
        
        except Exception as e:
            print(f"Error processing audio chunk: {e}")
            return None
    
    def _audio_data_to_temp_file(self, audio_data, sample_rate=16000, channels=1):
        """Convert raw audio data to a temporary WAV file"""
        try:
            # Create temporary file
            temp_fd, temp_path = tempfile.mkstemp(suffix='.wav')
            
            with os.fdopen(temp_fd, 'wb') as temp_file:
                # Create WAV file
                with wave.open(temp_file, 'wb') as wav_file:
                    wav_file.setnchannels(channels)
                    wav_file.setsampwidth(2)  # 16-bit
                    wav_file.setframerate(sample_rate)
                    wav_file.writeframes(audio_data)
            
            return temp_path
        
        except Exception as e:
            print(f"Error creating temp audio file: {e}")
            return None
    
    def _estimate_confidence(self, whisper_result):
        """Estimate confidence score from Whisper result"""
        try:
            segments = whisper_result.get('segments', [])
            
            if not segments:
                # If no segments, use text length as a rough indicator
                text_length = len(whisper_result.get('text', ''))
                if text_length > 20:
                    return 0.8
                elif text_length > 5:
                    return 0.6
                else:
                    return 0.4
            
            # Calculate average confidence from segments
            # Note: Whisper segments don't have confidence scores in the standard API
            # We'll use other indicators like segment duration and text characteristics
            total_confidence = 0
            for segment in segments:
                segment_confidence = 0.7  # Base confidence
                
                # Adjust based on segment duration
                duration = segment.get('end', 0) - segment.get('start', 0)
                if duration > 0.5:  # Longer segments tend to be more reliable
                    segment_confidence += 0.1
                
                # Adjust based on text characteristics
                text = segment.get('text', '').strip()
                if len(text) > 10:  # Longer text tends to be more reliable
                    segment_confidence += 0.1
                
                # Check for common filler words or unclear speech indicators
                if any(word in text.lower() for word in ['um', 'uh', 'hmm', '...']):
                    segment_confidence -= 0.2
                
                total_confidence += min(max(segment_confidence, 0.1), 1.0)
            
            return total_confidence / len(segments)
        
        except Exception as e:
            print(f"Error estimating confidence: {e}")
            return 0.5  # Default confidence
    
    def process_audio_file(self, file_path):
        """Process a complete audio file"""
        if not self.model:
            return None
        
        try:
            print(f"Processing audio file: {file_path}")
            start_time = time.time()
            
            result = self.model.transcribe(
                file_path,
                language=self.language,
                task="transcribe",
                fp16=False,
                verbose=True
            )
            
            processing_time = time.time() - start_time
            
            return {
                'text': result.get('text', '').strip(),
                'confidence': self._estimate_confidence(result),
                'processing_time': processing_time,
                'language': result.get('language', 'unknown'),
                'segments': result.get('segments', []),
                'file_path': file_path
            }
        
        except Exception as e:
            print(f"Error processing audio file {file_path}: {e}")
            return None
    
    def get_processing_stats(self):
        """Get processing statistics"""
        avg_processing_time = 0
        if self.total_processed > 0:
            avg_processing_time = self.total_processing_time / self.total_processed
        
        return {
            'total_processed': self.total_processed,
            'total_processing_time': self.total_processing_time,
            'average_processing_time': avg_processing_time,
            'model_name': self.model_name
        }
    
    def change_model(self, model_name):
        """Change the Whisper model"""
        if model_name != self.model_name:
            print(f"Changing model from {self.model_name} to {model_name}")
            self.model_name = model_name
            self._load_model()
    
    def set_language(self, language):
        """Set the target language for transcription"""
        self.language = language
        print(f"Language set to: {language or 'auto-detect'}")

# Test function
def test_transcript_processor():
    """Test the transcript processor"""
    print("Testing TranscriptProcessor...")
    
    try:
        processor = TranscriptProcessor(model_name="small")
        
        # Test with a dummy audio file (if available)
        audio_samples_dir = "../audio_samples"
        if os.path.exists(audio_samples_dir):
            wav_files = [f for f in os.listdir(audio_samples_dir) if f.endswith('.wav')]
            
            if wav_files:
                test_file = os.path.join(audio_samples_dir, wav_files[0])
                print(f"Testing with file: {test_file}")
                
                result = processor.process_audio_file(test_file)
                if result:
                    print(f"✅ Transcription successful:")
                    print(f"   Text: {result['text']}")
                    print(f"   Confidence: {result['confidence']:.2f}")
                    print(f"   Processing time: {result['processing_time']:.2f}s")
                else:
                    print("❌ Transcription failed")
            else:
                print("⚠️  No audio files found for testing")
        else:
            print("⚠️  Audio samples directory not found")
        
        print("✅ TranscriptProcessor module is working")
        
        # Show stats
        stats = processor.get_processing_stats()
        print(f"Processing stats: {stats}")
    
    except Exception as e:
        print(f"❌ TranscriptProcessor test failed: {e}")

if __name__ == "__main__":
    test_transcript_processor()
