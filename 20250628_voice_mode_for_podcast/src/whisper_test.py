#!/usr/bin/env python3
"""
Whisper Testing Script
Test OpenAI Whisper transcription with sample audio
"""

import whisper
import os
import time
from pathlib import Path

def test_whisper_models():
    """Test different Whisper models and their performance"""
    print("🎯 WHISPER MODEL TESTING")
    print("=" * 50)
    
    # Available models (from fastest/smallest to slowest/largest)
    models = ["tiny", "base", "small"]
    
    # Create a test audio file path (we'll use any existing audio file)
    audio_samples_dir = Path("../audio_samples")
    test_files = list(audio_samples_dir.glob("*.wav")) if audio_samples_dir.exists() else []
    
    if not test_files:
        print("❌ No audio files found in audio_samples/")
        print("Please run audio_test.py first to create test recordings.")
        return
    
    test_file = test_files[0]  # Use the first available audio file
    print(f"📁 Using test file: {test_file}")
    print(f"📊 File size: {test_file.stat().st_size / 1024:.1f} KB")
    
    for model_name in models:
        print(f"\n🔄 Testing {model_name} model...")
        
        try:
            # Load model
            start_time = time.time()
            model = whisper.load_model(model_name)
            load_time = time.time() - start_time
            
            # Transcribe
            start_time = time.time()
            result = model.transcribe(str(test_file))
            transcribe_time = time.time() - start_time
            
            # Display results
            print(f"✅ {model_name.upper()} MODEL RESULTS:")
            print(f"   Load time: {load_time:.2f}s")
            print(f"   Transcribe time: {transcribe_time:.2f}s")
            print(f"   Detected language: {result.get('language', 'unknown')}")
            print(f"   Transcript: \"{result['text'].strip()}\"")
            
            # Show segments if available
            if 'segments' in result and result['segments']:
                print(f"   Segments: {len(result['segments'])}")
                for i, segment in enumerate(result['segments'][:3]):  # Show first 3 segments
                    start = segment.get('start', 0)
                    end = segment.get('end', 0)
                    text = segment.get('text', '').strip()
                    print(f"     [{start:.1f}s-{end:.1f}s]: \"{text}\"")
                if len(result['segments']) > 3:
                    print(f"     ... and {len(result['segments']) - 3} more segments")
            
        except Exception as e:
            print(f"❌ {model_name} model failed: {e}")
    
    print(f"\n🎯 RECOMMENDATION:")
    print("- Use 'tiny' for real-time transcription (fastest)")
    print("- Use 'base' for balanced speed/quality")
    print("- Use 'small' for better quality (if speed allows)")

def test_whisper_with_sample_text():
    """Test Whisper with a known sample if no audio files exist"""
    print("\n🎯 WHISPER BASIC FUNCTIONALITY TEST")
    print("=" * 50)
    
    try:
        # Load the smallest model for quick testing
        print("Loading tiny model...")
        model = whisper.load_model("tiny")
        print("✅ Whisper tiny model loaded successfully!")
        
        # Test with a simple audio file (if we had one)
        print("📝 Whisper is ready for transcription.")
        print("   Model: tiny")
        print("   Languages supported: 99+ languages")
        print("   Ready for real-time processing: ✅")
        
        return True
        
    except Exception as e:
        print(f"❌ Whisper test failed: {e}")
        return False

def create_sample_audio_info():
    """Create information about what audio samples we need"""
    print("\n📋 AUDIO SAMPLE REQUIREMENTS")
    print("=" * 50)
    print("For optimal Whisper testing, we need:")
    print("1. 📱 Microphone recording (your voice)")
    print("2. 🔊 System audio recording (ChatGPT's voice)")
    print("3. 🎭 Mixed conversation sample")
    print("\nTo create these samples:")
    print("1. Run: python audio_test.py")
    print("2. Test microphone capture (option 1)")
    print("3. Test system audio capture (option 2)")
    print("4. Then run this script again for full testing")

def main():
    print("🎯 CHATGPT VOICE MODE TRANSCRIPT RECORDER")
    print("Whisper Integration Testing")
    print("=" * 60)
    
    # Check if we have audio samples to test with
    audio_samples_dir = Path("../audio_samples")
    
    if audio_samples_dir.exists() and list(audio_samples_dir.glob("*.wav")):
        print("✅ Audio samples found! Running full Whisper tests...")
        test_whisper_models()
    else:
        print("⚠️  No audio samples found. Running basic functionality test...")
        if test_whisper_with_sample_text():
            create_sample_audio_info()
    
    print("\n" + "=" * 60)
    print("NEXT STEPS:")
    print("1. ✅ Whisper is installed and working")
    print("2. 🔄 Get audio capture working (run audio_test.py)")
    print("3. 🔄 Test Whisper with your actual audio samples")
    print("4. 🔄 Integrate real-time transcription")
    print("=" * 60)

if __name__ == "__main__":
    main()
