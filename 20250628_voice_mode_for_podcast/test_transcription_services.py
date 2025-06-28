#!/usr/bin/env python3
"""
Test script for transcription services

This script tests both Whisper and AssemblyAI transcription services
to ensure they work correctly with the new abstraction.
"""

import os
import sys
import time
import logging
from pathlib import Path

# Add src directory to path
sys.path.append(str(Path(__file__).parent / 'src'))

from transcription_service import get_transcription_service, TranscriptionServiceFactory

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def test_whisper_service():
    """Test Whisper transcription service"""
    logger.info("🧪 Testing Whisper transcription service...")
    
    try:
        # Create Whisper service
        config = {
            'model_name': 'base',  # Use base model for faster testing
            'language': 'en'
        }
        
        service = get_transcription_service('whisper', config)
        logger.info("✅ Whisper service initialized successfully")
        
        # Test with a sample audio file (if available)
        test_audio_files = [
            'audio_samples/session_20250628_022058/microphone_audio.wav',
            'audio_samples/session_20250628_020736/microphone_audio.wav'
        ]
        
        for audio_file in test_audio_files:
            if os.path.exists(audio_file):
                logger.info(f"🎵 Testing with audio file: {audio_file}")
                
                start_time = time.time()
                result = service.transcribe_file(audio_file)
                processing_time = time.time() - start_time
                
                if result:
                    logger.info(f"📝 Whisper result: '{result.text[:100]}...'")
                    logger.info(f"⏱️  Processing time: {processing_time:.2f}s")
                    logger.info(f"🎯 Confidence: {result.confidence:.2f}")
                    logger.info(f"🌍 Language: {result.language}")
                    break
                else:
                    logger.warning(f"❌ No result from Whisper for {audio_file}")
            else:
                logger.debug(f"⚠️  Audio file not found: {audio_file}")
        
        # Get stats
        stats = service.get_stats()
        logger.info(f"📊 Whisper stats: {stats}")
        
        # Cleanup
        service.cleanup()
        logger.info("🧹 Whisper service cleaned up")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Whisper test failed: {e}")
        return False


def test_assemblyai_service():
    """Test AssemblyAI transcription service"""
    logger.info("🧪 Testing AssemblyAI transcription service...")
    
    try:
        # Create AssemblyAI service
        config = {
            'speech_model': 'best',
            'sample_rate': 16000,
            'format_turns': True
        }
        
        service = get_transcription_service('assemblyai', config)
        logger.info("✅ AssemblyAI service initialized successfully")
        
        # Test with a sample audio file (if available)
        test_audio_files = [
            'audio_samples/session_20250628_022058/microphone_audio.wav',
            'audio_samples/session_20250628_020736/microphone_audio.wav'
        ]
        
        for audio_file in test_audio_files:
            if os.path.exists(audio_file):
                logger.info(f"🎵 Testing with audio file: {audio_file}")
                
                start_time = time.time()
                result = service.transcribe_file(audio_file)
                processing_time = time.time() - start_time
                
                if result:
                    logger.info(f"📝 AssemblyAI result: '{result.text[:100]}...'")
                    logger.info(f"⏱️  Processing time: {processing_time:.2f}s")
                    logger.info(f"🎯 Confidence: {result.confidence:.2f}")
                    logger.info(f"🌍 Language: {result.language}")
                    break
                else:
                    logger.warning(f"❌ No result from AssemblyAI for {audio_file}")
            else:
                logger.debug(f"⚠️  Audio file not found: {audio_file}")
        
        # Test streaming capability
        if service.supports_streaming():
            logger.info("🌊 AssemblyAI supports streaming!")
        else:
            logger.warning("⚠️  AssemblyAI streaming not supported")
        
        # Get stats
        stats = service.get_stats()
        logger.info(f"📊 AssemblyAI stats: {stats}")
        
        # Cleanup
        service.cleanup()
        logger.info("🧹 AssemblyAI service cleaned up")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ AssemblyAI test failed: {e}")
        return False


def test_factory():
    """Test the transcription service factory"""
    logger.info("🧪 Testing TranscriptionServiceFactory...")
    
    try:
        # Test available providers
        providers = TranscriptionServiceFactory.get_available_providers()
        logger.info(f"📋 Available providers: {providers}")
        
        # Test creating each provider
        for provider in providers:
            try:
                service = TranscriptionServiceFactory.create_service(provider)
                logger.info(f"✅ Successfully created {provider} service")
                service.cleanup()
            except Exception as e:
                logger.error(f"❌ Failed to create {provider} service: {e}")
                return False
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Factory test failed: {e}")
        return False


def test_url_transcription():
    """Test transcription with a public URL (AssemblyAI only)"""
    logger.info("🧪 Testing URL transcription with AssemblyAI...")
    
    try:
        service = get_transcription_service('assemblyai')
        
        # Test with AssemblyAI's sample audio
        test_url = "https://assembly.ai/wildfires.mp3"
        logger.info(f"🌐 Testing with URL: {test_url}")
        
        start_time = time.time()
        result = service.transcribe_file(test_url)
        processing_time = time.time() - start_time
        
        if result:
            logger.info(f"📝 URL transcription result: '{result.text[:200]}...'")
            logger.info(f"⏱️  Processing time: {processing_time:.2f}s")
            logger.info(f"🎯 Confidence: {result.confidence:.2f}")
        else:
            logger.warning("❌ No result from URL transcription")
            return False
        
        service.cleanup()
        return True
        
    except Exception as e:
        logger.error(f"❌ URL transcription test failed: {e}")
        return False


def main():
    """Run all tests"""
    logger.info("🚀 Starting transcription service tests...")
    
    # Check if API key is available
    api_key = os.getenv('API_KEY')
    if not api_key:
        logger.warning("⚠️  No API_KEY found in environment. AssemblyAI tests may fail.")
    
    tests = [
        ("Factory", test_factory),
        ("Whisper", test_whisper_service),
        ("AssemblyAI", test_assemblyai_service),
        ("URL Transcription", test_url_transcription)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        logger.info(f"\n{'='*50}")
        logger.info(f"Running {test_name} test...")
        logger.info(f"{'='*50}")
        
        try:
            results[test_name] = test_func()
        except Exception as e:
            logger.error(f"❌ {test_name} test crashed: {e}")
            results[test_name] = False
    
    # Summary
    logger.info(f"\n{'='*50}")
    logger.info("📊 TEST SUMMARY")
    logger.info(f"{'='*50}")
    
    passed = 0
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        logger.info(f"{test_name}: {status}")
        if result:
            passed += 1
    
    logger.info(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All tests passed!")
        return 0
    else:
        logger.error("💥 Some tests failed!")
        return 1


if __name__ == "__main__":
    exit(main())
