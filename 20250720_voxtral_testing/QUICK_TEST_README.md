# Voxtral Mini 3B Quick Test

This document describes how to run the minimal quick test to verify that the Voxtral Mini 3B model and basic functionalities are working correctly.

## Purpose

The quick test verifies:
1. ✅ vLLM server can start and serve the Voxtral Mini 3B model
2. ✅ Model loads correctly and responds to requests
3. ✅ Basic transcription functionality works (speech-to-text)
4. ✅ Basic audio understanding functionality works (audio Q&A)

## Two Testing Approaches

We provide two testing scripts:

### 1. Automatic Test (`quick_test.py`)
- Automatically starts/stops the vLLM server
- Good for complete end-to-end testing
- Takes longer (model download + startup time)

### 2. Manual Test (`manual_server_test.py`)
- Assumes you start the server manually first
- Faster for repeated testing
- Better for development/debugging

## Prerequisites

1. **Environment Setup**: Ensure you have completed Phase 1 & 2 from RFD-000:
   - Virtual environment created with `uv`
   - vLLM with audio support installed
   - mistral-common with audio dependencies installed
   - All development dependencies installed

2. **Hardware Requirements**:
   - GPU with ~9.5 GB of GPU RAM
   - CUDA-compatible GPU for optimal performance

3. **Network Access**: 
   - Internet connection for downloading test audio files from HuggingFace

## Running the Tests

### Option 1: Automatic Test (Recommended for first run)
```bash
# From the project root directory
cd 20250720_voxtral_testing
python quick_test.py
```

This will:
- Check if a server is already running
- Start the vLLM server if needed (downloads model on first run)
- Run all tests
- Stop the server if it was started by the test

### Option 2: Manual Test (Faster for development)

First, start the vLLM server manually:
```bash
vllm serve mistralai/Voxtral-Mini-3B-2507 \
    --tokenizer_mode=mistral \
    --config_format=mistral \
    --load_format=mistral \
    --port=8000 \
    --host=localhost
```

Then run the test:
```bash
python manual_server_test.py
```

### Option 3: Using the Server Manager
```bash
# Start server
python examples/server_manager.py start

# Run manual test
python manual_server_test.py

# Stop server when done
python examples/server_manager.py stop
```

## What the Test Does

### Test 1: Server Startup
- Starts the vLLM server with Voxtral Mini 3B model
- Waits up to 3 minutes for the server to become healthy
- Verifies the server responds to health checks

### Test 2: Model Loading
- Confirms the model is loaded and accessible
- Retrieves and displays the model name
- Validates basic server communication

### Test 3: Basic Transcription
- Downloads a sample audio file from HuggingFace
- Performs speech-to-text transcription
- Validates the transcription output
- Measures processing time

### Test 4: Basic Audio Understanding
- Uses the same sample audio file
- Asks a simple question about the audio content
- Validates the AI's understanding response
- Measures processing time

## Expected Output

### Successful Test Run
```
🚀 Starting Voxtral Mini 3B Quick Test
============================================================

📋 Running: Server Startup
----------------------------------------
Starting vLLM server...
Waiting for server to become healthy (this may take a few minutes)...
✅ Server started successfully and is healthy
✅ Server Startup: PASSED

📋 Running: Model Loading
----------------------------------------
Testing model loading and basic health check...
✅ Model loaded successfully: mistralai/Voxtral-Mini-3B-2507
✅ Model Loading: PASSED

📋 Running: Basic Transcription
----------------------------------------
Testing transcription with sample audio: https://huggingface.co/datasets/hf-internal-testing/dummy-audio-samples/resolve/main/obama.mp3
✅ Transcription successful!
   Content: [Transcribed text content]...
   Language: en
   Processing time: X.XXs
✅ Basic Transcription: PASSED

📋 Running: Basic Audio Understanding
----------------------------------------
Testing audio understanding with sample question...
✅ Audio understanding successful!
   Question: What is this audio about?
   Answer: [AI's understanding response]...
   Processing time: X.XXs
✅ Basic Audio Understanding: PASSED

============================================================
📊 QUICK TEST RESULTS SUMMARY
============================================================
Server Startup.................. ✅ PASSED
Model Loading................... ✅ PASSED
Basic Transcription............. ✅ PASSED
Basic Audio Understanding....... ✅ PASSED
------------------------------------------------------------
🎉 ALL TESTS PASSED!
✅ Voxtral Mini 3B setup is working correctly
✅ Ready to proceed with FastAPI server implementation
📋 Model: mistralai/Voxtral-Mini-3B-2507
📋 Transcription time: X.XXs
📋 Understanding time: X.XXs
============================================================
🧹 Cleaning up...
✅ Server stopped
```

## Troubleshooting

### Common Issues

1. **GPU Memory Error**
   - Ensure you have at least 9.5 GB of free GPU RAM
   - Close other GPU-intensive applications
   - Check with `nvidia-smi`

2. **Server Startup Timeout**
   - Model download may take time on first run
   - Check internet connection
   - Increase timeout if needed

3. **Audio Download Failures**
   - Verify internet connectivity
   - Check if HuggingFace is accessible
   - Firewall/proxy issues

4. **Import Errors**
   - Ensure virtual environment is activated
   - Verify all dependencies are installed
   - Check Python path configuration

### Getting Help

If the quick test fails:
1. Check the error messages in the output
2. Review the server logs for detailed error information
3. Verify your environment setup against RFD-000 Phase 1 & 2
4. Ensure hardware requirements are met

## Next Steps

Once the quick test passes successfully:
1. ✅ Your Voxtral Mini 3B setup is working correctly
2. ✅ You're ready to proceed with Phase 3: FastAPI Server Implementation
3. ✅ The comprehensive integration tests should also pass

If the test fails, resolve the issues before proceeding with the full implementation.
