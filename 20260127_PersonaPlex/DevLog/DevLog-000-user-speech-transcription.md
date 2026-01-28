# DevLog-000: Adding User Speech Transcription to PersonaPlex

## Date: 2026-01-28

## Background

### What is PersonaPlex?

PersonaPlex is NVIDIA's 7 billion parameter full-duplex conversational AI model. It enables real-time bidirectional speech conversations where the model can listen and speak simultaneously, handle interruptions, and provide natural backchanneling.

Key characteristics:
- Speech-to-speech model (processes audio directly, not text)
- Built on the Moshi architecture from Kyutai
- Uses Mimi audio codec for encoding/decoding audio at 24kHz
- Supports customizable voice and persona through voice prompts and text prompts
- Single-stage model (eliminates traditional ASR to LLM to TTS cascade)

### The Problem

PersonaPlex only generates text transcripts for the assistant's speech, not the user's speech. The current implementation:

1. User audio is captured via microphone
2. Audio is Opus-encoded and sent to server via WebSocket
3. Server decodes to PCM, then encodes to audio codes via Mimi
4. Audio codes are fed to the LM model as input conditioning
5. Model generates assistant audio tokens AND text tokens
6. Assistant audio is decoded and sent back to client
7. Assistant text tokens are decoded and sent to client for display

The user's audio is never transcribed to text. It is only used as conditioning input for the model to generate responses.

### Goal

Add real-time transcription of user speech so that the conversation transcript includes both user and assistant voice lines.

## System Specifications

The target machine has:
- 2x NVIDIA H100 80GB HBM3 GPUs
- Intel Xeon Platinum 8480+ (52 cores)
- 442 GB RAM
- CUDA 12.8

This is a powerful setup that can run large ASR models with low latency.

## Research: ASR Model Options

### Standard Whisper

OpenAI's Whisper is designed for 30-second audio chunks, not streaming. Naive implementations result in 3-10 seconds latency, which is unacceptable for real-time transcription.

### Whisper-Streaming

Academic implementation achieving 3.3 seconds latency on long-form speech. Still too slow for good user experience.

### Faster-Whisper

Uses CTranslate2 for optimized inference. Achieves approximately 300ms latency with streaming mode. Good balance of speed and accuracy.

### Whisper Large V3 Turbo

Latest Whisper variant with:
- Word Error Rate (WER): 7.75%
- Significantly faster than standard Whisper Large V3
- Model size: approximately 1.5GB
- Supports 99+ languages
- Better context understanding than smaller models

### NVIDIA NeMo Parakeet ASR

State-of-the-art ASR models developed by NVIDIA in collaboration with Suno.ai:
- Optimized specifically for NVIDIA GPUs (including H100)
- Ultra-low latency streaming support
- Multiple model sizes available
- Part of NVIDIA NeMo framework
- Best-in-class accuracy for English

## Recommendation

Given the hardware (2x H100 80GB), two options are viable:

### Option 1: NVIDIA NeMo Parakeet (Optimal for Hardware)

Pros:
- Native optimization for H100 GPUs
- State-of-the-art accuracy
- Ultra-low latency streaming
- Part of NVIDIA ecosystem (same as PersonaPlex)

Cons:
- More complex setup (NeMo framework)
- Less community examples for this specific integration

### Option 2: Whisper Large V3 Turbo via Faster-Whisper (Easier Integration)

Pros:
- Simpler integration via faster-whisper library
- Excellent accuracy (7.75 WER)
- Large community and documentation
- Proven reliability

Cons:
- Not specifically optimized for H100 (still very fast)
- Slightly higher latency than Parakeet

## Architecture Design

### Current WebSocket Protocol

Message types (first byte):
- 0x00: Handshake
- 0x01: Audio data
- 0x02: Text (assistant transcript)
- 0x03: Control messages
- 0x04: Metadata
- 0x05: Error
- 0x06: Ping

### Proposed Changes

Add new message type:
- 0x07: User text (user transcript)

### Server-Side Integration

Location: `personaplex/moshi/moshi/server.py`

The `opus_loop()` function (lines 204-243) processes user audio:

```python
async def opus_loop():
    all_pcm_data = None
    while True:
        # ... read PCM from opus_reader
        chunk = all_pcm_data[: self.frame_size]
        # This is where we can buffer audio for ASR
        codes = self.mimi.encode(chunk)
        # ... rest of processing
```

Integration points:
1. Buffer user PCM audio in `opus_loop()`
2. Run ASR model asynchronously (separate thread/process)
3. Send user transcripts via new message type 0x07
4. Ensure ASR processing does not block audio loop

### Client-Side Integration

Files to modify:
- `client/src/protocol/types.ts`: Add user text message type
- `client/src/protocol/encoder.ts`: Add decoder for 0x07 messages
- `client/src/pages/Conversation/hooks/`: Add hook for user text
- `client/src/pages/Conversation/components/TextDisplay/`: Update UI

### GPU Allocation

Current state:
- GPU 0: PersonaPlex model (approximately 40GB VRAM)
- GPU 1: Available (80GB VRAM)

Proposed:
- GPU 0: PersonaPlex model
- GPU 1: ASR model (Whisper Large V3 Turbo uses approximately 3GB)

Alternatively, both can run on GPU 0 since there is sufficient VRAM.

## Implementation Plan

1. Install ASR dependencies (faster-whisper or nemo_toolkit)
2. Add ASR model loading in ServerState.__init__()
3. Create audio buffer and transcription queue
4. Implement async transcription worker
5. Modify opus_loop() to feed audio to transcription queue
6. Add new WebSocket message type for user text
7. Update client protocol types and encoder
8. Create useUserText hook
9. Update TextDisplay component to show both speakers
10. Test end-to-end

## Open Questions (Resolved)

1. ~~Should user and assistant transcripts be visually distinguished?~~ **YES** - Chat bubble UI with blue (user) and gray (assistant) bubbles, speaker labels, and timestamps.
2. ~~What buffer size provides best accuracy vs latency tradeoff?~~ **4 seconds default**, tunable via `--asr-buffer-duration`.
3. Should we support multiple languages or English only initially? **English only for now** (hardcoded `language="en"`).
4. Should transcripts be saved to a file for later review? **Not implemented yet** - future enhancement.

## Implementation Progress

### Decision: Using Whisper Large V3 Turbo via faster-whisper

Rationale: Simpler integration, proven reliability, excellent accuracy. Can switch to Parakeet later if needed.

### Phase 1: Server-Side ASR Integration

- [x] Step 1.1: Install faster-whisper dependency
- [x] Step 1.2: Create standalone ASR test script to verify model works
- [x] Step 1.3: Add ASR model loading to ServerState
- [x] Step 1.4: Implement audio buffer and transcription queue
- [x] Step 1.5: Implement async transcription worker
- [x] Step 1.6: Modify opus_loop() to feed audio buffer
- [x] Step 1.7: Add WebSocket message type 0x07 for user text
- [x] Step 1.8: Test server-side transcription end-to-end

### Phase 2: Client-Side Integration

- [x] Step 2.1: Update protocol/types.ts with user text type
- [x] Step 2.2: Update protocol/encoder.ts to decode 0x07 messages
- [x] Step 2.3: Create useUserText hook
- [x] Step 2.4: Update TextDisplay component for dual transcripts
- [x] Step 2.5: Test client-side display

### Phase 3: Integration Testing

- [x] Step 3.1: Full end-to-end test with live audio
- [x] Step 3.2: Verify latency is acceptable
- [x] Step 3.3: Document any issues and fixes

### Phase 4: Conversational UI Enhancement

- [x] Step 4.1: Add timestamps to text messages (JSON payload format)
- [x] Step 4.2: Create unified useConversation hook for chronological message display
- [x] Step 4.3: Redesign TextDisplay with chat bubble UI
- [x] Step 4.4: Add tunable ASR parameters (buffer duration, beam size, VAD)

---

## Implementation Log

### 2026-01-28: Starting Implementation

#### Step 1.1: Install faster-whisper - COMPLETE
- Installed faster-whisper 1.2.1 via uv
- Dependencies: ctranslate2, onnxruntime, tokenizers, av

#### Step 1.2: Standalone ASR Test - COMPLETE
- Created test script: `moshi/tests/test_asr_standalone.py`
- Results on H100 GPU 1:
  - Model: Whisper large-v3-turbo (1.62GB)
  - Load time: 5.56 seconds
  - RTF (Real-Time Factor): 0.002 (500x faster than real-time)
  - Suitable for real-time: YES
- Note: VAD filter correctly detected no speech in sine wave test audio

#### Step 1.3: Add ASR Module - COMPLETE
- Created `moshi/moshi/asr.py` with ASRConfig and ASRTranscriber classes
- Key features:
  - Configurable buffer duration (default 2.0 seconds)
  - Resampling from 24kHz (PersonaPlex) to 16kHz (Whisper)
  - Thread-based worker for non-blocking transcription
  - VAD filter enabled to skip silence
  - Async callback pattern for integration with event loop

#### Step 1.4-1.7: Server Integration - COMPLETE
- Modified `moshi/moshi/server.py`:
  - Added import for ASR module
  - Added `asr` field to ServerState dataclass
  - Added ASR initialization with command-line arguments
  - Modified `opus_loop()` to feed audio to ASR
  - Added `user_text_loop()` to send transcripts via WebSocket (0x07)
  - Added ASR start/stop lifecycle management
  - Added `--enable-asr`, `--disable-asr`, `--asr-device-index` CLI args
- Fixed CUDA device context issue: ASR model loading was changing default device

#### Step 1.8: Server-Side Test - COMPLETE
- Server successfully started with ASR enabled
- Transcripts appearing in logs:
  ```
  [2P8H] user transcript: Hello, hello, how are you?
  [2P8H] user transcript: Thank you.
  [2P8H] user transcript: Good, good.
  ```

#### Step 2.1-2.4: Client-Side Integration - COMPLETE
- Updated `client/src/protocol/types.ts`: Added "user_text" message type
- Updated `client/src/protocol/encoder.ts`: Added encoder/decoder for 0x07
- Created `client/src/pages/Conversation/hooks/useUserText.ts`
- Updated `client/src/pages/Conversation/components/TextDisplay/TextDisplay.tsx`:
  - Now shows both user and assistant transcripts
  - User: blue left border with "You" label
  - Assistant: green left border with "Assistant" label
- Client build successful

#### Step 2.5: Client-Side Display Test - COMPLETE
- Server running at https://172.28.127.171:8998
- Serving updated client from `client/dist`
- User transcription working end-to-end

#### Step 3.1-3.3: Integration Testing - COMPLETE
- Full end-to-end test successful
- User and assistant transcripts both appearing
- Latency acceptable for real-time conversation

### 2026-01-28: Conversational UI Enhancement

#### Step 4.1: Add Timestamps - COMPLETE
- Modified server to send JSON payloads with timestamps for both message types:
  - `0x02` (assistant text): `{"text": "...", "timestamp": 1706...}`
  - `0x07` (user text): `{"text": "...", "timestamp": 1706...}`
- Updated client decoder with backward compatibility for plain text

#### Step 4.2: Unified Conversation Hook - COMPLETE
- Created `useConversation.ts` hook that merges user and assistant messages chronologically
- Consecutive assistant tokens are merged into single messages
- Consecutive user transcripts within 5 seconds are merged

#### Step 4.3: Chat Bubble UI - COMPLETE
- Redesigned TextDisplay component with chat bubble layout:
  - User messages: blue bubbles, right-aligned
  - Assistant messages: gray bubbles, left-aligned
  - Each message shows speaker label and HH:MM:SS timestamp
  - Latest message highlighted with ring effect
  - Empty state placeholder text

#### Step 4.4: Tunable ASR Parameters - COMPLETE
- Added new CLI arguments:
  - `--asr-buffer-duration` (default 4.0s): Longer buffer = better accuracy
  - `--asr-beam-size` (default 5): Higher = better accuracy, slower
  - `--asr-no-vad`: Disable VAD filter if sentences are cut mid-word
- Updated ASRConfig with `vad_min_silence_duration_ms` parameter

### Fork and Commit
- Forked NVIDIA/personaplex to jwt625/personaplex
- Committed all changes: `5ffb3ad`
- Repository: https://github.com/jwt625/personaplex

