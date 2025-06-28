# ChatGPT Voice Mode Transcript Recorder - Implementation Plan

## Project Overview
Build a local transcript recorder for ChatGPT voice mode conversations using OpenAI Whisper, with real-time display and automatic saving.

## Architecture
- **Backend**: Python Flask server
- **Audio Capture**: pyaudio for mic input + system audio output
- **Transcription**: OpenAI Whisper (local)
- **Frontend**: Flask templates with JavaScript for real-time updates
- **Storage**: SQLite + JSON backup files

---

## Phase 1: Foundation & Audio Testing 🎯 **PRIORITY**

### 1.1 Development Environment Setup
- [ ] Initialize Python virtual environment
- [ ] Install core dependencies:
  - [ ] Flask
  - [ ] pyaudio
  - [ ] openai-whisper
  - [ ] pydub
  - [ ] sqlite3 (built-in)
  - [ ] websockets (flask-socketio)
- [ ] Create basic project structure
- [ ] Set up requirements.txt

### 1.2 Audio Capture Testing 🔥 **CRITICAL**
- [ ] **Test microphone input capture**
  - [ ] List available audio input devices
  - [ ] Record 10-second mic sample
  - [ ] Save as WAV file and verify playback
  - [ ] Test different sample rates (16kHz, 44.1kHz)
- [ ] **Test system audio output capture**
  - [ ] Research OS-specific solutions:
    - [ ] macOS: BlackHole virtual audio device
    - [ ] Windows: VB-Cable or Stereo Mix
    - [ ] Linux: PulseAudio loopback
  - [ ] Install and configure virtual audio device
  - [ ] Capture system audio while playing test audio
  - [ ] Verify captured audio quality
- [ ] **Dual audio stream testing**
  - [ ] Capture both mic and system audio simultaneously
  - [ ] Test for audio interference/feedback
  - [ ] Verify separate channel processing

---

## Phase 2: Core Transcription Engine

### 2.1 Whisper Integration
- [ ] Install and test Whisper models:
  - [ ] Test `tiny` model (fastest, lower quality)
  - [ ] Test `base` model (balanced)
  - [ ] Test `small` model (better quality)
- [ ] Implement real-time audio chunking
- [ ] Test transcription accuracy with conversational audio
- [ ] Implement confidence score monitoring
- [ ] Add language detection (if needed)

### 2.2 Audio Processing Pipeline
- [ ] Implement audio preprocessing:
  - [ ] Noise reduction
  - [ ] Volume normalization
  - [ ] Format conversion (to 16kHz mono for Whisper)
- [ ] Create audio buffer management
- [ ] Implement speaker detection logic
- [ ] Add audio quality monitoring

---

## Phase 3: Flask Web Interface

### 3.1 Basic Flask App
- [ ] Create Flask app structure
- [ ] Set up basic routes:
  - [ ] `/` - Main transcript display page
  - [ ] `/api/start` - Start recording
  - [ ] `/api/stop` - Stop recording
  - [ ] `/api/status` - Get current status
- [ ] Implement WebSocket connection for real-time updates
- [ ] Create basic HTML template

### 3.2 Real-time Transcript Display
- [ ] Design transcript UI:
  - [ ] Conversation turns (user vs ChatGPT)
  - [ ] Timestamps
  - [ ] Confidence indicators
  - [ ] Audio level meters
- [ ] Implement JavaScript for real-time updates
- [ ] Add transcript quality monitoring display
- [ ] Create conversation session management

---

## Phase 4: Storage & Persistence

### 4.1 Database Setup
- [ ] Design SQLite schema:
  - [ ] `sessions` table (conversation sessions)
  - [ ] `transcripts` table (individual transcript segments)
  - [ ] `audio_files` table (optional audio storage)
- [ ] Implement database models
- [ ] Create database initialization script

### 4.2 Saving Logic
- [ ] **Auto-save every conversation round**
  - [ ] Detect conversation turn completion
  - [ ] Save transcript segment to database
  - [ ] Update session metadata
- [ ] **Periodic saves during long monologues**
  - [ ] Implement 60-second timer
  - [ ] Save partial transcripts
  - [ ] Handle transcript continuation
- [ ] **Backup to JSON files**
  - [ ] Export sessions to JSON
  - [ ] Implement auto-backup on session end

---

## Phase 5: Advanced Features

### 5.1 Conversation Management
- [ ] Session organization:
  - [ ] Start/end session detection
  - [ ] Session naming and tagging
  - [ ] Session history browser
- [ ] Speaker identification:
  - [ ] Distinguish user vs ChatGPT
  - [ ] Voice pattern recognition
  - [ ] Manual speaker correction

### 5.2 Quality & Monitoring
- [ ] Transcript quality metrics:
  - [ ] Whisper confidence scores
  - [ ] Audio level monitoring
  - [ ] Real-time quality indicators
- [ ] Error handling:
  - [ ] Audio device disconnection
  - [ ] Whisper processing errors
  - [ ] Storage failures

### 5.3 Export & Integration
- [ ] Export formats:
  - [ ] Plain text
  - [ ] Markdown with timestamps
  - [ ] JSON with metadata
  - [ ] PDF reports
- [ ] Integration options:
  - [ ] Copy to clipboard
  - [ ] Save to specific folders
  - [ ] API for external tools

---

## Phase 6: Testing & Optimization

### 6.1 Performance Testing
- [ ] Test with long conversations (30+ minutes)
- [ ] Memory usage optimization
- [ ] CPU usage monitoring
- [ ] Storage space management

### 6.2 Reliability Testing
- [ ] Test audio device switching
- [ ] Test system sleep/wake cycles
- [ ] Test network interruptions (if applicable)
- [ ] Test crash recovery

### 6.3 User Experience
- [ ] Keyboard shortcuts
- [ ] System tray integration (optional)
- [ ] Configuration settings
- [ ] User documentation

---

## Technical Notes

### Audio Capture Considerations
- **Sample Rate**: 16kHz for Whisper compatibility
- **Channels**: Mono for processing, but capture stereo for separation
- **Buffer Size**: Balance between latency and processing efficiency
- **Format**: 16-bit PCM for Whisper input

### Whisper Model Selection
- **tiny**: ~39 MB, fastest, good for real-time
- **base**: ~74 MB, better accuracy, still fast enough
- **small**: ~244 MB, best balance for quality

### Flask Architecture
- **Main Thread**: Flask web server
- **Audio Thread**: Continuous audio capture
- **Processing Thread**: Whisper transcription
- **WebSocket Thread**: Real-time updates to frontend

---

## Getting Started
1. Start with Phase 1.2 (Audio Capture Testing) - this is the critical foundation
2. Verify both mic and system audio work before proceeding
3. Test on your specific OS and audio setup
4. Build incrementally and test each component

## Success Criteria
- ✅ Reliable dual audio capture (mic + system)
- ✅ Real-time transcription with <2 second delay
- ✅ Auto-save every conversation round
- ✅ Periodic saves during long monologues
- ✅ Clean, readable transcript display
- ✅ Session management and history
