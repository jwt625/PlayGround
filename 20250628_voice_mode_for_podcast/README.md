# ChatGPT Voice Mode Transcript Recorder

A real-time transcript recorder for ChatGPT voice conversations with beautiful dark mode interface, automatic saving, and quality monitoring.

## 🎯 Features

- **Real-time transcription** using OpenAI Whisper (local processing)
- **Dual audio capture** (microphone + system audio)
- **Beautiful dark mode web interface** with live updates
- **Automatic saving** every conversation round + periodic backups
- **Quality monitoring** with confidence scores and audio levels
- **Session management** with SQLite database storage
- **WebSocket real-time communication** for instant updates

## 🏗️ Architecture

### Single Server Application
This is a **single Flask application** that handles everything:

```
┌─────────────────────────────────────────────────────────────┐
│                    Flask Application                        │
│                     (app.py)                               │
├─────────────────────────────────────────────────────────────┤
│  🌐 Web Server (Flask)                                     │
│  🔌 WebSocket Server (SocketIO)                            │
│  🎤 Audio Capture (PyAudio)                                │
│  🧠 AI Transcription (Whisper)                             │
│  💾 Database (SQLite)                                      │
└─────────────────────────────────────────────────────────────┘
```

**No multiple servers** - everything runs in one Python process with:
- **Main thread**: Flask web server + WebSocket handling
- **Background thread**: Audio capture and processing
- **Whisper processing**: On-demand in background thread

## 🚀 Quick Start

### 1. Setup Environment
```bash
# Navigate to project directory
cd 20250628_voice_mode_for_podcast

# Activate virtual environment
source venv/bin/activate

# Verify dependencies (already installed)
pip list | grep -E "(flask|whisper|pyaudio)"
```

### 2. Start the Application
```bash
# Start the single Flask server
python app.py
```

**That's it!** The application will:
- ✅ Start Flask web server on `http://localhost:5001`
- ✅ Initialize WebSocket server for real-time updates
- ✅ Load Whisper AI model for transcription
- ✅ Set up SQLite database for storage
- ✅ Display startup information

### 3. Access the Interface
Open your browser to: **http://localhost:5001**

## 📁 Project Structure

```
20250628_voice_mode_for_podcast/
├── app.py                 # 🚀 Main Flask application (START HERE)
├── requirements.txt       # 📦 Python dependencies
├── transcripts.db        # 💾 SQLite database (auto-created)
├── README.md             # 📖 This file
├── TODO.md               # 📋 Implementation plan
├── STATUS.md             # 📊 Current progress
├── AUDIO_SETUP.md        # 🎤 Audio hardware setup guide
│
├── src/                  # 🔧 Core modules
│   ├── audio_capture.py     # 🎤 Audio recording logic
│   ├── transcript_processor.py # 🧠 Whisper integration
│   ├── audio_test.py        # 🧪 Audio testing utility
│   └── whisper_test.py      # 🧪 Whisper testing utility
│
├── templates/            # 🌐 HTML templates
│   └── index.html           # 📄 Main web interface
│
├── static/               # 🎨 Frontend assets
│   ├── css/
│   │   └── style.css        # 🌙 Dark mode styles
│   └── js/
│       └── app.js           # ⚡ Real-time frontend logic
│
├── audio_samples/        # 🎵 Recorded audio files (auto-created)
└── venv/                 # 🐍 Python virtual environment
```

## 🎮 How to Use

### Starting a Recording Session
1. **Open** http://localhost:5001 in your browser
2. **Click** "🎤 Start Recording" button
3. **Speak** into your microphone
4. **Watch** real-time transcripts appear
5. **Monitor** audio levels and quality metrics
6. **Click** "⏹️ Stop Recording" when done

### Features in Action
- **Volume bars** show real-time audio levels
- **Transcripts** appear every ~3 seconds of speech
- **Confidence scores** indicate transcription quality
- **Session info** shows recording duration
- **Auto-save** happens every conversation round

## 🔧 Testing & Debugging

### Test Audio Capture
```bash
# Test microphone and system audio
cd src
python audio_test.py
```

### Test Whisper Transcription
```bash
# Test AI transcription
cd src
python whisper_test.py
```

### Debug WebSocket Connection
1. **Open browser console** (F12 → Console)
2. **Start recording** and look for:
   - `📝 Transcript update received:` messages
   - `🔊 Audio level received:` messages
3. **Check Flask console** for:
   - `📤 Transcript emitted:` messages
   - `🔊 Audio level emitted:` messages

## ⚙️ Configuration

### Audio Settings
- **Sample Rate**: 16kHz (optimized for Whisper)
- **Channels**: Mono (1 channel)
- **Chunk Size**: 1024 samples
- **Processing Frequency**: Every 50 chunks (~3 seconds)

### Whisper Model
- **Default**: `tiny` (~39MB, fastest, good quality for real-time)
- **Alternative**: `base` (~74MB) or `small` (~244MB) (better quality, slower)
- **Current**: Using `tiny` model for optimal real-time performance
- **Change in**: `src/transcript_processor.py` line 18

### Server Settings
- **Host**: `0.0.0.0` (accessible from network)
- **Port**: `5001`
- **Debug Mode**: Enabled (auto-reload on changes)

## 🎤 Audio Hardware Setup

### Required Hardware
1. **Microphone**: USB mic, headset, or AirPods
2. **Virtual Audio Device**: BlackHole for system audio capture

### Setup Instructions
```bash
# Install BlackHole for system audio capture
brew install blackhole-2ch

# Grant microphone permissions
# System Preferences → Security & Privacy → Privacy → Microphone
# ✅ Check "Terminal" or your Python app
```

**Detailed setup**: See `AUDIO_SETUP.md`

## 🐛 Troubleshooting

### No Microphone Detected
- **Connect** a USB microphone or headset
- **Check** System Preferences → Sound → Input
- **Grant** microphone permissions to Terminal

### No Transcripts Appearing
- **Check** browser console for WebSocket errors
- **Verify** Flask console shows "📤 Transcript emitted" messages
- **Ensure** you're speaking clearly for 3+ seconds

### Audio Levels Not Moving
- **Test** microphone with `python src/audio_test.py`
- **Check** audio permissions
- **Try** different microphone device

### WebSocket Connection Issues
- **Refresh** the browser page
- **Check** Flask console for "Client connected" messages
- **Disable** browser ad blockers or extensions

## 📊 Current Status

### ✅ Working Features
- ✅ Flask web server with dark mode UI
- ✅ OpenAI Whisper transcription
- ✅ Audio capture framework
- ✅ WebSocket real-time communication
- ✅ SQLite database storage
- ✅ Session management

### ⚠️ Known Issues
- **Audio levels**: May not update in real-time (debugging in progress)
- **System audio**: Requires BlackHole virtual device setup
- **WebSocket**: Occasional connection issues (refresh browser)

### 🔄 Next Steps
1. **Connect microphone** for full functionality
2. **Install BlackHole** for ChatGPT audio capture
3. **Test end-to-end** with real conversations

## 🚀 Development

### Making Changes
The Flask server runs in **debug mode** with auto-reload:
- **Python changes**: Server automatically restarts
- **Frontend changes**: Refresh browser to see updates
- **CSS/JS changes**: Hard refresh (Cmd+Shift+R / Ctrl+Shift+R)

### Adding Features
- **Backend**: Modify `app.py` or modules in `src/`
- **Frontend**: Update `templates/index.html` or `static/`
- **Database**: Schema defined in `app.py` `init_database()`

## 📝 License

This project is for educational and personal use. OpenAI Whisper is subject to its own license terms.

---

**🎯 Ready to record your ChatGPT conversations!**

Start with: `python app.py` → Open http://localhost:5001 → Click "Start Recording"
