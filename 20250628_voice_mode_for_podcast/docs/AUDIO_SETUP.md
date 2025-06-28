# Audio Setup Guide for ChatGPT Voice Mode Transcript Recorder

## 🚨 Critical Issue Detected: No Input Devices Found

Your system currently shows **no audio input devices**, which means we need to set up audio capture before proceeding.

## macOS Audio Setup Requirements

### 1. Microphone Setup
Since you're using a Mac mini, you'll need an external microphone:

**Options:**
- **USB Microphone** (easiest): Blue Yeti, Audio-Technica ATR2100x-USB, etc.
- **3.5mm Microphone**: Connect to audio input jack (if available)
- **USB Headset**: With built-in microphone
- **Bluetooth Headset**: AirPods, etc. (may have latency issues)

### 2. System Audio Capture Setup
To capture ChatGPT's voice output, you need a **virtual audio device**:

#### Option A: BlackHole (Recommended - Free)
```bash
# Install via Homebrew
brew install blackhole-2ch

# Or download from: https://github.com/ExistentialAudio/BlackHole
```

#### Option B: SoundFlower (Alternative)
```bash
# Download from: https://github.com/mattingalls/Soundflower
```

### 3. Privacy Permissions
macOS requires explicit permission for microphone access:

1. **System Preferences** → **Security & Privacy** → **Privacy**
2. Select **Microphone** from the left sidebar
3. Check the box next to **Terminal** (or your Python app)
4. You may need to restart Terminal

### 4. Audio Routing Configuration

Once you have BlackHole installed:

1. **Open Audio MIDI Setup** (Applications → Utilities)
2. **Create Multi-Output Device**:
   - Include your speakers/headphones
   - Include BlackHole 2ch
3. **Set ChatGPT to output to Multi-Output Device**
4. **Set our recorder to capture from BlackHole 2ch**

## Testing Steps

### Step 1: Connect Microphone
1. Connect a USB microphone or headset
2. Check **System Preferences** → **Sound** → **Input**
3. Verify the microphone appears and shows input levels

### Step 2: Install Virtual Audio Device
```bash
# Install BlackHole
brew install blackhole-2ch

# Restart your system after installation
```

### Step 3: Configure Audio Routing
1. Open **Audio MIDI Setup**
2. Create a **Multi-Output Device** that includes:
   - Your normal speakers/headphones
   - BlackHole 2ch
3. Set this as your default output device

### Step 4: Test Audio Capture
```bash
# Run our audio test again
source venv/bin/activate
cd src
python audio_test.py
```

## Expected Results After Setup

You should see devices like:
```
INPUT DEVICES:
  3: USB Microphone (or similar)
  4: BlackHole 2ch

OUTPUT DEVICES:
  0: VN279
  1: HP U27 4k WL
  2: Mac mini Speakers
  5: Multi-Output Device
```

## Troubleshooting

### No Microphone Detected
- Check USB connection
- Try different USB port
- Check System Preferences → Sound → Input
- Grant microphone permissions to Terminal

### No System Audio Captured
- Verify BlackHole is installed
- Check Audio MIDI Setup configuration
- Ensure ChatGPT is outputting to Multi-Output Device
- Test with music/video first

### Permission Denied Errors
- Grant microphone access in System Preferences
- Restart Terminal after granting permissions
- Try running with `sudo` (not recommended for regular use)

## Next Steps

1. **Get a microphone** if you don't have one
2. **Install BlackHole** for system audio capture
3. **Configure audio routing** as described above
4. **Re-run the audio test** to verify everything works
5. **Proceed with Whisper integration** once audio capture is confirmed

## Hardware Recommendations

### Budget Option (~$30-50)
- **Audio-Technica ATR2100x-USB**: Great USB microphone
- **Samson Q2U**: Dynamic USB/XLR microphone

### Premium Option (~$100-150)
- **Blue Yeti**: Popular USB microphone with multiple patterns
- **Rode PodMic**: Professional podcasting microphone

### Quick Test Option
- **Any USB headset**: Just to get started and test the system
- **AirPods**: For initial testing (may have latency)

---

**🎯 Priority**: Get microphone input working first, then tackle system audio capture. The microphone is essential for capturing your voice to ChatGPT.
