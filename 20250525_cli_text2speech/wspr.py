#!/usr/bin/env python3
import argparse
import collections
import sys
import time
import wave
import os
import tempfile

import pyaudio
import webrtcvad
import whisper

# Audio settings    
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
FRAME_DURATION = 30  # ms
FRAME_SIZE = int(RATE * FRAME_DURATION / 1000)  # samples
FRAME_BYTES = FRAME_SIZE * 2  # 16-bit audio

class AudioBuffer:
    def __init__(self, vad_mode=0, max_silence_ms=1000, quiet=False):
        self.vad = webrtcvad.Vad(vad_mode)
        self.frames = collections.deque()
        self.silence_limit = max_silence_ms / 1000.0
        self.last_voice_time = time.time()
        self.speech_detected = False
        self.quiet = quiet

    def add_frame(self, frame):
        is_speech = self.vad.is_speech(frame, RATE)
        if is_speech:
            self.last_voice_time = time.time()
            if not self.speech_detected:
                if not self.quiet:
                    print("Speech detected!")
                self.speech_detected = True
        self.frames.append(frame)
        return is_speech

    def is_silence_timeout(self):
        # Require at least 2 seconds of recording and some speech detected
        min_recording_time = 2.0
        recording_time = len(self.frames) * FRAME_DURATION / 1000.0
        
        if recording_time < min_recording_time:
            return False
            
        if not self.speech_detected:
            return False
            
        return (time.time() - self.last_voice_time) > self.silence_limit

    def save_wav(self):
        temp_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        with wave.open(temp_wav.name, 'wb') as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(2)
            wf.setframerate(RATE)
            wf.writeframes(b''.join(self.frames))
        return temp_wav.name


def record_until_silence(quiet=False):
    audio = pyaudio.PyAudio()
    stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=FRAME_SIZE)

    if not quiet:
        print("Listening...")
    buffer = AudioBuffer(quiet=quiet)
    while True:
        frame = stream.read(FRAME_SIZE, exception_on_overflow=False)
        buffer.add_frame(frame)
        if buffer.is_silence_timeout():
            break

    stream.stop_stream()
    stream.close()
    audio.terminate()
    return buffer.save_wav()


def transcribe_with_whisper(audio_path, model_name="base", quiet=False):
    model = whisper.load_model(model_name)
    result = model.transcribe(audio_path)
    return result['text']


def main():
    parser = argparse.ArgumentParser(description="Stream mic to Whisper with auto stop on pause")
    parser.add_argument('--model', default='base', help='Whisper model to use')
    parser.add_argument('--quiet', '-q', action='store_true', help='Output only transcribed text (good for piping)')
    args = parser.parse_args()

    audio_path = record_until_silence(quiet=args.quiet)
    if not args.quiet:
        print("Transcribing...")
    
    # Suppress Whisper warnings in quiet mode
    if args.quiet:
        import warnings
        warnings.filterwarnings("ignore")
    
    text = transcribe_with_whisper(audio_path, model_name=args.model, quiet=args.quiet)
    
    if args.quiet:
        print(text.strip())
    else:
        print("Transcription:")
        print(text)

    os.remove(audio_path)


if __name__ == '__main__':
    main()