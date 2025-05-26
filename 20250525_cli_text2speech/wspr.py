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
    def __init__(self, vad_mode=3, max_silence_ms=1500):
        self.vad = webrtcvad.Vad(vad_mode)
        self.frames = collections.deque()
        self.silence_limit = max_silence_ms / 1000.0
        self.last_voice_time = time.time()

    def add_frame(self, frame):
        is_speech = self.vad.is_speech(frame, RATE)
        if is_speech:
            self.last_voice_time = time.time()
        self.frames.append(frame)
        return is_speech

    def is_silence_timeout(self):
        return (time.time() - self.last_voice_time) > self.silence_limit

    def save_wav(self):
        temp_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        with wave.open(temp_wav.name, 'wb') as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(2)
            wf.setframerate(RATE)
            wf.writeframes(b''.join(self.frames))
        return temp_wav.name


def record_until_silence():
    audio = pyaudio.PyAudio()
    stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=FRAME_SIZE)

    print("Listening...")
    buffer = AudioBuffer()
    while True:
        frame = stream.read(FRAME_SIZE, exception_on_overflow=False)
        buffer.add_frame(frame)
        if buffer.is_silence_timeout():
            break

    stream.stop_stream()
    stream.close()
    audio.terminate()
    return buffer.save_wav()


def transcribe_with_whisper(audio_path, model_name="base"):
    model = whisper.load_model(model_name)
    result = model.transcribe(audio_path)
    return result['text']


def main():
    parser = argparse.ArgumentParser(description="Stream mic to Whisper with auto stop on pause")
    parser.add_argument('--model', default='base', help='Whisper model to use')
    args = parser.parse_args()

    audio_path = record_until_silence()
    print("Transcribing...")
    text = transcribe_with_whisper(audio_path, model_name=args.model)
    print("Transcription:")
    print(text)

    os.remove(audio_path)


if __name__ == '__main__':
    main()