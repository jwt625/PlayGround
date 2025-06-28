#!/usr/bin/env python3
"""
ChatGPT Voice Mode Transcript Recorder
Main Flask Application
"""

from flask import Flask, render_template, request, jsonify, Response
import os
import json
import sqlite3
from datetime import datetime
import threading
import time
import queue

# Import our custom modules
from src.audio_capture import AudioCapture
from src.transcript_processor import TranscriptProcessor

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'

# SSE streaming queue
stream_queue = queue.Queue(maxsize=1000)  # Prevent memory issues

# Global state
recording_state = {
    'is_recording': False,
    'session_id': None,
    'start_time': None
}

audio_capture = None
transcript_processor = None

@app.route('/')
def index():
    """Main transcript display page"""
    return render_template('index.html')

@app.route('/stream')
def stream():
    """Server-Sent Events endpoint for real-time updates"""
    def event_stream():
        while True:
            try:
                # Get data from queue (blocks until available)
                data = stream_queue.get(timeout=1)
                yield f"data: {json.dumps(data)}\n\n"
            except queue.Empty:
                # Send heartbeat to keep connection alive
                yield "data: {\"type\": \"heartbeat\"}\n\n"
            except Exception as e:
                print(f"SSE stream error: {e}")
                break

    return Response(event_stream(),
                   mimetype='text/event-stream',
                   headers={
                       'Cache-Control': 'no-cache',
                       'Connection': 'keep-alive',
                       'Access-Control-Allow-Origin': '*'
                   })

@app.route('/api/status')
def get_status():
    """Get current recording status"""
    return jsonify({
        'is_recording': recording_state['is_recording'],
        'session_id': recording_state['session_id'],
        'start_time': recording_state['start_time']
    })

@app.route('/api/start', methods=['POST'])
def start_recording():
    """Start recording and transcription"""
    global audio_capture, transcript_processor, recording_state
    
    try:
        if recording_state['is_recording']:
            return jsonify({'error': 'Already recording'}), 400
        
        # Initialize components
        audio_capture = AudioCapture()
        transcript_processor = TranscriptProcessor()
        
        # Check audio devices
        input_devices, output_devices = audio_capture.list_devices()
        if not input_devices:
            return jsonify({
                'error': 'No microphone detected',
                'message': 'Please connect a microphone and check audio permissions'
            }), 400
        
        # Start recording
        session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        recording_state.update({
            'is_recording': True,
            'session_id': session_id,
            'start_time': datetime.now().isoformat()
        })
        
        # Send recording started message via SSE
        try:
            stream_queue.put({
                'type': 'transcript_update',
                'session_id': session_id,
                'timestamp': datetime.now().isoformat(),
                'source': 'system',
                'text': '🎯 Recording started! SSE connection working.',
                'confidence': 1.0,
                'is_final': True
            }, block=False)
        except queue.Full:
            print("⚠️ Stream queue full, dropping message")

        # Start audio capture in background thread
        def audio_thread():
            audio_capture.start_recording(
                session_id=session_id,
                callback=on_audio_chunk
            )

        threading.Thread(target=audio_thread, daemon=True).start()
        
        return jsonify({
            'success': True,
            'session_id': session_id,
            'message': 'Recording started'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/stop', methods=['POST'])
def stop_recording():
    """Stop recording and transcription"""
    global audio_capture, recording_state
    
    try:
        if not recording_state['is_recording']:
            return jsonify({'error': 'Not currently recording'}), 400
        
        # Stop audio capture
        if audio_capture:
            audio_capture.stop_recording()
        
        # Update state
        recording_state.update({
            'is_recording': False,
            'session_id': None,
            'start_time': None
        })
        
        return jsonify({
            'success': True,
            'message': 'Recording stopped'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/sessions')
def get_sessions():
    """Get list of recording sessions"""
    try:
        # TODO: Implement database query for sessions
        sessions = []
        return jsonify({'sessions': sessions})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/transcript/<session_id>')
def get_transcript(session_id):
    """Get transcript for a specific session"""
    try:
        # TODO: Implement database query for transcript
        transcript = []
        return jsonify({'transcript': transcript})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def on_audio_chunk(audio_data, source='microphone', audio_level=None, is_transcription=False):
    """Callback for when new audio data is available"""
    global transcript_processor

    # Handle audio level updates
    if audio_level is not None:
        level_data = {
            'type': 'audio_level',
            'timestamp': datetime.now().isoformat()
        }
        if source == 'microphone':
            level_data['microphone_level'] = audio_level
        elif source == 'system':
            level_data['system_level'] = audio_level

        # Send audio level via SSE queue
        try:
            stream_queue.put(level_data, block=False)
            print(f"🔊 Audio level queued: {level_data}")
        except queue.Full:
            # Skip if queue is full (audio levels are frequent)
            pass

    # Handle transcription processing
    if is_transcription and transcript_processor:
        try:
            # Process audio chunk with Whisper
            transcript_result = transcript_processor.process_audio_chunk(
                audio_data,
                source=source
            )

            if transcript_result and transcript_result.get('text', '').strip():
                print(f"📝 Transcript ({source}): {transcript_result['text']}")

                # Send transcript update via SSE queue
                transcript_data = {
                    'type': 'transcript_update',
                    'session_id': recording_state['session_id'],
                    'timestamp': datetime.now().isoformat(),
                    'source': source,
                    'text': transcript_result['text'],
                    'confidence': transcript_result.get('confidence', 0),
                    'is_final': transcript_result.get('is_final', False)
                }
                try:
                    stream_queue.put(transcript_data, block=False)
                    print(f"📤 Transcript queued: {transcript_data}")
                except queue.Full:
                    print("⚠️ Stream queue full, dropping transcript")
        except Exception as e:
            print(f"Error processing transcript: {e}")

# SSE doesn't need connection handlers - connections are automatic

def init_database():
    """Initialize SQLite database"""
    conn = sqlite3.connect('transcripts.db')
    cursor = conn.cursor()
    
    # Create sessions table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS sessions (
            id TEXT PRIMARY KEY,
            start_time TEXT NOT NULL,
            end_time TEXT,
            duration INTEGER,
            total_segments INTEGER DEFAULT 0
        )
    ''')
    
    # Create transcripts table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS transcripts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            source TEXT NOT NULL,
            text TEXT NOT NULL,
            confidence REAL,
            is_final BOOLEAN DEFAULT 0,
            FOREIGN KEY (session_id) REFERENCES sessions (id)
        )
    ''')
    
    conn.commit()
    conn.close()

if __name__ == '__main__':
    # Initialize database
    init_database()
    
    print("🎯 ChatGPT Voice Mode Transcript Recorder")
    print("=" * 50)
    print("Starting Flask server with SSE...")
    print("Open http://localhost:5001 in your browser")
    print("SSE stream available at http://localhost:5001/stream")
    print("=" * 50)

    # Run the app (regular Flask, no SocketIO)
    app.run(debug=True, host='0.0.0.0', port=5001, threaded=True)
