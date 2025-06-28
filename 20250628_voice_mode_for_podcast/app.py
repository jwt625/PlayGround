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

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("✅ Loaded environment variables from .env file")
except ImportError:
    print("⚠️  python-dotenv not installed. Install with: pip install python-dotenv")
except Exception as e:
    print(f"⚠️  Could not load .env file: {e}")

# Import our custom modules
from src.audio_capture import AudioCapture
from src.transcript_processor import TranscriptProcessor
from src.transcription_service import get_transcription_service

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'

# Configuration for transcription provider
TRANSCRIPTION_PROVIDER = os.getenv('TRANSCRIPTION_PROVIDER', 'whisper')  # 'whisper' or 'assemblyai'
TRANSCRIPTION_CONFIG = {
    'whisper': {
        'model_name': 'small',  # or 'base', 'large', etc.
        'language': 'en'
    },
    'assemblyai': {
        'speech_model': 'best',  # or 'nano'
        'sample_rate': 16000,
        'format_turns': True
    }
}

# SSE streaming queue
stream_queue = queue.Queue(maxsize=1000)  # Prevent memory issues

# Global state
recording_state = {
    'is_recording': False,
    'session_id': None,
    'start_time': None,
    'transcription_provider': TRANSCRIPTION_PROVIDER
}

audio_capture = None
transcript_processor = None  # Keep for backward compatibility
transcription_service = None

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
        'start_time': recording_state['start_time'],
        'transcription_provider': recording_state['transcription_provider'],
        'available_providers': ['whisper', 'assemblyai']
    })

@app.route('/api/set-provider', methods=['POST'])
def set_transcription_provider():
    """Set the transcription provider"""
    global transcription_service

    try:
        if recording_state['is_recording']:
            return jsonify({'error': 'Cannot change provider while recording'}), 400

        data = request.get_json()
        provider = data.get('provider', '').lower()

        if provider not in ['whisper', 'assemblyai']:
            return jsonify({'error': 'Invalid provider. Must be "whisper" or "assemblyai"'}), 400

        # Clean up existing service
        if transcription_service:
            transcription_service.cleanup()
            transcription_service = None

        # Update global configuration
        recording_state['transcription_provider'] = provider

        return jsonify({
            'success': True,
            'provider': provider,
            'message': f'Transcription provider set to {provider}'
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/start', methods=['POST'])
def start_recording():
    """Start recording and transcription"""
    global audio_capture, transcript_processor, transcription_service, recording_state

    try:
        if recording_state['is_recording']:
            return jsonify({'error': 'Already recording'}), 400

        # Initialize audio capture
        audio_capture = AudioCapture()

        # Initialize transcription service
        provider = recording_state['transcription_provider']
        config = TRANSCRIPTION_CONFIG.get(provider, {})

        try:
            transcription_service = get_transcription_service(provider, config)
            print(f"✅ Initialized {provider} transcription service")

            # If using AssemblyAI, start streaming mode
            if provider == 'assemblyai' and transcription_service.supports_streaming():
                def streaming_callback(result):
                    """Handle streaming transcription results"""
                    if result and result.text.strip():
                        print(f"📝 Streaming transcript ({provider}): {result.text}")

                        # Send transcript update via SSE queue
                        transcript_data = {
                            'type': 'transcript_update',
                            'session_id': recording_state['session_id'],
                            'timestamp': result.timestamp,
                            'source': 'microphone',
                            'text': result.text,
                            'confidence': result.confidence,
                            'is_final': result.is_final,
                            'provider': provider,
                            'processing_time': result.processing_time
                        }
                        try:
                            stream_queue.put(transcript_data, block=False)
                            print(f"📤 Streaming transcript queued: {transcript_data}")
                        except queue.Full:
                            print("⚠️ Stream queue full, dropping streaming transcript")

                # Start streaming transcription
                streaming_success = transcription_service.start_streaming(streaming_callback, sample_rate=16000)
                if streaming_success:
                    print(f"🌊 Started {provider} streaming transcription")
                else:
                    print(f"⚠️ Failed to start {provider} streaming transcription")

        except Exception as e:
            return jsonify({
                'error': f'Failed to initialize {provider} transcription service',
                'message': str(e)
            }), 500

        # Keep backward compatibility with old transcript processor
        if provider == 'whisper':
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
    global audio_capture, transcription_service, recording_state

    try:
        if not recording_state['is_recording']:
            return jsonify({'error': 'Not currently recording'}), 400

        # Stop audio capture
        if audio_capture:
            audio_capture.stop_recording()

        # Clean up transcription service
        if transcription_service:
            transcription_service.cleanup()

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
    global transcript_processor, transcription_service

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
    if is_transcription and transcription_service:
        provider = recording_state['transcription_provider']

        # For AssemblyAI, transcription is handled by streaming, skip chunk processing
        if provider == 'assemblyai':
            print(f"🌊 Skipping chunk processing for {provider} (using streaming)")
            return

        try:
            # Process audio chunk with the selected transcription service (Whisper)
            transcript_result = transcription_service.transcribe_audio_data(
                audio_data,
                source=source
            )

            if transcript_result and transcript_result.text.strip():
                print(f"📝 Transcript ({source}, {provider}): {transcript_result.text}")

                # Send transcript update via SSE queue
                transcript_data = {
                    'type': 'transcript_update',
                    'session_id': recording_state['session_id'],
                    'timestamp': transcript_result.timestamp,
                    'source': source,
                    'text': transcript_result.text,
                    'confidence': transcript_result.confidence,
                    'is_final': transcript_result.is_final,
                    'provider': provider,
                    'processing_time': transcript_result.processing_time
                }
                try:
                    stream_queue.put(transcript_data, block=False)
                    print(f"📤 Transcript queued: {transcript_data}")
                except queue.Full:
                    print("⚠️ Stream queue full, dropping transcript")
        except Exception as e:
            print(f"Error processing transcript: {e}")

            # Fallback to old processor if new service fails and we're using Whisper
            if provider == 'whisper' and transcript_processor:
                try:
                    transcript_result = transcript_processor.process_audio_chunk(
                        audio_data,
                        source=source
                    )
                    if transcript_result and transcript_result.get('text', '').strip():
                        transcript_data = {
                            'type': 'transcript_update',
                            'session_id': recording_state['session_id'],
                            'timestamp': datetime.now().isoformat(),
                            'source': source,
                            'text': transcript_result['text'],
                            'confidence': transcript_result.get('confidence', 0),
                            'is_final': transcript_result.get('is_final', False),
                            'provider': 'whisper_fallback'
                        }
                        stream_queue.put(transcript_data, block=False)
                        print(f"📤 Fallback transcript queued: {transcript_data}")
                except Exception as fallback_error:
                    print(f"Fallback transcription also failed: {fallback_error}")

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

    port = int(os.getenv('FLASK_RUN_PORT', 5002))

    print("🎯 ChatGPT Voice Mode Transcript Recorder")
    print("=" * 50)
    print("Starting Flask server with SSE...")
    print(f"Open http://localhost:{port} in your browser")
    print(f"SSE stream available at http://localhost:{port}/stream")
    print("=" * 50)

    # Run the app (regular Flask, no SocketIO)
    app.run(debug=True, host='0.0.0.0', port=port, threaded=True)
