// ChatGPT Voice Mode Transcript Recorder - Frontend JavaScript

class TranscriptRecorder {
    constructor() {
        this.socket = io();
        this.isRecording = false;
        this.sessionId = null;
        this.startTime = null;
        this.transcriptEntries = [];
        this.segmentCount = 0;
        this.wordCount = 0;
        this.confidenceSum = 0;
        this.confidenceCount = 0;
        
        this.initializeElements();
        this.setupEventListeners();
        this.setupSocketListeners();
    }
    
    initializeElements() {
        // Control buttons
        this.startBtn = document.getElementById('start-btn');
        this.stopBtn = document.getElementById('stop-btn');
        this.clearBtn = document.getElementById('clear-btn');
        
        // Status elements
        this.statusDot = document.getElementById('status-dot');
        this.statusText = document.getElementById('status-text');
        this.sessionInfo = document.getElementById('session-info');
        this.sessionIdSpan = document.getElementById('session-id');
        this.durationSpan = document.getElementById('duration');
        
        // Audio level meters
        this.micLevel = document.getElementById('mic-level');
        this.systemLevel = document.getElementById('system-level');
        
        // Transcript elements
        this.transcriptContent = document.getElementById('transcript-content');
        this.segmentCountSpan = document.getElementById('segment-count');
        this.wordCountSpan = document.getElementById('word-count');
        
        // Quality monitor
        this.avgConfidenceSpan = document.getElementById('avg-confidence');
        this.processingDelaySpan = document.getElementById('processing-delay');
        this.lastSaveSpan = document.getElementById('last-save');
    }
    
    setupEventListeners() {
        this.startBtn.addEventListener('click', () => this.startRecording());
        this.stopBtn.addEventListener('click', () => this.stopRecording());
        this.clearBtn.addEventListener('click', () => this.clearTranscript());
    }
    
    setupSocketListeners() {
        this.socket.on('connect', () => {
            console.log('Connected to server');
            this.updateStatus('ready', 'Connected');
        });
        
        this.socket.on('disconnect', () => {
            console.log('Disconnected from server');
            this.updateStatus('error', 'Disconnected');
        });
        
        this.socket.on('status', (data) => {
            console.log('Status update:', data);
            if (data.recording_state) {
                this.updateRecordingState(data.recording_state);
            }
        });
        
        this.socket.on('transcript_update', (data) => {
            console.log('Transcript update:', data);
            this.addTranscriptEntry(data);
        });
        
        this.socket.on('audio_level', (data) => {
            this.updateAudioLevels(data);
        });
    }
    
    async startRecording() {
        try {
            this.updateStatus('recording', 'Starting...');
            this.startBtn.disabled = true;
            
            const response = await fetch('/api/start', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                }
            });
            
            const result = await response.json();
            
            if (response.ok) {
                this.isRecording = true;
                this.sessionId = result.session_id;
                this.startTime = new Date();
                
                this.updateStatus('recording', 'Recording');
                this.startBtn.disabled = true;
                this.stopBtn.disabled = false;
                
                this.showSessionInfo();
                this.startDurationTimer();
                this.clearTranscriptPlaceholder();
                
                console.log('Recording started:', result);
            } else {
                throw new Error(result.error || 'Failed to start recording');
            }
            
        } catch (error) {
            console.error('Error starting recording:', error);
            this.updateStatus('error', 'Error: ' + error.message);
            this.startBtn.disabled = false;
            
            // Show error message to user
            this.showErrorMessage(error.message);
        }
    }
    
    async stopRecording() {
        try {
            this.updateStatus('ready', 'Stopping...');
            this.stopBtn.disabled = true;
            
            const response = await fetch('/api/stop', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                }
            });
            
            const result = await response.json();
            
            if (response.ok) {
                this.isRecording = false;
                this.sessionId = null;
                this.startTime = null;
                
                this.updateStatus('ready', 'Ready');
                this.startBtn.disabled = false;
                this.stopBtn.disabled = true;
                
                this.hideSessionInfo();
                this.stopDurationTimer();
                
                console.log('Recording stopped:', result);
            } else {
                throw new Error(result.error || 'Failed to stop recording');
            }
            
        } catch (error) {
            console.error('Error stopping recording:', error);
            this.updateStatus('error', 'Error: ' + error.message);
            this.stopBtn.disabled = false;
        }
    }
    
    clearTranscript() {
        this.transcriptEntries = [];
        this.segmentCount = 0;
        this.wordCount = 0;
        this.confidenceSum = 0;
        this.confidenceCount = 0;
        
        this.transcriptContent.innerHTML = `
            <div class="transcript-placeholder">
                <p>🎙️ Transcript cleared. Click "Start Recording" to begin again.</p>
            </div>
        `;
        
        this.updateStats();
        this.updateQualityMetrics();
    }
    
    addTranscriptEntry(data) {
        // Remove placeholder if it exists
        this.clearTranscriptPlaceholder();
        
        // Create transcript entry
        const entry = document.createElement('div');
        entry.className = `transcript-entry ${data.source === 'microphone' ? 'user' : 'chatgpt'}`;
        if (!data.is_final) {
            entry.classList.add('processing');
        }
        
        const timestamp = new Date(data.timestamp).toLocaleTimeString();
        const confidence = data.confidence || 0;
        const confidenceClass = this.getConfidenceClass(confidence);
        
        entry.innerHTML = `
            <div class="transcript-meta">
                <span>${data.source === 'microphone' ? '🎤 You' : '🤖 ChatGPT'}</span>
                <span>${timestamp}</span>
                <span class="confidence-indicator ${confidenceClass}">
                    ${Math.round(confidence * 100)}%
                </span>
            </div>
            <div class="transcript-text">${data.text}</div>
        `;
        
        // Add to transcript
        this.transcriptContent.appendChild(entry);
        
        // Scroll to bottom
        this.transcriptContent.scrollTop = this.transcriptContent.scrollHeight;
        
        // Update statistics
        this.transcriptEntries.push(data);
        if (data.is_final) {
            this.segmentCount++;
            this.wordCount += data.text.split(' ').length;
            this.confidenceSum += confidence;
            this.confidenceCount++;
        }
        
        this.updateStats();
        this.updateQualityMetrics();
        
        // Update last save time
        this.lastSaveSpan.textContent = new Date().toLocaleTimeString();
    }
    
    updateStatus(type, text) {
        this.statusDot.className = `status-dot ${type}`;
        this.statusText.textContent = text;
    }
    
    updateRecordingState(state) {
        if (state.is_recording) {
            this.isRecording = true;
            this.sessionId = state.session_id;
            this.startTime = state.start_time ? new Date(state.start_time) : null;
            
            this.updateStatus('recording', 'Recording');
            this.startBtn.disabled = true;
            this.stopBtn.disabled = false;
            this.showSessionInfo();
            this.startDurationTimer();
        } else {
            this.isRecording = false;
            this.sessionId = null;
            this.startTime = null;
            
            this.updateStatus('ready', 'Ready');
            this.startBtn.disabled = false;
            this.stopBtn.disabled = true;
            this.hideSessionInfo();
            this.stopDurationTimer();
        }
    }
    
    updateAudioLevels(data) {
        if (data.microphone_level !== undefined) {
            this.micLevel.style.width = `${data.microphone_level * 100}%`;
        }
        if (data.system_level !== undefined) {
            this.systemLevel.style.width = `${data.system_level * 100}%`;
        }
    }
    
    showSessionInfo() {
        this.sessionInfo.style.display = 'block';
        this.sessionIdSpan.textContent = this.sessionId || 'Unknown';
    }
    
    hideSessionInfo() {
        this.sessionInfo.style.display = 'none';
    }
    
    startDurationTimer() {
        this.durationTimer = setInterval(() => {
            if (this.startTime) {
                const elapsed = Math.floor((new Date() - this.startTime) / 1000);
                const minutes = Math.floor(elapsed / 60);
                const seconds = elapsed % 60;
                this.durationSpan.textContent = 
                    `${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}`;
            }
        }, 1000);
    }
    
    stopDurationTimer() {
        if (this.durationTimer) {
            clearInterval(this.durationTimer);
            this.durationTimer = null;
        }
        this.durationSpan.textContent = '00:00';
    }
    
    clearTranscriptPlaceholder() {
        const placeholder = this.transcriptContent.querySelector('.transcript-placeholder');
        if (placeholder) {
            placeholder.remove();
        }
    }
    
    updateStats() {
        this.segmentCountSpan.textContent = this.segmentCount;
        this.wordCountSpan.textContent = this.wordCount;
    }
    
    updateQualityMetrics() {
        // Average confidence
        if (this.confidenceCount > 0) {
            const avgConfidence = this.confidenceSum / this.confidenceCount;
            this.avgConfidenceSpan.textContent = `${Math.round(avgConfidence * 100)}%`;
        } else {
            this.avgConfidenceSpan.textContent = '--';
        }
        
        // Processing delay (placeholder for now)
        this.processingDelaySpan.textContent = '< 2s';
    }
    
    getConfidenceClass(confidence) {
        if (confidence >= 0.8) return 'confidence-high';
        if (confidence >= 0.6) return 'confidence-medium';
        return 'confidence-low';
    }
    
    showErrorMessage(message) {
        // Create error notification
        const errorDiv = document.createElement('div');
        errorDiv.className = 'error-notification';
        errorDiv.innerHTML = `
            <div style="background: #ff3b30; color: white; padding: 15px; border-radius: 8px; margin: 10px 0;">
                <strong>Error:</strong> ${message}
                <br><small>Check the console for more details or see AUDIO_SETUP.md for help.</small>
            </div>
        `;
        
        // Insert at top of container
        const container = document.querySelector('.container');
        container.insertBefore(errorDiv, container.firstChild);
        
        // Remove after 10 seconds
        setTimeout(() => {
            errorDiv.remove();
        }, 10000);
    }
}

// Initialize the app when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    console.log('Initializing ChatGPT Voice Mode Transcript Recorder...');
    window.transcriptRecorder = new TranscriptRecorder();
});
