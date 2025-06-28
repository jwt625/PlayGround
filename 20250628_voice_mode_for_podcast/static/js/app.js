// ChatGPT Voice Mode Transcript Recorder - Frontend JavaScript

class TranscriptRecorder {
    constructor() {
        this.eventSource = null;
        this.isRecording = false;
        this.sessionId = null;
        this.startTime = null;
        this.transcriptEntries = [];
        this.segmentCount = 0;
        this.wordCount = 0;
        this.confidenceSum = 0;
        this.confidenceCount = 0;

        // Transcript combining state
        this.currentMessage = null;  // Current message being built
        this.lastSpeaker = null;     // Last speaker (microphone/system)
        this.lastUpdateTime = null;  // Time of last transcript update
        this.combineTimeoutMs = 5000; // Combine chunks within 5 seconds
        this.combineTimer = null;    // Timer for finalizing messages

        this.initializeElements();
        this.setupEventListeners();
        this.setupSSEConnection();
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
    
    setupSSEConnection() {
        console.log('Setting up SSE connection...');
        this.eventSource = new EventSource('/stream');

        this.eventSource.onopen = () => {
            console.log('📡 SSE connection opened');
            this.updateStatus('ready', 'Connected');
        };

        this.eventSource.onerror = (error) => {
            console.error('❌ SSE connection error:', error);
            this.updateStatus('error', 'Connection Error');
        };

        this.eventSource.onmessage = (event) => {
            try {
                console.log('🔍 Raw SSE event data:', event.data);
                const data = JSON.parse(event.data);
                console.log('📨 SSE message parsed:', data);

                switch (data.type) {
                    case 'transcript_update':
                        console.log('📝 Processing transcript update:', data);
                        this.addTranscriptEntry(data);
                        break;
                    case 'audio_level':
                        console.log('🔊 Processing audio level:', data);
                        console.log('🔊 Microphone level:', data.microphone_level);
                        console.log('🔊 System level:', data.system_level);
                        this.updateAudioLevels(data);
                        break;
                    case 'heartbeat':
                        console.log('💓 Heartbeat received');
                        break;
                    default:
                        console.log('❓ Unknown SSE message type:', data.type, data);
                }
            } catch (error) {
                console.error('❌ Error parsing SSE message:', error, 'Raw data:', event.data);
            }
        };
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

            // Finalize any current message before stopping
            if (this.currentMessage) {
                this.finalizeCurrentMessage();
            }

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
        // Finalize any current message
        if (this.currentMessage) {
            this.finalizeCurrentMessage();
        }

        // Clear combine timer
        if (this.combineTimer) {
            clearTimeout(this.combineTimer);
            this.combineTimer = null;
        }

        // Reset all state
        this.transcriptEntries = [];
        this.segmentCount = 0;
        this.wordCount = 0;
        this.confidenceSum = 0;
        this.confidenceCount = 0;
        this.currentMessage = null;
        this.lastSpeaker = null;
        this.lastUpdateTime = null;

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

        const currentTime = new Date(data.timestamp);
        const speaker = data.source;

        // Check if we should combine with existing message
        const shouldCombine = this.shouldCombineWithCurrent(speaker, currentTime);

        if (shouldCombine && this.currentMessage) {
            // Update existing message
            this.updateCurrentMessage(data, currentTime);
        } else {
            // Finalize previous message if exists
            if (this.currentMessage) {
                this.finalizeCurrentMessage();
            }

            // Start new message
            this.startNewMessage(data, currentTime);
        }

        // Set timer to finalize message after timeout
        this.resetCombineTimer();

        // Update last save time
        this.lastSaveSpan.textContent = new Date().toLocaleTimeString();
    }

    shouldCombineWithCurrent(speaker, currentTime) {
        if (!this.currentMessage || !this.lastUpdateTime) {
            return false;
        }

        // Same speaker?
        if (this.lastSpeaker !== speaker) {
            return false;
        }

        // Within time window?
        const timeDiff = currentTime - this.lastUpdateTime;
        if (timeDiff > this.combineTimeoutMs) {
            return false;
        }

        return true;
    }

    startNewMessage(data, timestamp) {
        const confidence = data.confidence || 0;
        const confidenceClass = this.getConfidenceClass(confidence);

        // Create new transcript entry
        const entry = document.createElement('div');
        entry.className = `transcript-entry ${data.source === 'microphone' ? 'user' : 'chatgpt'}`;
        if (!data.is_final) {
            entry.classList.add('processing');
        }

        const timeString = timestamp.toLocaleTimeString();

        entry.innerHTML = `
            <div class="transcript-meta">
                <span>${data.source === 'microphone' ? '🎤 You' : '🤖 ChatGPT'}</span>
                <span class="timestamp">${timeString}</span>
                <span class="confidence-indicator ${confidenceClass}">
                    ${Math.round(confidence * 100)}%
                </span>
            </div>
            <div class="transcript-text">${data.text}</div>
        `;

        // Mark as processing initially (will be removed when finalized)
        entry.classList.add('processing');

        // Add to transcript
        this.transcriptContent.appendChild(entry);
        this.transcriptContent.scrollTop = this.transcriptContent.scrollHeight;

        // Store current message state
        this.currentMessage = {
            element: entry,
            data: data,
            text: data.text,
            confidenceSum: confidence,
            confidenceCount: 1,
            wordCount: data.text.split(' ').length,
            startTime: timestamp
        };

        this.lastSpeaker = data.source;
        this.lastUpdateTime = timestamp;
    }

    updateCurrentMessage(data, timestamp) {
        if (!this.currentMessage) return;

        const confidence = data.confidence || 0;

        // Combine text (add space if needed)
        const newText = data.text.trim();
        if (newText) {
            // Check if we need a space between chunks
            const currentText = this.currentMessage.text.trim();
            const needsSpace = currentText.length > 0 &&
                              !currentText.endsWith(' ') &&
                              !currentText.endsWith('.') &&
                              !currentText.endsWith(',') &&
                              !currentText.endsWith('!') &&
                              !currentText.endsWith('?') &&
                              !newText.startsWith(' ');

            this.currentMessage.text = currentText + (needsSpace ? ' ' : '') + newText;
        }

        // Update confidence and word count
        this.currentMessage.confidenceSum += confidence;
        this.currentMessage.confidenceCount++;
        this.currentMessage.wordCount = this.currentMessage.text.split(' ').length;

        // Update the DOM element
        const avgConfidence = this.currentMessage.confidenceSum / this.currentMessage.confidenceCount;
        const confidenceClass = this.getConfidenceClass(avgConfidence);

        // Update confidence indicator
        const confidenceIndicator = this.currentMessage.element.querySelector('.confidence-indicator');
        confidenceIndicator.className = `confidence-indicator ${confidenceClass}`;
        confidenceIndicator.textContent = `${Math.round(avgConfidence * 100)}%`;

        // Update text content
        const textElement = this.currentMessage.element.querySelector('.transcript-text');
        textElement.textContent = this.currentMessage.text;

        // Scroll to bottom
        this.transcriptContent.scrollTop = this.transcriptContent.scrollHeight;

        this.lastUpdateTime = timestamp;
    }

    finalizeCurrentMessage() {
        if (!this.currentMessage) return;

        // Remove processing class
        this.currentMessage.element.classList.remove('processing');

        // Update statistics
        this.transcriptEntries.push(this.currentMessage.data);
        this.segmentCount++;
        this.wordCount += this.currentMessage.wordCount;
        this.confidenceSum += this.currentMessage.confidenceSum / this.currentMessage.confidenceCount;
        this.confidenceCount++;

        this.updateStats();
        this.updateQualityMetrics();

        // Clear current message
        this.currentMessage = null;
    }

    resetCombineTimer() {
        if (this.combineTimer) {
            clearTimeout(this.combineTimer);
        }

        this.combineTimer = setTimeout(() => {
            if (this.currentMessage) {
                console.log('🕐 Finalizing message due to timeout');
                this.finalizeCurrentMessage();
            }
        }, this.combineTimeoutMs);
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
        console.log('🎚️ updateAudioLevels called with:', data);

        if (data.microphone_level !== undefined) {
            const percentage = data.microphone_level * 100;
            console.log(`🎤 Setting mic level to ${percentage}%`);
            this.micLevel.style.width = `${percentage}%`;
        }
        if (data.system_level !== undefined) {
            const percentage = data.system_level * 100;
            console.log(`🔊 Setting system level to ${percentage}%`);
            this.systemLevel.style.width = `${percentage}%`;
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
