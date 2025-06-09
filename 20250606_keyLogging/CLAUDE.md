# Keystroke Tracking with Prometheus & Grafana

*Last Updated: June 8, 2025*

## Project Overview
Building a passive keystroke tracker in Go to learn backend observability with Prometheus and Grafana visualization.

## Implementation Plan

### Phase 1: Basic Keystroke Counter ✅ COMPLETED
*Completed: June 6, 2025*
**Goal**: Get end-to-end pipeline working with immediate visual feedback

**Tasks**:
- [x] Set up Go project structure
- [x] Implement basic keystroke event listener (OS-specific, read-only)
- [x] Add Prometheus counter metric: `keystrokes_total`
- [x] Create HTTP endpoint to expose metrics (`/metrics`)
- [x] Set up Prometheus configuration to scrape our service
- [x] Create basic Grafana dashboard showing keystrokes over time
- [x] Test: Type and watch graph update in real-time

**Key Libraries**:
- `github.com/prometheus/client_golang` for metrics
- **Native macOS CGEventTap API** (via CGO) - switched from gohook due to compatibility issues

**Implementation Notes**:
- Initial attempt with `github.com/robotn/gohook` failed on modern macOS
- Successfully implemented using native CGEventTap API with CGO
- Grafana running on port 3001 (3000 was occupied)
- Prometheus scraping every 1 second for real-time feel
- Full pipeline working: Keystrokes → Go → Prometheus → Grafana

### Phase 2: Basic Categorization ✅ COMPLETED
*Completed: June 6, 2025*
**Goal**: Add more meaningful metrics with labels

**Tasks**:
- [x] Categorize keystrokes by type (letters, numbers, special keys)
- [x] Update metric: `keystrokes_total{key_type="letter|number|special"}`
- [x] Enhance Grafana dashboard with breakdown by key type

**Implementation Notes**:
- Enhanced CGEventTap callback to analyze macOS keycodes
- Switched from simple Counter to CounterVec with labels
- Created rich Grafana dashboard with stacked charts, gauges, and pie chart
- No Prometheus config changes needed - labels auto-discovered

### Phase 3: Time-based Insights
*Status: Planned*
**Goal**: Calculate derived metrics for better analysis

**Tasks**:
- [ ] Add keystrokes per minute calculation using Prometheus queries
- [ ] Create hourly/daily pattern visualizations
- [ ] Add typing speed trends over time

### Phase 4: Application Awareness ✅ COMPLETED
*Completed: June 6, 2025*
**Goal**: Track typing activity per application

**Tasks**:
- [x] Detect active application during keystroke events
- [x] Add app label: `keystrokes_total{key_type="letter", app="vscode"}`
- [x] Create application-specific dashboards

### Phase 5: Chrome Domain Tracking ✅ COMPLETED
*Completed: June 8, 2025*
**Goal**: Fine-grained tracking of Chrome tab activity by domain

**Tasks**:
- [x] Chrome extension for domain detection
- [x] HTTP endpoint for real-time communication
- [x] Domain-aware metrics with labels
- [x] Chrome tab session duration tracking
- [x] Grafana panels for domain analytics

**Implementation**:
- Chrome extension detects domain changes in real-time
- HTTP POST to `localhost:8080/chrome-update` for immediate updates
- Metrics include: `keystrokes_total{domain="youtube_com"}`, `chrome_tab_total_time_seconds`, `chrome_tab_session_duration_seconds`
- No native messaging or file I/O - pure HTTP communication

## Technical Approach
*Documented: June 6, 2025*

### Security & Privacy
- **Passive monitoring**: Read-only event listening, no keystroke interception
- **No key content logging**: Only count events, never log actual keys pressed
- **Local only**: All data stays on local machine
- **Configurable exclusions**: Option to exclude sensitive applications

### Architecture
```
[Keyboard Events] → [Go Service] → [Prometheus Metrics] → [Grafana Dashboard]
[Mouse Events]   → [Swift Helper] ↗
[App Detection]  ↗               ↓
[Chrome Domains] → [HTTP Endpoint] → [HTTP /metrics endpoint]
```

### Chrome Extension Integration
- **Real-time communication**: HTTP POST to `localhost:8080/chrome-update`
- **Domain detection**: Automatic extraction and sanitization of domains
- **Session tracking**: Duration metrics for each domain
- **Debug interface**: Built-in popup for troubleshooting

### Platform Implementation
- **macOS**: `CGEventTap` with `kCGEventTapOptionListenOnly`
- **Linux**: `/dev/input/event*` or `libevdev`
- **Windows**: `SetWindowsHookEx` with `WH_KEYBOARD_LL`

## Success Criteria for Phase 1 ✅ COMPLETED
- [x] Service runs in background without affecting other programs
- [x] Prometheus successfully scrapes keystroke metrics
- [x] Grafana dashboard shows real-time keystroke activity
- [x] Can see immediate feedback when typing

## Current Status
*As of June 8, 2025*
- **Phase 1**: ✅ Complete - End-to-end pipeline working
- **Phase 2**: ✅ Complete - Key categorization with labels
- **Phase 3**: 🚧 In Progress - Time-based insights
- **Phase 4**: ✅ Complete - Application awareness
- **Phase 5**: ✅ Complete - Chrome domain tracking
- **Access URLs**:
  - Grafana: http://localhost:3001 (admin/admin)
  - Prometheus: http://localhost:9090
  - Metrics: http://localhost:8080/metrics
  - Chrome Extension Debug: Click extension icon in Chrome toolbar

## Key Go Programming Concepts Worth Learning
*Documented: June 6, 2025*

### 1. CGO Integration (`main-categorized.go`)
```go
/*
#cgo CFLAGS: -x objective-c
#cgo LDFLAGS: -framework CoreGraphics -framework CoreFoundation
#include <CoreGraphics/CoreGraphics.h>
*/
import "C"
```
- **CGO**: Calling C/Objective-C from Go for native OS APIs
- **Build directives**: How to link system frameworks
- **Type conversion**: Go ↔ C data exchange (`C.int`, `int(C.getAndResetLetters())`)

### 2. Prometheus Metrics with Labels
```go
keystrokesTotal = prometheus.NewCounterVec(
    prometheus.CounterOpts{
        Name: "keystrokes_total",
        Help: "Total number of keystrokes recorded by key type",
    },
    []string{"key_type"}, // Label keys
)

// Usage with labels
keystrokesTotal.WithLabelValues("letter").Add(float64(letters))
```
- **CounterVec vs Counter**: When to use labeled metrics
- **Prometheus integration**: Auto-registration and HTTP handler
- **Observability patterns**: Structured metrics for monitoring

### 3. Goroutine Patterns
```go
go startKeystrokeMonitoring() // Long-running OS event loop
go collectMetrics()           // Periodic metric collection
http.ListenAndServe(":8080", nil) // Main thread HTTP server
```
- **Concurrent design**: Separating concerns with goroutines
- **Background processing**: Non-blocking event capture
- **Channel communication**: Could be extended with channels for data flow

### 4. Native OS Integration
```c
CGEventRef keyboardCallback(CGEventTapProxy proxy, CGEventType type, CGEventRef event, void *refcon) {
    CGKeyCode keycode = (CGKeyCode)CGEventGetIntegerValueField(event, kCGKeyboardEventKeycode);
    // Key categorization logic
}
```
- **System-level programming**: Interfacing with OS APIs
- **Event-driven architecture**: Callback-based programming
- **Platform-specific code**: How to handle OS differences

### 5. Error Handling & Logging
```go
if result == 0 {
    log.Fatal("Failed to create keyboard event tap. Check Accessibility permissions!")
}
```
- **Graceful degradation**: Clear error messages for user action
- **System dependency handling**: Permission and configuration requirements

### 6. Multi-Process Architecture & IPC
```go
// File-based communication between processes
appData, _ := os.ReadFile("/tmp/current_app.json")
```
- **Process separation**: When to split functionality across processes
- **Inter-process communication**: Files, pipes, and message passing
- **Reliability patterns**: Combining multiple technologies for robust solutions

### 7. HTTP API Design & Chrome Extension Integration
```go
// HTTP endpoint for Chrome extension communication
func handleChromeUpdate(w http.ResponseWriter, r *http.Request) {
    // CORS headers for Chrome extension
    w.Header().Set("Access-Control-Allow-Origin", "*")
    chrome.UpdateCurrentDomain(update.Domain)
}
```
```javascript
// Chrome extension real-time domain updates
fetch('http://localhost:8080/chrome-update', {
    method: 'POST',
    body: JSON.stringify({domain: 'youtube_com'})
})
```
- **CORS handling**: Cross-origin requests from Chrome extension
- **Real-time communication**: Immediate domain updates via HTTP
- **Error handling**: Graceful fallbacks and debug interfaces

## Implementation Challenges & Solutions
*Documented: June 6, 2025*

### Challenge 1: CGO NSWorkspace Reliability
*Resolved: June 6, 2025*
**Problem**: NSWorkspace app detection worked in pure Swift but failed in CGO context
**Root Cause**: Threading, memory management, or framework linking issues in CGO
**Solution**: Hybrid architecture with Swift helper process + file-based IPC

**Key Learning**: Sometimes the "simple" approach (separate processes) is more reliable than complex integration

### Challenge 2: macOS Security & Permissions
*Resolved: June 6, 2025*
**Problem**: Accessibility permissions and app detection behavior varies by context
**Solution**: Test with minimal reproducible examples to isolate issues
**Tools**: Pure Swift tests, isolated CGO tests, permission verification

### Challenge 3: Real-time Multi-Language Integration
*Resolved: June 6, 2025*
**Problem**: Combining Go (metrics), C (keyboard events), and Swift (app detection)
**Solution**: 
- **C for low-level**: Keyboard event capture (CGEventTap)
- **Swift for high-level**: App detection (NSWorkspace)  
- **Go for business logic**: Metrics aggregation and HTTP serving
- **Files for communication**: Simple, debuggable IPC

### Challenge 4: Chrome Extension Communication
*Resolved: June 8, 2025*
**Problem**: Complex native messaging setup with multiple failure points
**Root Cause**: Chrome security restrictions, file system limitations, manifest complexity
**Solution**: Direct HTTP communication to Go application

**Key Learning**: Simple HTTP endpoints are often more reliable than complex platform-specific APIs

## Installation & Setup

### Chrome Extension Setup
1. **Load extension**: Chrome → Extensions → Developer mode → "Load unpacked" → Select `chrome-extension/` folder
2. **Reload tracker**: Run `./start.sh` to ensure HTTP endpoint is available
3. **Test communication**: Click extension icon → Should show "✅ HTTP Working"
4. **Grant permissions**: Extension will request tabs and storage permissions

### Grafana Chrome Domain Panel
Add this query to track time spent by domain:
```
increase(chrome_tab_total_time_seconds[$__range])
```

## Next Steps
*Updated: June 6, 2025*
Start with Phase 1 implementation, validate the complete pipeline works, then iterate through phases 2-4 based on learning and interest.