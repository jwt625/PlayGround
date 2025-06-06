# Keystroke Tracking with Prometheus & Grafana

## Project Overview
Building a passive keystroke tracker in Go to learn backend observability with Prometheus and Grafana visualization.

## Implementation Plan

### Phase 1: Basic Keystroke Counter ✅ COMPLETED
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
**Goal**: Calculate derived metrics for better analysis

**Tasks**:
- [ ] Add keystrokes per minute calculation using Prometheus queries
- [ ] Create hourly/daily pattern visualizations
- [ ] Add typing speed trends over time

### Phase 4: Application Awareness
**Goal**: Track typing activity per application

**Tasks**:
- [ ] Detect active application during keystroke events
- [ ] Add app label: `keystrokes_total{key_type="letter", app="vscode"}`
- [ ] Create application-specific dashboards

## Technical Approach

### Security & Privacy
- **Passive monitoring**: Read-only event listening, no keystroke interception
- **No key content logging**: Only count events, never log actual keys pressed
- **Local only**: All data stays on local machine
- **Configurable exclusions**: Option to exclude sensitive applications

### Architecture
```
[Keyboard Events] → [Go Service] → [Prometheus Metrics] → [Grafana Dashboard]
                         ↓
                  [HTTP /metrics endpoint]
```

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
- **Phase 1**: ✅ Complete - End-to-end pipeline working
- **Phase 2**: ✅ Complete - Key categorization with labels
- **Phase 3**: 🚧 In Progress - Time-based insights
- **Access URLs**:
  - Grafana: http://localhost:3001 (admin/admin)
  - Prometheus: http://localhost:9090
  - Metrics: http://localhost:8080/metrics

## Key Go Programming Concepts Worth Learning

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

## Next Steps
Start with Phase 1 implementation, validate the complete pipeline works, then iterate through phases 2-4 based on learning and interest.