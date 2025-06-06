# Keystroke Tracking with Prometheus & Grafana

## Project Overview
Building a passive keystroke tracker in Go to learn backend observability with Prometheus and Grafana visualization.

## Implementation Plan

### Phase 1: Basic Keystroke Counter (MVP - ~30 minutes)
**Goal**: Get end-to-end pipeline working with immediate visual feedback

**Tasks**:
- [ ] Set up Go project structure
- [ ] Implement basic keystroke event listener (OS-specific, read-only)
- [ ] Add Prometheus counter metric: `keystrokes_total`
- [ ] Create HTTP endpoint to expose metrics (`/metrics`)
- [ ] Set up Prometheus configuration to scrape our service
- [ ] Create basic Grafana dashboard showing keystrokes over time
- [ ] Test: Type and watch graph update in real-time

**Key Libraries**:
- `github.com/prometheus/client_golang` for metrics
- Platform-specific keyboard event library (TBD based on OS)

### Phase 2: Basic Categorization
**Goal**: Add more meaningful metrics with labels

**Tasks**:
- [ ] Categorize keystrokes by type (letters, numbers, special keys)
- [ ] Update metric: `keystrokes_total{key_type="letter|number|special"}`
- [ ] Enhance Grafana dashboard with breakdown by key type

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

## Success Criteria for Phase 1
- [ ] Service runs in background without affecting other programs
- [ ] Prometheus successfully scrapes keystroke metrics
- [ ] Grafana dashboard shows real-time keystroke activity
- [ ] Can see immediate feedback when typing

## Next Steps
Start with Phase 1 implementation, validate the complete pipeline works, then iterate through phases 2-4 based on learning and interest.