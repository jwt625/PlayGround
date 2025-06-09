# Keystroke Tracker - Future Development TODO

*Last Updated: June 6, 2025*

## ✅ Completed (Current State)
*As of June 6, 2025*
- [x] Basic keystroke tracking with Go + CGO + macOS CGEventTap
- [x] Key categorization (letters, numbers, special keys)
- [x] App awareness using hybrid Swift helper + Go architecture
- [x] App session time tracking and metrics
- [x] Prometheus metrics integration
- [x] Grafana dashboard with real-time visualization
- [x] File-based IPC between Swift and Go processes

## 🚀 Phase 5: Multi-Machine Tracking
*Planned: June 6, 2025*

### Core Requirements
*Status: Planned*
- [ ] Add machine identification to all metrics labels
  - [ ] Use `os.Hostname()` or custom machine names
  - [ ] Update all metrics: `keystrokes_total{machine="macbook-pro", app="chrome", key_type="letter"}`

### Deployment Options
*Status: Planned*
- [ ] **Option A: Centralized Prometheus**
  - [ ] Set up central Prometheus/Grafana server
  - [ ] Configure each machine to be scraped by central Prometheus
  - [ ] Network configuration and firewall rules
  
- [ ] **Option B: Push Gateway**
  - [ ] Set up Prometheus Push Gateway
  - [ ] Modify Go tracker to push metrics instead of expose
  - [ ] Handle network failures and retry logic

- [ ] **Option C: Remote Write**
  - [ ] Configure direct remote write to central Prometheus
  - [ ] Authentication and security setup

### Infrastructure
*Status: Planned*
- [ ] Docker containers for easy deployment across machines
- [ ] Systemd services for auto-start on Linux
- [ ] Launch agents for auto-start on macOS
- [ ] Cloud hosting options for central infrastructure

## 🌐 Phase 6: Chrome Fine-Grained Tracking
*Planned: June 6, 2025*

### Chrome Extension Development
*Status: Planned*
- [ ] **Basic Extension Setup**
  - [ ] Create manifest.json with required permissions
  - [ ] Set up content scripts and background service worker
  - [ ] Test basic tab detection and URL access

- [ ] **Core Functionality**
  - [ ] Track current domain (google.com, github.com, etc.)
  - [ ] Detect tab switches and timing
  - [ ] Monitor time spent per domain/page
  - [ ] Track URL path changes for deeper insights

- [ ] **Integration with Go Tracker**
  - [ ] Implement Chrome Native Messaging API
  - [ ] Create native messaging host for Go application
  - [ ] Protocol design for extension ↔ Go communication
  - [ ] Update Go tracker to handle Chrome-specific metrics

### Advanced Chrome Tracking
*Status: Planned*
- [ ] User interaction tracking (clicks, scrolls, form interactions)
- [ ] Reading time estimation
- [ ] Productivity scoring (work vs entertainment domains)
- [ ] Privacy controls and data filtering

### New Metrics to Add
*Status: Planned*
- [ ] `chrome_time_spent_seconds{domain="github.com"}`
- [ ] `chrome_tab_switches_total{from_domain="", to_domain=""}`
- [ ] `chrome_page_views_total{domain="", path=""}`
- [ ] `chrome_interactions_total{domain="", type="click|scroll|form"}`

## 🔧 Technical Improvements
*Planned: June 6, 2025*

### Current System Enhancements
*Status: Planned*
- [ ] Better error handling and retry logic
- [ ] Configuration file support (YAML/JSON)
- [ ] Logging improvements and structured logging
- [ ] Performance optimizations for high-frequency typing
- [ ] Memory usage monitoring and optimization

### Security & Privacy
*Status: Planned*
- [ ] Data encryption for sensitive metrics
- [ ] Privacy controls and filtering options
- [ ] Audit logging for data access
- [ ] GDPR compliance considerations

## 📊 Analytics & Insights
*Planned: June 6, 2025*

### Advanced Dashboards
*Status: Planned*
- [ ] Productivity heat maps by time of day
- [ ] App usage patterns and trends
- [ ] Typing speed and efficiency metrics
- [ ] Cross-machine productivity comparison
- [ ] Chrome browsing behavior analysis

### Machine Learning Opportunities
*Status: Planned*
- [ ] Productivity pattern recognition
- [ ] Focus time prediction
- [ ] Anomaly detection for unusual behavior
- [ ] Personalized productivity recommendations

## 🚀 Deployment & Operations
*Planned: June 6, 2025*

### Production Readiness
*Status: Planned*
- [ ] Health checks and monitoring
- [ ] Graceful shutdown handling
- [ ] Configuration management
- [ ] Backup and recovery procedures
- [ ] Documentation and runbooks

### Scaling Considerations
*Status: Planned*
- [ ] Database backend for historical data
- [ ] Time series data retention policies
- [ ] Horizontal scaling for multiple users
- [ ] Performance testing and benchmarking

---

## ✅ Phase 7: Code Refactoring & Architecture Improvements
*Completed: June 6, 2025*

### Core Requirements
*Status: Complete*
- [x] Split monolithic main.go (345 lines) into logical packages
- [x] Create clean package structure with proper separation of concerns
- [x] Maintain all existing functionality while improving maintainability
- [x] Fix CGO compilation issues and import path problems

### Package Structure Created
*Status: Complete*
- [x] **types/** - Shared data structures (AppInfo, AppSession, CounterState)
- [x] **metrics/** - Prometheus metric definitions and persistence logic
- [x] **app/** - Application detection, session tracking, and file monitoring
- [x] **keyboard/** - CGO-based keyboard capture and C implementation
- [x] **main.go** - Reduced to 35 lines, entry point only

### Technical Challenges & Solutions
*Status: Complete*

#### Challenge 1: Go Module Import Paths
*Resolved: June 6, 2025*
**Problem**: Relative imports like `"./app"` and `"../types"` fail in Go modules
**Error**: `"./app" is relative, but relative import paths are not supported in module mode`
**Solution**: Use module-based imports like `"keystroke-tracker/app"`
**Learning**: Always use full module paths in Go packages, never relative paths

#### Challenge 2: CGO Duplicate Symbol Errors
*Resolved: June 6, 2025*
**Problem**: Duplicate symbol errors when linking C code
**Error**: `duplicate symbol '_getAndResetLetters'` and 4 other C functions
**Root Cause**: Including both `keyboard.h` and `keyboard.c` in CGO comment block
**Solution**: Only include header file (`keyboard.h`), let linker handle implementation
**Learning**: In CGO, include headers only - don't include .c files directly

#### Challenge 3: C File Organization
*Resolved: June 6, 2025*
**Problem**: `C source files not allowed when not using cgo` when C files in root
**Solution**: Move `keyboard.c` and `keyboard.h` into `keyboard/` package directory
**Learning**: C files must be co-located with the Go package that uses CGO

### Enhanced Metrics Implementation
*Status: Complete*
- [x] Fine-grained histogram buckets (32 buckets from 0.5s to 4h)
  - Quick switches: 0.5s - 30s (7 buckets)
  - Work sessions: 1-60min in 5-min intervals (13 buckets)
  - Extended sessions: 1-2h in 10-min intervals (6 buckets)
  - Long sessions: 2-4h in 20-min intervals (6 buckets)
- [x] App switch events tracking: `app_switch_events_total{from_app="X", to_app="Y"}`
- [x] Data persistence through Docker named volumes

### Key Go Programming Lessons Learned
*Documented: June 6, 2025*

#### 1. Package Organization Best Practices
```go
// Good: Module-based imports
import "keystroke-tracker/metrics"

// Bad: Relative imports (fails)
import "../metrics"
```

#### 2. CGO Integration Patterns
```go
/*
#cgo CFLAGS: -x objective-c
#cgo LDFLAGS: -framework CoreGraphics -framework CoreFoundation
#include "keyboard.h"  // ✅ Header only
// #include "keyboard.c"  // ❌ Causes duplicate symbols
*/
import "C"
```

#### 3. Prometheus Metrics Separation
- **Centralized definitions**: All metrics in `metrics/definitions.go`
- **Registration function**: `metrics.RegisterMetrics()` for clean initialization
- **Package-level access**: Exported variables for cross-package use

### Persistence Architecture Improvements
*Status: Complete*
- [x] Docker named volumes for Prometheus data persistence
- [x] Survives container restarts and system reboots
- [x] 200-hour (8+ days) data retention configured
- [x] Go app state persistence framework (partial implementation)

### Results & Benefits
*Achieved: June 6, 2025*
- **Maintainability**: main.go reduced from 345 to 35 lines
- **Modularity**: Clean separation by functional area
- **Testability**: Each package can be tested independently
- **Extensibility**: Easy to add new features (trackpad tracking planned)
- **Build Success**: All compilation issues resolved

## 🚀 Phase 8: Trackpad Click Tracking
*Completed: June 6, 2025*

### Core Requirements
*Status: Complete*
- [x] Add trackpad/mouse click event capture using Swift
- [x] Create new metrics: `mouse_clicks_total{button_type="left|right|middle", app="X"}`
- [x] Integrate with existing file-based IPC architecture
- [x] Add monitoring goroutine in Go server

### Implementation Strategy
*Status: Complete*
- [x] Extend Swift unified tracker with CGEventTap mouse monitoring
- [x] Create `/tmp/keystroke_tracker_trackpad_events.jsonl` for click events
- [x] Add `trackpad/` package to clean architecture
- [x] Follow same pattern as keyboard capture but for mouse events

## 🌐 Phase 9: Chrome Extension Fine-Grained Tracking
*Started: June 6, 2025*

### Core Requirements
*Status: In Progress*
- [ ] **Chrome Extension Development**
  - [ ] Create manifest.json with tabs and activeTab permissions
  - [ ] Background script for tab switch detection
  - [ ] Content script for URL change monitoring within tabs
  - [ ] Domain extraction and sanitization logic
  
- [ ] **File-based IPC Integration** 
  - [ ] Write to `/tmp/keystroke_tracker_chrome_events.jsonl` for tab switches
  - [ ] Write to `/tmp/keystroke_tracker_current_domain.json` for current domain
  - [ ] Follow same JSONL pattern as Swift tracker

- [ ] **Go Server Integration**
  - [ ] Create `chrome/` package to monitor Chrome JSONL files
  - [ ] Enhance existing metrics with domain labels when app="google_chrome"
  - [ ] Add Chrome-specific metrics: `chrome_tab_switches_total{from_domain, to_domain}`

### Enhanced Metrics Structure
*Status: Planned*
```promql
# Existing metrics enhanced with domain context
keystrokes_total{app="google_chrome", domain="github.com", key_type="letter"}
mouse_clicks_total{app="google_chrome", domain="stackoverflow.com", button_type="left"}

# New Chrome-specific metrics
chrome_tab_switches_total{from_domain="github.com", to_domain="stackoverflow.com"}
chrome_time_spent_seconds{domain="github.com"}
```

### Technical Considerations
*Status: Planned*
- [ ] Chrome extension cannot directly write to `/tmp/` - needs Native Messaging Host
- [ ] Alternative: Use Chrome storage API + periodic sync to Go server
- [ ] Domain privacy filtering and sanitization
- [ ] Performance optimization for high-frequency tab switching

---

**Next Session Priority:**
Complete Chrome extension development with proper file system access, then integrate with existing Go observability architecture.