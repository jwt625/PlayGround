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

**Next Session Priority:**
Start with either Multi-Machine setup or Chrome extension development based on immediate needs.