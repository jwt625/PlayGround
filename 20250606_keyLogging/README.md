# Keystroke Tracker with Prometheus & Grafana

*Last Updated: June 8, 2025*

A comprehensive productivity tracking system built with Go, Prometheus, and Grafana featuring keystroke monitoring, app detection, mouse tracking, and Chrome domain analytics.

## 🎯 What This Project Does
*Features as of: June 8, 2025*

- **Captures keystrokes** passively (read-only, no interference)
- **Categorizes keys** by type (letters, numbers, special keys)
- **Tracks applications** and session durations
- **Monitors mouse clicks** and trackpad events
- **Chrome domain tracking** with real-time tab switching detection
- **Exposes metrics** via Prometheus with domain-aware labels
- **Visualizes data** in real-time Grafana dashboards
- **Tracks productivity patterns** across apps and websites

## 📋 Prerequisites
*Updated: June 6, 2025*

You'll need to install several tools. Don't worry - we'll walk through each one!

### 1. Install Go (Programming Language)

**macOS:**
```bash
# Using Homebrew (recommended)
brew install go

# Or download from: https://golang.org/dl/
```

**Verify installation:**
```bash
go version
# Should show: go version go1.21.x darwin/amd64 (or similar)
```

### 2. Install Docker (For Prometheus & Grafana)

**macOS:**
1. Download Docker Desktop from: https://www.docker.com/products/docker-desktop/
2. Install the .dmg file
3. Launch Docker Desktop from Applications
4. Wait for Docker to start (whale icon in menu bar)

**Verify installation:**
```bash
docker --version
# Should show: Docker version 24.x.x

docker-compose --version
# Should show: Docker Compose version v2.x.x
```

### 3. Install Git (Version Control)

**macOS:**
```bash
# Usually pre-installed, but if not:
brew install git

# Or download from: https://git-scm.com/download/mac
```

## 🚀 Quick Start Guide
*Updated: June 6, 2025*

### Step 1: Clone & Setup

```bash
# Clone this repository
git clone <your-repo-url>
cd keystroke-tracker

# Initialize Go module (if not done)
go mod init keystroke-tracker

# Install Go dependencies
go get github.com/prometheus/client_golang/prometheus
go get github.com/prometheus/client_golang/prometheus/promhttp
```

### Step 2: Grant Accessibility Permissions

**This is CRITICAL for keystroke capture to work:**

1. Open **System Settings** → **Privacy & Security** → **Accessibility**
2. Click the **"+" button**
3. Navigate to your **Terminal** app (e.g., Terminal.app, iTerm.app, or VS Code)
4. **Enable** the checkbox next to your terminal
5. **Restart your terminal** after granting permissions

### Step 3: Start the Services

**Start everything with one command (Recommended):**
```bash
./start.sh
```

**Or with detailed logs:**
```bash
./start-with-logs.sh
```

You should see:
```
🚀 Starting Keystroke Tracker...
🔧 Checking Go installation...
🔨 Checking Go binaries...
🐳 Checking Docker containers...
📱 Starting Swift unified tracker (app + trackpad)...
🌐 Chrome extension will use HTTP endpoint
⌨️  Starting Go keystroke tracker...
🎯 All processes running!
```

### Step 4: Setup Chrome Extension (Optional)

**For Chrome domain tracking:**
1. Open Chrome → Extensions → **Developer mode** (toggle on)
2. Click **"Load unpacked"** → Select the `chrome-extension/` folder
3. **Click the extension icon** → Should show "✅ HTTP Working"
4. **Grant permissions** when prompted (tabs, storage)

### Step 5: Access the Dashboards

**Grafana Dashboard:**
1. Open: http://localhost:3001
2. Login: `admin` / `admin`
3. Import dashboard from `dashboard-app-aware-v3.json`

**Prometheus (Optional):**
- Open: http://localhost:9090
- Try queries: `keystrokes_total`, `chrome_tab_total_time_seconds`

## 🧪 Testing the System
*Updated: June 8, 2025*

1. **Type anywhere** on your computer
2. **Watch the terminal** - you should see logs like:
   ```
   ⌨️  App: code | Total: 15 (L:12 N:2 S:1)
   🖱️  Mouse: left click in code
   🌐 Chrome domain via HTTP: youtube_com
   ```
3. **Switch between apps** - should see app detection
4. **Use Chrome with extension** - should see domain tracking
5. **Check Grafana** - all metrics update in real-time

## 📁 Project Structure
*Updated: June 6, 2025*

```
keystroke-tracker/
├── README.md                     # This file
├── CLAUDE.md                     # Detailed implementation notes
├── go.mod                        # Go dependencies
├── main-categorized-fixed.go     # Main application
├── docker-compose.yml            # Prometheus & Grafana services
├── prometheus.yml                # Prometheus configuration
└── dashboard-categorized.json    # Grafana dashboard definition
```

## 🔧 Troubleshooting
*Updated: June 6, 2025*

### "Failed to create keyboard event tap"
- **Solution**: Grant Accessibility permissions (Step 2 above)
- **Still failing?** Restart your terminal after granting permissions

### "Docker command not found"
- **Solution**: Install Docker Desktop and make sure it's running
- **Check**: See the whale icon in your menu bar

### "Port already in use"
- **Solution**: Kill existing processes:
  ```bash
  # Check what's using the port
  lsof -i :8080
  # Kill the process
  kill <PID>
  ```

### Grafana shows "No data"
- **Check**: Go server is running and showing keystroke logs
- **Check**: Prometheus data source URL is correct
- **Try**: Type something and wait 5-10 seconds

### "Go command not found"
- **Solution**: Install Go (see Prerequisites)
- **Check**: Run `go version`

## 🎛️ Configuration
*Updated: June 6, 2025*

### Change Update Frequency

**Prometheus scraping** (edit `prometheus.yml`):
```yaml
scrape_interval: 1s  # Change from 5s to 1s
```

**Grafana refresh** (in dashboard):
- Top-right dropdown: change from "5s" to "1s"

**Restart after changes:**
```bash
docker-compose restart prometheus
```

## 📊 Understanding the Data
*Updated: June 6, 2025*

### Metrics Available
- `keystrokes_total{key_type="letter", app="code", domain=""}` - A-Z keys by app
- `mouse_clicks_total{button_type="left", app="chrome", domain="youtube_com"}` - Mouse clicks  
- `app_total_time_seconds{app="code"}` - Time spent per application
- `chrome_tab_total_time_seconds{domain="github_com"}` - Time spent per domain
- `chrome_tab_session_duration_seconds` - Individual Chrome session durations

### Useful Prometheus Queries
```bash
# Typing speed (keystrokes per minute)
rate(keystrokes_total[1m]) * 60

# Time spent by application (exclude system apps)
increase(app_total_time_seconds{app!~"loginwindow|finder|dock|.*window.*"}[$__range])

# Time spent by Chrome domain
increase(chrome_tab_total_time_seconds[$__range])

# Chrome session duration percentiles
histogram_quantile(0.95, chrome_tab_session_duration_seconds)
```

## 🛑 Privacy & Security
*Updated: June 6, 2025*

- **Local only**: All data stays on your machine
- **Read-only**: No keystroke content is logged (only counts)
- **No network**: No data sent to external servers
- **Passive**: Doesn't interfere with normal typing

## 🔄 Stopping the System
*Updated: June 6, 2025*

```bash
# Stop the Go application
Ctrl+C (in the Go terminal)

# Stop Prometheus & Grafana
docker-compose down
```

## 📚 Learning Resources
*Updated: June 6, 2025*

- **Go Language**: https://tour.golang.org/
- **Prometheus**: https://prometheus.io/docs/
- **Grafana**: https://grafana.com/docs/
- **Docker**: https://docs.docker.com/get-started/

## 🎓 Next Steps
*Updated: June 6, 2025*

Once you have the basic system running, explore:
- **Phase 3**: Time-based insights and typing speed calculation
- **Phase 4**: Application-specific tracking
- **Custom dashboards**: Create your own visualizations
- **Advanced queries**: Complex Prometheus analytics

## 🆘 Need Help?
*Updated: June 6, 2025*

If you get stuck:
1. **Check the logs** - both Go terminal and Docker logs
2. **Verify permissions** - Accessibility settings are crucial
3. **Restart services** - Sometimes a fresh start helps
4. **Check the troubleshooting section** above

Happy keystroke tracking! 🚀