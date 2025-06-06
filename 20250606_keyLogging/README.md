# Keystroke Tracker with Prometheus & Grafana

*Last Updated: June 6, 2025*

A real-time keystroke monitoring system built with Go, Prometheus, and Grafana to learn backend observability and data visualization.

## 🎯 What This Project Does
*Features as of: June 6, 2025*

- **Captures keystrokes** passively (read-only, no interference)
- **Categorizes keys** by type (letters, numbers, special keys)
- **Exposes metrics** via Prometheus
- **Visualizes data** in real-time Grafana dashboards
- **Tracks typing patterns** and productivity insights

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

**Terminal 1 - Start Prometheus & Grafana:**
```bash
docker-compose up -d
```

**Terminal 2 - Start the Go keystroke tracker:**
```bash
go run main-categorized-fixed.go
```

You should see:
```
Starting keystroke tracker with categorization...
Starting native macOS keyboard event monitoring with categorization...
Metrics server starting on :8080/metrics
```

### Step 4: Access the Dashboards

**Grafana Dashboard:**
1. Open: http://localhost:3001
2. Login: `admin` / `admin`
3. Go to **Configuration** → **Data Sources** → **Add data source**
4. Select **Prometheus**
5. Set URL: `http://host.docker.internal:9090`
  - this is because Grafana is running inside the docker container
6. Click **Save & Test**

**Import Dashboard:**
1. Click **"+" → Import**
2. Copy content from `dashboard-categorized.json`
3. Paste and click **Load** → **Import**

**Prometheus (Optional):**
- Open: http://localhost:9090
- Try query: `keystrokes_total`

## 🧪 Testing the System
*Updated: June 6, 2025*

1. **Type anywhere** on your computer
2. **Watch the terminal** - you should see logs like:
   ```
   Total: 15 (Letters:12 Numbers:2 Special:1)
   ```
3. **Check Grafana** - graphs should update in real-time
4. **Try different typing** - code vs text vs numbers

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
- `keystrokes_total{key_type="letter"}` - A-Z keys
- `keystrokes_total{key_type="number"}` - 0-9 keys  
- `keystrokes_total{key_type="special"}` - Space, Enter, punctuation, etc.

### Useful Prometheus Queries
```bash
# Typing speed (keystrokes per minute)
rate(keystrokes_total[1m]) * 60

# Letters vs numbers ratio
rate(keystrokes_total{key_type="letter"}[1m]) / rate(keystrokes_total{key_type="number"}[1m])

# Total keystrokes in last hour
increase(keystrokes_total[1h])
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