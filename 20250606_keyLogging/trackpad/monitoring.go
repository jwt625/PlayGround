package trackpad

import (
	"encoding/json"
	"log"
	"os"
	"strings"
	"time"

	"keystroke-tracker/chrome"
	"keystroke-tracker/metrics"
)

// ClickEvent represents a trackpad click event from Swift
type ClickEvent struct {
	ButtonType string  `json:"buttonType"`
	App        string  `json:"app"`
	Timestamp  float64 `json:"timestamp"`
	ClickCount int     `json:"clickCount"`
}

// getDomainForApp returns the domain label for metrics based on the current app
func getDomainForApp(appName string) string {
	if appName == "google_chrome" {
		domain := chrome.GetCurrentDomain()
		if domain == "" {
			return "unknown"
		}
		return domain
	}
	// For non-Chrome apps, use empty domain
	return ""
}

// MonitorClickEvents watches the trackpad events file for new clicks
func MonitorClickEvents() {
	clickEventsFile := "/tmp/keystroke_tracker_trackpad_events.jsonl"
	ticker := time.NewTicker(100 * time.Millisecond) // Check more frequently than keystrokes
	defer ticker.Stop()

	var lastFileSize int64 = 0

	for {
		select {
		case <-ticker.C:
			// Check for new click events in the log file
			if stat, err := os.Stat(clickEventsFile); err == nil {
				if stat.Size() > lastFileSize {
					// File has grown, read new content
					if file, err := os.Open(clickEventsFile); err == nil {
						file.Seek(lastFileSize, 0) // Start reading from where we left off
						newData := make([]byte, stat.Size()-lastFileSize)
						file.Read(newData)
						file.Close()
						lines := strings.Split(string(newData), "\n")
						for _, line := range lines {
							if strings.TrimSpace(line) == "" {
								continue
							}

							var clickEvent ClickEvent
							if err := json.Unmarshal([]byte(line), &clickEvent); err == nil {
								// Convert app name to clean format (same as keystrokes)
								appName := sanitizeAppName(clickEvent.App)
								
								// Get domain for current app (Chrome-aware)
								domain := getDomainForApp(appName)
								
								// Increment Prometheus counter with domain
								metrics.MouseClicksTotal.WithLabelValues(clickEvent.ButtonType, appName, domain).Inc()
								
								domainInfo := ""
								if domain != "" {
									domainInfo = " (" + domain + ")"
								}
								log.Printf("🖱️ %s click in %s%s (Prometheus updated)", 
									strings.Title(clickEvent.ButtonType), appName, domainInfo)
							}
						}
					}
					lastFileSize = stat.Size()
				}
			}
		}
	}
}

// sanitizeAppName cleans app names for Prometheus labels (same as app package)
func sanitizeAppName(appName string) string {
	if appName == "" {
		return "unknown"
	}
	// Clean app name for Prometheus labels
	appName = strings.ReplaceAll(appName, " ", "_")
	appName = strings.ReplaceAll(appName, "-", "_")
	appName = strings.ReplaceAll(appName, ".", "_")
	appName = strings.ToLower(appName)
	return appName
}