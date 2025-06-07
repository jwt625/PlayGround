package app

import (
	"encoding/json"
	"log"
	"os"
	"strings"
	"time"

	"keystroke-tracker/metrics"
	"keystroke-tracker/types"
)

var (
	CurrentApp  string = "unknown"
	PreviousApp string = "unknown"
)

// ReadAppInfo continuously reads app info from Swift helper
func ReadAppInfo() {
	ticker := time.NewTicker(500 * time.Millisecond)
	defer ticker.Stop()

	for {
		select {
		case <-ticker.C:
			// Read current app from Swift helper
			if data, err := os.ReadFile("/tmp/keystroke_tracker_current_app.json"); err == nil {
				var appInfo types.AppInfo
				if err := json.Unmarshal(data, &appInfo); err == nil {
					newApp := SanitizeAppName(appInfo.Name)
					if newApp != CurrentApp {
						UpdateCurrentApp(newApp, appInfo.Name)
					}
				}
			}
		}
	}
}

// UpdateCurrentApp handles app changes and records switch events
func UpdateCurrentApp(newApp, originalName string) {
	if newApp != CurrentApp && CurrentApp != "unknown" {
		// Record the switch event
		metrics.AppSwitchEvents.WithLabelValues(CurrentApp, newApp).Inc()
		log.Printf("🔄 App switch: %s → %s", CurrentApp, newApp)
	} else {
		log.Printf("📱 App detected by Swift: %s", originalName)
	}

	PreviousApp = CurrentApp
	CurrentApp = newApp
}

// SanitizeAppName cleans app names for Prometheus labels
func SanitizeAppName(appName string) string {
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