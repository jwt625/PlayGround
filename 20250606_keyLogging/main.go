package main

/*
#cgo CFLAGS: -x objective-c
#cgo LDFLAGS: -framework CoreGraphics -framework CoreFoundation
#include "keyboard.h"
#include "keyboard.c"
*/
import "C"
import (
	"encoding/json"
	"log"
	"net/http"
	"os"
	"strings"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promhttp"
)

// App info structure matching Swift helper
type AppInfo struct {
	Name      string  `json:"name"`
	BundleID  string  `json:"bundleId"`
	PID       int32   `json:"pid"`
	Timestamp float64 `json:"timestamp"`
}

var (
	// Keystroke metrics with app awareness
	keystrokesTotal = prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Name: "keystrokes_total",
			Help: "Total number of keystrokes recorded by key type and application",
		},
		[]string{"key_type", "app"},
	)

	// App time tracking metrics
	appSessionDuration = prometheus.NewHistogramVec(
		prometheus.HistogramOpts{
			Name:    "app_session_duration_seconds",
			Help:    "Duration of app focus sessions",
			Buckets: []float64{1, 5, 10, 30, 60, 300, 600, 1800, 3600}, // 1s to 1h
		},
		[]string{"app"},
	)

	appTotalTime = prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Name: "app_total_time_seconds",
			Help: "Total time spent in each application",
		},
		[]string{"app"},
	)

	currentAppGauge = prometheus.NewGaugeVec(
		prometheus.GaugeOpts{
			Name: "current_app_session_seconds",
			Help: "Current session duration for the active app",
		},
		[]string{"app"},
	)

	appSwitchTotal = prometheus.NewCounter(
		prometheus.CounterOpts{
			Name: "app_switch_total",
			Help: "Total number of application switches",
		},
	)
)

// App session tracking
type AppSession struct {
	AppName   string
	StartTime time.Time
}

var (
	currentSession *AppSession
	currentApp     string = "unknown"
)

func init() {
	prometheus.MustRegister(keystrokesTotal)
	prometheus.MustRegister(appSessionDuration)
	prometheus.MustRegister(appTotalTime)
	prometheus.MustRegister(currentAppGauge)
	prometheus.MustRegister(appSwitchTotal)
}

func main() {
	log.Println("Starting hybrid keystroke tracker...")
	log.Println("📊 Keystrokes: C/CGEventTap")
	log.Println("📱 App Detection: Swift Helper")
	log.Println("")
	log.Println("🚀 Start the Swift helper in another terminal:")
	log.Println("   swift app-detector-helper.swift")

	// Start keystroke monitoring in a goroutine
	go startKeystrokeMonitoring()

	// Start app info reader
	go readAppInfo()

	// Start app switch monitor
	go monitorAppSwitches()

	// Start metrics collection
	go collectMetrics()

	// Start HTTP server for Prometheus metrics
	http.Handle("/metrics", promhttp.Handler())
	log.Println("Metrics server starting on :8080/metrics")
	log.Fatal(http.ListenAndServe(":8080", nil))
}

func readAppInfo() {
	ticker := time.NewTicker(500 * time.Millisecond)
	defer ticker.Stop()

	for {
		select {
		case <-ticker.C:
			// Read current app from Swift helper
			if data, err := os.ReadFile("/tmp/keystroke_tracker_current_app.json"); err == nil {
				var appInfo AppInfo
				if err := json.Unmarshal(data, &appInfo); err == nil {
					newApp := sanitizeAppName(appInfo.Name)
					if newApp != currentApp {
						log.Printf("📱 App detected by Swift: %s", appInfo.Name)
						currentApp = newApp
					}
				}
			}
		}
	}
}

func startKeystrokeMonitoring() {
	log.Println("Starting C-based keyboard event monitoring...")
	log.Println("This requires Accessibility permissions for your terminal.")

	result := C.startEventTap()
	if result == 0 {
		log.Fatal("Failed to create keyboard event tap. Check Accessibility permissions!")
	}
}

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

func monitorAppSwitches() {
	ticker := time.NewTicker(1 * time.Second)
	defer ticker.Stop()

	var lastFileSize int64 = 0

	for {
		select {
		case <-ticker.C:
			// Check for new app switches in the log file
			if stat, err := os.Stat("/tmp/keystroke_tracker_app_switches.jsonl"); err == nil {
				if stat.Size() > lastFileSize {
					// File has grown, read new content
					if data, err := os.ReadFile("/tmp/keystroke_tracker_app_switches.jsonl"); err == nil {
						lines := strings.Split(string(data), "\n")
						for _, line := range lines {
							if strings.TrimSpace(line) == "" {
								continue
							}
							
							var switchInfo map[string]interface{}
							if err := json.Unmarshal([]byte(line), &switchInfo); err == nil {
								fromApp := sanitizeAppName(switchInfo["from"].(string))
								toApp := sanitizeAppName(switchInfo["to"].(string))
								timestamp := switchInfo["timestamp"].(float64)
								
								// Handle app session tracking
								handleAppSwitch(fromApp, toApp, time.Unix(int64(timestamp), 0))
							}
						}
					}
					lastFileSize = stat.Size()
				}
			}
		}
	}
}

func handleAppSwitch(fromApp, toApp string, switchTime time.Time) {
	// End current session if there was one
	if currentSession != nil && fromApp == currentSession.AppName {
		duration := switchTime.Sub(currentSession.StartTime).Seconds()
		if duration > 0.5 { // Ignore very short sessions
			appSessionDuration.WithLabelValues(currentSession.AppName).Observe(duration)
			appTotalTime.WithLabelValues(currentSession.AppName).Add(duration)
			log.Printf("⏱️  Session ended: %s (%.1fs)", currentSession.AppName, duration)
		}
	}

	// Start new session
	if toApp != "" && toApp != "unknown" {
		currentSession = &AppSession{
			AppName:   toApp,
			StartTime: switchTime,
		}
		appSwitchTotal.Inc()
		log.Printf("🔄 App switch: %s → %s", fromApp, toApp)
	}
}

func collectMetrics() {
	ticker := time.NewTicker(1 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-ticker.C:
			// Get keystroke counts by category
			letters := int(C.getAndResetLetters())
			numbers := int(C.getAndResetNumbers())
			special := int(C.getAndResetSpecial())

			// Update Prometheus counters with current app labels
			if letters > 0 {
				keystrokesTotal.WithLabelValues("letter", currentApp).Add(float64(letters))
			}
			if numbers > 0 {
				keystrokesTotal.WithLabelValues("number", currentApp).Add(float64(numbers))
			}
			if special > 0 {
				keystrokesTotal.WithLabelValues("special", currentApp).Add(float64(special))
			}

			total := letters + numbers + special
			if total > 0 {
				log.Printf("⌨️  App: %s | Total: %d (L:%d N:%d S:%d)", currentApp, total, letters, numbers, special)
			}

			// Update current session gauge
			if currentSession != nil {
				duration := time.Since(currentSession.StartTime).Seconds()
				currentAppGauge.WithLabelValues(currentSession.AppName).Set(duration)
			}
		}
	}
}

