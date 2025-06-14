package keyboard

/*
#cgo CFLAGS: -x objective-c
#cgo LDFLAGS: -framework CoreGraphics -framework CoreFoundation
#include "keyboard.h"
*/
import "C"
import (
	"encoding/json"
	"log"
	"os"
	"time"

	"keystroke-tracker/app"
	"keystroke-tracker/chrome"
	"keystroke-tracker/metrics"
	"keystroke-tracker/types"
)

// File path for keystroke interval logging  
const keystrokeIntervalsFile = "/tmp/keystroke_tracker_keystroke_intervals.jsonl"

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

// StartKeystrokeMonitoring begins CGO-based keyboard event capture
func StartKeystrokeMonitoring() {
	log.Println("Starting C-based keyboard event monitoring...")
	log.Println("This requires Accessibility permissions for your terminal.")

	result := C.startEventTap()
	if result == 0 {
		log.Fatal("Failed to create keyboard event tap. Check Accessibility permissions!")
	}
}

// logKeystrokeInterval logs keystroke activity for a 1-second interval
func logKeystrokeInterval(letters, numbers, special int, appName string) {
	total := letters + numbers + special
	if total == 0 {
		return // Don't log empty intervals
	}

	interval := types.KeystrokeInterval{
		App:       appName,
		Letters:   letters,
		Numbers:   numbers,
		Special:   special,
		Total:     total,
		Timestamp: float64(time.Now().Unix()),
	}

	jsonData, err := json.Marshal(interval)
	if err != nil {
		log.Printf("❌ Error marshaling keystroke interval: %v", err)
		return
	}

	line := string(jsonData) + "\n"

	// Append to file
	if _, err := os.Stat(keystrokeIntervalsFile); os.IsNotExist(err) {
		// File doesn't exist, create it
		if err := os.WriteFile(keystrokeIntervalsFile, []byte(line), 0644); err != nil {
			log.Printf("❌ Error creating keystroke intervals file: %v", err)
		}
	} else {
		// File exists, append to it
		file, err := os.OpenFile(keystrokeIntervalsFile, os.O_APPEND|os.O_WRONLY, 0644)
		if err != nil {
			log.Printf("❌ Error opening keystroke intervals file: %v", err)
			return
		}
		defer file.Close()

		if _, err := file.WriteString(line); err != nil {
			log.Printf("❌ Error writing to keystroke intervals file: %v", err)
		}
	}
}

// CollectMetrics periodically collects keystroke counts and updates metrics
func CollectMetrics() {
	ticker := time.NewTicker(1 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-ticker.C:
			// Get keystroke counts by category
			letters := int(C.getAndResetLetters())
			numbers := int(C.getAndResetNumbers())
			special := int(C.getAndResetSpecial())

			// Get domain for current app (Chrome-aware)
			domain := getDomainForApp(app.CurrentApp)

			// Update Prometheus counters with current app and domain labels
			if letters > 0 {
				metrics.KeystrokesTotal.WithLabelValues("letter", app.CurrentApp, domain).Add(float64(letters))
			}
			if numbers > 0 {
				metrics.KeystrokesTotal.WithLabelValues("number", app.CurrentApp, domain).Add(float64(numbers))
			}
			if special > 0 {
				metrics.KeystrokesTotal.WithLabelValues("special", app.CurrentApp, domain).Add(float64(special))
			}

			// ALSO log interval events for persistence & detailed analysis
			logKeystrokeInterval(letters, numbers, special, app.CurrentApp)

			// AND expose as Prometheus metrics for observability - SET the interval activity
			metrics.KeystrokeIntervalActivity.WithLabelValues(app.CurrentApp, "letter", domain).Set(float64(letters))
			metrics.KeystrokeIntervalActivity.WithLabelValues(app.CurrentApp, "number", domain).Set(float64(numbers))
			metrics.KeystrokeIntervalActivity.WithLabelValues(app.CurrentApp, "special", domain).Set(float64(special))

			total := letters + numbers + special
			if total > 0 {
				domainInfo := ""
				if domain != "" {
					domainInfo = " | Domain: " + domain
				}
				log.Printf("⌨️  App: %s%s | Total: %d (L:%d N:%d S:%d)", app.CurrentApp, domainInfo, total, letters, numbers, special)
			}

			// Update current session gauge
			app.UpdateCurrentSessionGauge()
			
			// Update Chrome tab session gauge if we're in Chrome
			if app.CurrentApp == "google_chrome" {
				chrome.UpdateCurrentTabGauge()
			}
		}
	}
}