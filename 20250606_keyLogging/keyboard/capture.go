package keyboard

/*
#cgo CFLAGS: -x objective-c
#cgo LDFLAGS: -framework CoreGraphics -framework CoreFoundation
#include "keyboard.h"
*/
import "C"
import (
	"log"
	"time"

	"keystroke-tracker/app"
	"keystroke-tracker/metrics"
)

// StartKeystrokeMonitoring begins CGO-based keyboard event capture
func StartKeystrokeMonitoring() {
	log.Println("Starting C-based keyboard event monitoring...")
	log.Println("This requires Accessibility permissions for your terminal.")

	result := C.startEventTap()
	if result == 0 {
		log.Fatal("Failed to create keyboard event tap. Check Accessibility permissions!")
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

			// Update Prometheus counters with current app labels
			if letters > 0 {
				metrics.KeystrokesTotal.WithLabelValues("letter", app.CurrentApp).Add(float64(letters))
			}
			if numbers > 0 {
				metrics.KeystrokesTotal.WithLabelValues("number", app.CurrentApp).Add(float64(numbers))
			}
			if special > 0 {
				metrics.KeystrokesTotal.WithLabelValues("special", app.CurrentApp).Add(float64(special))
			}

			total := letters + numbers + special
			if total > 0 {
				log.Printf("⌨️  App: %s | Total: %d (L:%d N:%d S:%d)", app.CurrentApp, total, letters, numbers, special)
			}

			// Update current session gauge
			app.UpdateCurrentSessionGauge()
		}
	}
}