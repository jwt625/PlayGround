package app

import (
	"log"
	"time"

	"keystroke-tracker/metrics"
	"keystroke-tracker/types"
)

var CurrentSession *types.AppSession

// HandleAppSwitch manages app session tracking and metrics
func HandleAppSwitch(fromApp, toApp string, switchTime time.Time) {
	// Only count as a switch if we're actually switching between different apps
	isActualSwitch := fromApp != "" && toApp != "" && fromApp != "unknown" && toApp != "unknown" && fromApp != toApp

	// End current session if there was one
	if CurrentSession != nil && fromApp == CurrentSession.AppName {
		duration := switchTime.Sub(CurrentSession.StartTime).Seconds()
		if duration > 0.5 { // Ignore very short sessions
			metrics.AppSessionDuration.WithLabelValues(CurrentSession.AppName).Observe(duration)
			metrics.AppTotalTime.WithLabelValues(CurrentSession.AppName).Add(duration)
			log.Printf("⏱️  Session ended: %s (%.1fs)", CurrentSession.AppName, duration)
		}
	}

	// Start new session
	if toApp != "" && toApp != "unknown" {
		CurrentSession = &types.AppSession{
			AppName:   toApp,
			StartTime: switchTime,
		}

		// Only increment switch counter for actual switches between different apps
		if isActualSwitch {
			metrics.AppSwitchTotal.Inc()
			metrics.AppSwitchEvents.WithLabelValues(fromApp, toApp).Inc()
			log.Printf("🔄 App switch: %s → %s", fromApp, toApp)
		} else {
			log.Printf("📱 App detected: %s", toApp)
		}
	}
}

// UpdateCurrentSessionGauge updates the current session duration gauge
func UpdateCurrentSessionGauge() {
	if CurrentSession != nil {
		duration := time.Since(CurrentSession.StartTime).Seconds()
		metrics.CurrentAppGauge.WithLabelValues(CurrentSession.AppName).Set(duration)
	}
}