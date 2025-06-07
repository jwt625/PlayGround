package metrics

import (
	"encoding/json"
	"log"
	"os"

	"keystroke-tracker/types"
)

// LoadPersistedCounters loads previous counter state from disk
func LoadPersistedCounters() {
	data, err := os.ReadFile("/tmp/keystroke_tracker_state.json")
	if err != nil {
		log.Println("📂 No previous state found, starting fresh")
		return
	}

	var state types.CounterState
	if err := json.Unmarshal(data, &state); err != nil {
		log.Printf("⚠️  Failed to load previous state: %v", err)
		return
	}

	// Restore keystroke counters
	for app, keyTypes := range state.Keystrokes {
		for keyType, value := range keyTypes {
			KeystrokesTotal.WithLabelValues(keyType, app).Add(value)
		}
	}

	// Restore app switch counters
	for fromApp, toApps := range state.AppSwitches {
		for toApp, value := range toApps {
			AppSwitchEvents.WithLabelValues(fromApp, toApp).Add(value)
		}
	}

	log.Printf("📂 Restored previous state: %.0f total switches", state.TotalSwitches)
}