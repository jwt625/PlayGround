package main

import (
	"encoding/json"
	"log"
	"net/http"

	"github.com/prometheus/client_golang/prometheus/promhttp"

	"keystroke-tracker/app"
	"keystroke-tracker/chrome"
	"keystroke-tracker/keyboard"
	"keystroke-tracker/metrics"
	"keystroke-tracker/trackpad"
)

// ChromeUpdateRequest represents data from Chrome extension
type ChromeUpdateRequest struct {
	Type       string  `json:"type"`
	Domain     string  `json:"domain"`
	URL        string  `json:"url"`
	Title      string  `json:"title"`
	FromDomain string  `json:"from_domain,omitempty"`
	ToDomain   string  `json:"to_domain,omitempty"`
	Timestamp  float64 `json:"timestamp"`
}

// handleChromeUpdate processes Chrome extension updates via HTTP
func handleChromeUpdate(w http.ResponseWriter, r *http.Request) {
	// Enable CORS for Chrome extension
	w.Header().Set("Access-Control-Allow-Origin", "*")
	w.Header().Set("Access-Control-Allow-Methods", "POST, OPTIONS")
	w.Header().Set("Access-Control-Allow-Headers", "Content-Type")
	
	if r.Method == "OPTIONS" {
		w.WriteHeader(http.StatusOK)
		return
	}
	
	if r.Method != "POST" {
		http.Error(w, "Only POST method allowed", http.StatusMethodNotAllowed)
		return
	}
	
	var update ChromeUpdateRequest
	if err := json.NewDecoder(r.Body).Decode(&update); err != nil {
		log.Printf("❌ Chrome update decode error: %v", err)
		http.Error(w, "Invalid JSON", http.StatusBadRequest)
		return
	}
	
	// Update Chrome domain directly
	if update.Domain != "" && update.Domain != "unknown" {
		chrome.UpdateCurrentDomain(update.Domain)
		log.Printf("🌐 Chrome domain via HTTP: %s", update.Domain)
	}
	
	// Respond with success
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]string{
		"status": "ok",
		"domain": update.Domain,
	})
}

func main() {
	log.Println("Starting hybrid keystroke tracker...")
	log.Println("📊 Keystrokes: C/CGEventTap")
	log.Println("📱 App Detection: Swift Helper")
	log.Println("")
	log.Println("🚀 Start the Swift helper in another terminal:")
	log.Println("   swift swift/tracker.swift")

	// Register all Prometheus metrics
	metrics.RegisterMetrics()

	// Load previous state
	metrics.LoadPersistedCounters()

	// Start keystroke monitoring in a goroutine
	go keyboard.StartKeystrokeMonitoring()

	// Start app info reader
	go app.ReadAppInfo()

	// Start app switch monitor
	go app.MonitorAppSwitches()

	// Start trackpad click monitor
	go trackpad.MonitorClickEvents()

	// Chrome domain updates now come via HTTP endpoint

	// Start metrics collection
	go keyboard.CollectMetrics()

	// Start HTTP server for Prometheus metrics
	http.Handle("/metrics", promhttp.Handler())
	
	// Add Chrome extension endpoint
	http.HandleFunc("/chrome-update", handleChromeUpdate)
	
	log.Println("Metrics server starting on :8080/metrics")
	log.Println("Chrome extension endpoint: :8080/chrome-update")
	log.Fatal(http.ListenAndServe(":8080", nil))
}