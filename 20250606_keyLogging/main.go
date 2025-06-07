package main

import (
	"log"
	"net/http"

	"github.com/prometheus/client_golang/prometheus/promhttp"

	"keystroke-tracker/app"
	"keystroke-tracker/keyboard"
	"keystroke-tracker/metrics"
)

func main() {
	log.Println("Starting hybrid keystroke tracker...")
	log.Println("📊 Keystrokes: C/CGEventTap")
	log.Println("📱 App Detection: Swift Helper")
	log.Println("")
	log.Println("🚀 Start the Swift helper in another terminal:")
	log.Println("   swift app-detector-helper.swift")

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

	// Start metrics collection
	go keyboard.CollectMetrics()

	// Start HTTP server for Prometheus metrics
	http.Handle("/metrics", promhttp.Handler())
	log.Println("Metrics server starting on :8080/metrics")
	log.Fatal(http.ListenAndServe(":8080", nil))
}