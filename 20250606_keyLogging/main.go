package main

import (
	"log"
	"net/http"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promhttp"
	"github.com/robotn/gohook"
)

var (
	keystrokesTotal = prometheus.NewCounter(prometheus.CounterOpts{
		Name: "keystrokes_total",
		Help: "Total number of keystrokes recorded",
	})
)

func init() {
	prometheus.MustRegister(keystrokesTotal)
}

func main() {
	log.Println("Starting keystroke tracker...")

	// Start keyboard event listener in a goroutine
	go startKeystrokeListener()

	// Start HTTP server for Prometheus metrics
	http.Handle("/metrics", promhttp.Handler())
	log.Println("Metrics server starting on :8080/metrics")
	log.Fatal(http.ListenAndServe(":8080", nil))
}

func startKeystrokeListener() {
	log.Println("Starting keyboard event listener...")
	
	// Register keyboard event hook
	hook.Register(hook.KeyDown, []string{}, func(e hook.Event) {
		// Increment counter for any key press
		keystrokesTotal.Inc()
		log.Printf("Keystroke detected! Total: %v", e.Keychar)
	})

	// Start the event loop
	s := hook.Start()
	defer hook.End()
	
	log.Println("Keyboard hook started. If you don't see keystroke events, check macOS Accessibility permissions.")
	<-s
}