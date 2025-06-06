package main

import (
	"log"
	"math/rand"
	"net/http"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promhttp"
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
	log.Println("Starting keystroke tracker (simulator mode)...")

	// Start keystroke simulator in a goroutine
	go startKeystrokeSimulator()

	// Start HTTP server for Prometheus metrics
	http.Handle("/metrics", promhttp.Handler())
	log.Println("Metrics server starting on :8080/metrics")
	log.Fatal(http.ListenAndServe(":8080", nil))
}

func startKeystrokeSimulator() {
	log.Println("Starting keystroke simulator...")
	log.Println("Simulating realistic typing patterns...")
	
	ticker := time.NewTicker(time.Second)
	defer ticker.Stop()
	
	for {
		select {
		case <-ticker.C:
			// Simulate realistic typing: 0-8 keystrokes per second
			// Average human types ~5 characters per second during active typing
			keystrokes := rand.Intn(9) // 0-8 keystrokes
			
			if keystrokes > 0 {
				for i := 0; i < keystrokes; i++ {
					keystrokesTotal.Inc()
				}
				log.Printf("Simulated %d keystrokes", keystrokes)
			}
			
			// Add some randomness - sometimes no typing for a few seconds
			if rand.Intn(10) < 2 { // 20% chance of pause
				time.Sleep(time.Duration(rand.Intn(3)+1) * time.Second)
			}
		}
	}
}