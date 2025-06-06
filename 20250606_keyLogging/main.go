package main

/*
#cgo CFLAGS: -x objective-c
#cgo LDFLAGS: -framework CoreGraphics -framework CoreFoundation -framework ApplicationServices -framework AppKit
#include "keyboard.h"
#include "keyboard.c"
*/
import "C"
import (
	"log"
	"net/http"
	"strings"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promhttp"
)

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

	// Typing speed metrics
	typingSpeedWPM = prometheus.NewGaugeVec(
		prometheus.GaugeOpts{
			Name: "typing_speed_wpm",
			Help: "Current typing speed in words per minute",
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
	typingBuffer   []time.Time // Buffer for WPM calculation
)

func init() {
	prometheus.MustRegister(keystrokesTotal)
	prometheus.MustRegister(appSessionDuration)
	prometheus.MustRegister(appTotalTime)
	prometheus.MustRegister(currentAppGauge)
	prometheus.MustRegister(typingSpeedWPM)
	prometheus.MustRegister(appSwitchTotal)
}

func main() {
	log.Println("Starting keystroke tracker with app awareness and time tracking...")

	// Initialize typing buffer
	typingBuffer = make([]time.Time, 0, 1000)

	// Start app monitoring
	go startAppMonitoring()

	// Start keystroke monitoring in a goroutine
	go startKeystrokeMonitoring()

	// Start metrics collection in a goroutine
	go collectMetrics()

	// Start app time tracking
	go trackAppTime()

	// Start WPM calculation
	go calculateTypingSpeed()

	// Start HTTP server for Prometheus metrics
	http.Handle("/metrics", promhttp.Handler())
	log.Println("Metrics server starting on :8080/metrics")
	log.Fatal(http.ListenAndServe(":8080", nil))
}

func startAppMonitoring() {
	log.Println("Starting application monitoring...")
	C.startAppMonitoring()
}

func startKeystrokeMonitoring() {
	log.Println("Starting native macOS keyboard event monitoring with app awareness...")
	log.Println("This requires Accessibility permissions for your terminal.")

	result := C.startEventTap()
	if result == 0 {
		log.Fatal("Failed to create keyboard event tap. Check Accessibility permissions!")
	}
}

func sanitizeAppName(appName string) string {
	// Clean app name for Prometheus labels (remove spaces, special chars)
	appName = strings.ReplaceAll(appName, " ", "_")
	appName = strings.ReplaceAll(appName, "-", "_")
	appName = strings.ReplaceAll(appName, ".", "_")
	appName = strings.ToLower(appName)
	if appName == "" {
		appName = "unknown"
	}
	return appName
}

func collectMetrics() {
	ticker := time.NewTicker(1 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-ticker.C:
			// Get current app info
			currentApp := C.getCurrentApp()
			appName := C.GoString(&currentApp.app_name[0])
			appName = sanitizeAppName(appName)

			// Get keystroke counts by category
			letters := int(C.getAndResetLetters())
			numbers := int(C.getAndResetNumbers())
			special := int(C.getAndResetSpecial())

			// Update Prometheus counters with app labels
			if letters > 0 {
				keystrokesTotal.WithLabelValues("letter", appName).Add(float64(letters))
				// Add to typing buffer for WPM calculation
				now := time.Now()
				for i := 0; i < letters; i++ {
					typingBuffer = append(typingBuffer, now)
				}
			}
			if numbers > 0 {
				keystrokesTotal.WithLabelValues("number", appName).Add(float64(numbers))
				// Add to typing buffer for WPM calculation
				now := time.Now()
				for i := 0; i < numbers; i++ {
					typingBuffer = append(typingBuffer, now)
				}
			}
			if special > 0 {
				keystrokesTotal.WithLabelValues("special", appName).Add(float64(special))
			}

			total := letters + numbers + special
			if total > 0 {
				log.Printf("App: %s | Total: %d (L:%d N:%d S:%d)", appName, total, letters, numbers, special)
			}
		}
	}
}

func trackAppTime() {
	ticker := time.NewTicker(2 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-ticker.C:
			// Get app switch events
			var events [100]C.AppInfo
			var count C.int

			C.getAppSwitchEvents(&events[0], &count, 100)

			// Process app switch events
			for i := 0; i < int(count); i++ {
				event := events[i]
				appName := C.GoString(&event.app_name[0])
				appName = sanitizeAppName(appName)

				// End current session if there was one
				if currentSession != nil {
					duration := time.Since(currentSession.StartTime).Seconds()
					if duration > 0.5 { // Ignore very short sessions
						// Record session duration
						appSessionDuration.WithLabelValues(currentSession.AppName).Observe(duration)
						appTotalTime.WithLabelValues(currentSession.AppName).Add(duration)
						log.Printf("App session ended: %s (%.1fs)", currentSession.AppName, duration)
					}
				}

				// Start new session
				if appName != "" && appName != "unknown" {
					currentSession = &AppSession{
						AppName:   appName,
						StartTime: time.Unix(int64(event.timestamp), 0),
					}
					appSwitchTotal.Inc()
					log.Printf("App switch: %s", appName)
				}
			}

			// Update current session gauge
			if currentSession != nil {
				duration := time.Since(currentSession.StartTime).Seconds()
				currentAppGauge.WithLabelValues(currentSession.AppName).Set(duration)
			}
		}
	}
}

func calculateTypingSpeed() {
	ticker := time.NewTicker(10 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-ticker.C:
			now := time.Now()
			oneMinuteAgo := now.Add(-1 * time.Minute)

			// Filter typing buffer to last minute
			var recentKeystrokes []time.Time
			for _, keystrokeTime := range typingBuffer {
				if keystrokeTime.After(oneMinuteAgo) {
					recentKeystrokes = append(recentKeystrokes, keystrokeTime)
				}
			}
			typingBuffer = recentKeystrokes

			// Calculate WPM (assume 5 characters per word)
			if len(recentKeystrokes) > 0 {
				wpm := float64(len(recentKeystrokes)) / 5.0 // 5 chars per word

				// Get current app for WPM tracking
				currentApp := C.getCurrentApp()
				appName := C.GoString(&currentApp.app_name[0])
				appName = sanitizeAppName(appName)

				if appName != "" && appName != "unknown" {
					typingSpeedWPM.WithLabelValues(appName).Set(wpm)
					if wpm > 0 {
						log.Printf("Typing speed: %.1f WPM in %s", wpm, appName)
					}
				}
			}
		}
	}
}