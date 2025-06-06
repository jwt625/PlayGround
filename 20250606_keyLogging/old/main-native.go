package main

/*
#cgo CFLAGS: -x objective-c
#cgo LDFLAGS: -framework CoreGraphics -framework CoreFoundation
#include <CoreGraphics/CoreGraphics.h>
#include <CoreFoundation/CoreFoundation.h>

// Callback function for keyboard events
CGEventRef keyboardCallback(CGEventTapProxy proxy, CGEventType type, CGEventRef event, void *refcon);

// Global variable to track keystrokes from C
static int keystroke_count = 0;

CGEventRef keyboardCallback(CGEventTapProxy proxy, CGEventType type, CGEventRef event, void *refcon) {
    if (type == kCGEventKeyDown) {
        keystroke_count++;
    }
    // Return the event unmodified (passthrough)
    return event;
}

// Function to get current keystroke count and reset it
int getAndResetKeystrokeCount() {
    int count = keystroke_count;
    keystroke_count = 0;
    return count;
}

// Function to start the event tap
int startEventTap() {
    CGEventMask eventMask = CGEventMaskBit(kCGEventKeyDown);
    CFMachPortRef eventTap = CGEventTapCreate(
        kCGSessionEventTap,
        kCGHeadInsertEventTap,
        kCGEventTapOptionListenOnly,  // Listen only, don't modify
        eventMask,
        keyboardCallback,
        NULL
    );
    
    if (!eventTap) {
        return 0; // Failed to create event tap
    }
    
    CFRunLoopSourceRef runLoopSource = CFMachPortCreateRunLoopSource(kCFAllocatorDefault, eventTap, 0);
    CFRunLoopAddSource(CFRunLoopGetCurrent(), runLoopSource, kCFRunLoopCommonModes);
    CGEventTapEnable(eventTap, true);
    
    // Run the event loop
    CFRunLoopRun();
    
    return 1;
}
*/
import "C"
import (
	"log"
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
	log.Println("Starting keystroke tracker (native macOS)...")

	// Start keystroke monitoring in a goroutine
	go startKeystrokeMonitoring()

	// Start metrics collection in a goroutine
	go collectMetrics()

	// Start HTTP server for Prometheus metrics
	http.Handle("/metrics", promhttp.Handler())
	log.Println("Metrics server starting on :8080/metrics")
	log.Fatal(http.ListenAndServe(":8080", nil))
}

func startKeystrokeMonitoring() {
	log.Println("Starting native macOS keyboard event monitoring...")
	log.Println("This requires Accessibility permissions for your terminal.")
	
	// This will block and run the CFRunLoop
	result := C.startEventTap()
	if result == 0 {
		log.Fatal("Failed to create keyboard event tap. Check Accessibility permissions!")
	}
}

func collectMetrics() {
	ticker := time.NewTicker(1 * time.Second)
	defer ticker.Stop()
	
	for {
		select {
		case <-ticker.C:
			// Get keystroke count from C and add to Prometheus counter
			count := C.getAndResetKeystrokeCount()
			if count > 0 {
				for i := 0; i < int(count); i++ {
					keystrokesTotal.Inc()
				}
				log.Printf("Captured %d keystrokes", count)
			}
		}
	}
}