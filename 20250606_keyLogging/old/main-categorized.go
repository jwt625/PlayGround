package main

/*
#cgo CFLAGS: -x objective-c
#cgo LDFLAGS: -framework CoreGraphics -framework CoreFoundation
#include <CoreGraphics/CoreGraphics.h>
#include <CoreFoundation/CoreFoundation.h>

// Structure to track different key types
typedef struct {
    int letters;
    int numbers;
    int special;
} KeystrokeStats;

static KeystrokeStats keystroke_stats = {0, 0, 0};

// Callback function for keyboard events
CGEventRef keyboardCallback(CGEventTapProxy proxy, CGEventType type, CGEventRef event, void *refcon);

CGEventRef keyboardCallback(CGEventTapProxy proxy, CGEventType type, CGEventRef event, void *refcon) {
    if (type == kCGEventKeyDown) {
        CGKeyCode keycode = (CGKeyCode)CGEventGetIntegerValueField(event, kCGKeyboardEventKeycode);
        
        // Categorize the key based on keycode
        // macOS keycodes: https://stackoverflow.com/questions/3202629/where-can-i-find-a-list-of-mac-virtual-key-codes
        if ((keycode >= 0 && keycode <= 9) ||     // Q,W,E,R,T,Y,U,I,O,P
            (keycode >= 11 && keycode <= 14) ||   // A,S,D,F
            (keycode >= 16 && keycode <= 19) ||   // G,H,J,K
            (keycode >= 31 && keycode <= 35) ||   // Z,X,C,V,B
            (keycode >= 37 && keycode <= 41) ||   // L,;,',Enter,N
            keycode == 45 || keycode == 46 ||     // M,comma
            keycode == 10 || keycode == 15 ||     // grave,R
            keycode == 20 || keycode == 21 ||     // 4,6
            keycode == 22 || keycode == 26 ||     // 7,J
            keycode == 28 || keycode == 25 ||     // 5,9
            keycode == 29 || keycode == 23 ||     // 0,5
            keycode == 27 || keycode == 24) {     // minus,equals
            keystroke_stats.letters++;
        } else if ((keycode >= 18 && keycode <= 23) ||  // 1,2,3,4,5,6
                   (keycode >= 25 && keycode <= 29)) {   // 7,8,9,0
            keystroke_stats.numbers++;
        } else {
            keystroke_stats.special++;
        }
    }
    return event;
}

// Functions to get and reset specific keystroke counts
int getAndResetLetters() {
    int count = keystroke_stats.letters;
    keystroke_stats.letters = 0;
    return count;
}

int getAndResetNumbers() {
    int count = keystroke_stats.numbers;
    keystroke_stats.numbers = 0;
    return count;
}

int getAndResetSpecial() {
    int count = keystroke_stats.special;
    keystroke_stats.special = 0;
    return count;
}

// Function to start the event tap
int startEventTap() {
    CGEventMask eventMask = CGEventMaskBit(kCGEventKeyDown);
    CFMachPortRef eventTap = CGEventTapCreate(
        kCGSessionEventTap,
        kCGHeadInsertEventTap,
        kCGEventTapOptionListenOnly,
        eventMask,
        keyboardCallback,
        NULL
    );
    
    if (!eventTap) {
        return 0;
    }
    
    CFRunLoopSourceRef runLoopSource = CFMachPortCreateRunLoopSource(kCFAllocatorDefault, eventTap, 0);
    CFRunLoopAddSource(CFRunLoopGetCurrent(), runLoopSource, kCFRunLoopCommonModes);
    CGEventTapEnable(eventTap, true);
    
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
	keystrokesTotal = prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Name: "keystrokes_total",
			Help: "Total number of keystrokes recorded by key type",
		},
		[]string{"key_type"},
	)
)

func init() {
	prometheus.MustRegister(keystrokesTotal)
}

func main() {
	log.Println("Starting keystroke tracker with categorization...")

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
	log.Println("Starting native macOS keyboard event monitoring with categorization...")
	log.Println("This requires Accessibility permissions for your terminal.")
	
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
			// Get keystroke counts by category
			letters := int(C.getAndResetLetters())
			numbers := int(C.getAndResetNumbers())
			special := int(C.getAndResetSpecial())
			
			// Update Prometheus counters with labels
			if letters > 0 {
				keystrokesTotal.WithLabelValues("letter").Add(float64(letters))
				log.Printf("Letters: %d", letters)
			}
			if numbers > 0 {
				keystrokesTotal.WithLabelValues("number").Add(float64(numbers))
				log.Printf("Numbers: %d", numbers)
			}
			if special > 0 {
				keystrokesTotal.WithLabelValues("special").Add(float64(special))
				log.Printf("Special keys: %d", special)
			}
			
			total := letters + numbers + special
			if total > 0 {
				log.Printf("Total this second: %d (L:%d N:%d S:%d)", total, letters, numbers, special)
			}
		}
	}
}