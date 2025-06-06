package main

/*
#cgo CFLAGS: -x objective-c
#cgo LDFLAGS: -framework CoreGraphics -framework CoreFoundation
#include <CoreGraphics/CoreGraphics.h>
#include <CoreFoundation/CoreFoundation.h>

// Structure to track different key types including modifiers
typedef struct {
    int letters;
    int numbers;
    int special;
    int shift;
    int cmd;
    int option;
    int control;
    int function;
} KeystrokeStats;

static KeystrokeStats keystroke_stats = {0, 0, 0, 0, 0, 0, 0, 0};

// Forward declaration
CGEventRef keyboardCallback(CGEventTapProxy proxy, CGEventType type, CGEventRef event, void *refcon);

CGEventRef keyboardCallback(CGEventTapProxy proxy, CGEventType type, CGEventRef event, void *refcon) {
    (void)proxy; // Suppress unused parameter warning
    (void)refcon; // Suppress unused parameter warning
    
    if (type == kCGEventKeyDown) {
        CGKeyCode keycode = (CGKeyCode)CGEventGetIntegerValueField(event, kCGKeyboardEventKeycode);
        
        // Check for modifier keys first
        if (keycode == 56 || keycode == 60) {  // Left/Right Shift
            keystroke_stats.shift++;
        } else if (keycode == 55 || keycode == 54) {  // Left/Right Cmd
            keystroke_stats.cmd++;
        } else if (keycode == 58 || keycode == 61) {  // Left/Right Option
            keystroke_stats.option++;
        } else if (keycode == 59 || keycode == 62) {  // Left/Right Control
            keystroke_stats.control++;
        } else if (keycode >= 122 && keycode <= 131) {  // Function keys F1-F10
            keystroke_stats.function++;
        } else if (
            // Letters A-Z
            keycode == 0 ||   // A
            keycode == 11 ||  // B
            keycode == 8 ||   // C
            keycode == 2 ||   // D
            keycode == 14 ||  // E
            keycode == 3 ||   // F
            keycode == 5 ||   // G
            keycode == 4 ||   // H
            keycode == 34 ||  // I
            keycode == 38 ||  // J
            keycode == 40 ||  // K
            keycode == 37 ||  // L
            keycode == 46 ||  // M
            keycode == 45 ||  // N
            keycode == 31 ||  // O
            keycode == 35 ||  // P
            keycode == 12 ||  // Q
            keycode == 15 ||  // R
            keycode == 1 ||   // S
            keycode == 17 ||  // T
            keycode == 32 ||  // U
            keycode == 9 ||   // V
            keycode == 13 ||  // W
            keycode == 7 ||   // X
            keycode == 16 ||  // Y
            keycode == 6      // Z
        ) {
            keystroke_stats.letters++;
        } else if (
            // Numbers 0-9
            keycode == 29 ||  // 0
            keycode == 18 ||  // 1
            keycode == 19 ||  // 2
            keycode == 20 ||  // 3
            keycode == 21 ||  // 4
            keycode == 23 ||  // 5
            keycode == 22 ||  // 6
            keycode == 26 ||  // 7
            keycode == 28 ||  // 8
            keycode == 25     // 9
        ) {
            keystroke_stats.numbers++;
        } else {
            // Everything else: space, enter, punctuation, arrows, etc.
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

int getAndResetShift() {
    int count = keystroke_stats.shift;
    keystroke_stats.shift = 0;
    return count;
}

int getAndResetCmd() {
    int count = keystroke_stats.cmd;
    keystroke_stats.cmd = 0;
    return count;
}

int getAndResetOption() {
    int count = keystroke_stats.option;
    keystroke_stats.option = 0;
    return count;
}

int getAndResetControl() {
    int count = keystroke_stats.control;
    keystroke_stats.control = 0;
    return count;
}

int getAndResetFunction() {
    int count = keystroke_stats.function;
    keystroke_stats.function = 0;
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
	log.Println("Starting keystroke tracker with modifier key support...")

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
	log.Println("Starting native macOS keyboard event monitoring with modifier tracking...")
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
			shift := int(C.getAndResetShift())
			cmd := int(C.getAndResetCmd())
			option := int(C.getAndResetOption())
			control := int(C.getAndResetControl())
			function := int(C.getAndResetFunction())
			
			// Update Prometheus counters with labels
			if letters > 0 {
				keystrokesTotal.WithLabelValues("letter").Add(float64(letters))
			}
			if numbers > 0 {
				keystrokesTotal.WithLabelValues("number").Add(float64(numbers))
			}
			if special > 0 {
				keystrokesTotal.WithLabelValues("special").Add(float64(special))
			}
			if shift > 0 {
				keystrokesTotal.WithLabelValues("shift").Add(float64(shift))
			}
			if cmd > 0 {
				keystrokesTotal.WithLabelValues("cmd").Add(float64(cmd))
			}
			if option > 0 {
				keystrokesTotal.WithLabelValues("option").Add(float64(option))
			}
			if control > 0 {
				keystrokesTotal.WithLabelValues("control").Add(float64(control))
			}
			if function > 0 {
				keystrokesTotal.WithLabelValues("function").Add(float64(function))
			}
			
			total := letters + numbers + special + shift + cmd + option + control + function
			if total > 0 {
				log.Printf("Total: %d (L:%d N:%d S:%d Shift:%d Cmd:%d Opt:%d Ctrl:%d Fn:%d)", 
					total, letters, numbers, special, shift, cmd, option, control, function)
			}
		}
	}
}