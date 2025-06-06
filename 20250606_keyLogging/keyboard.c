#include "keyboard.h"
#include <CoreGraphics/CoreGraphics.h>
#include <CoreFoundation/CoreFoundation.h>
#include <ApplicationServices/ApplicationServices.h>
#include <AppKit/AppKit.h>
#include <string.h>
#include <time.h>

// Global variables for keystroke tracking
static KeystrokeStats keystroke_stats = {0, 0, 0};
static char current_app_name[256] = "";
static AppInfo app_switch_events[100];
static int app_switch_count = 0;
static double last_app_switch_time = 0;

// Get current timestamp
double getCurrentTimestamp() {
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    return ts.tv_sec + ts.tv_nsec / 1000000000.0;
}

// Get the name of the currently active application
void updateCurrentApp() {
    NSWorkspace *workspace = [NSWorkspace sharedWorkspace];
    NSRunningApplication *app = [workspace frontmostApplication];
    
    if (app && app.localizedName) {
        const char* appName = [app.localizedName UTF8String];
        
        // Check if app changed
        if (strcmp(current_app_name, appName) != 0) {
            // Record app switch event
            if (app_switch_count < 100) {
                strncpy(app_switch_events[app_switch_count].app_name, current_app_name, 255);
                app_switch_events[app_switch_count].app_name[255] = '\0';
                app_switch_events[app_switch_count].timestamp = getCurrentTimestamp();
                app_switch_count++;
            }
            
            // Update current app
            strncpy(current_app_name, appName, 255);
            current_app_name[255] = '\0';
            last_app_switch_time = getCurrentTimestamp();
        }
    }
}

// Keyboard event callback
CGEventRef keyboardCallback(CGEventTapProxy proxy, CGEventType type, CGEventRef event, void *refcon) {
    (void)proxy;
    (void)refcon;
    
    if (type == kCGEventKeyDown) {
        // Update current app on each keystroke (efficient way to track app switches)
        updateCurrentApp();
        
        CGKeyCode keycode = (CGKeyCode)CGEventGetIntegerValueField(event, kCGKeyboardEventKeycode);
        
        // Categorize keys based on macOS virtual key codes
        if (
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

// Start keyboard event monitoring
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

// Get and reset keystroke counts
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

// Get current application info
AppInfo getCurrentApp() {
    AppInfo info;
    strncpy(info.app_name, current_app_name, 255);
    info.app_name[255] = '\0';
    info.timestamp = last_app_switch_time;
    return info;
}

// Get app switch events
void getAppSwitchEvents(AppInfo* events, int* count, int max_events) {
    int copy_count = app_switch_count < max_events ? app_switch_count : max_events;
    
    for (int i = 0; i < copy_count; i++) {
        events[i] = app_switch_events[i];
    }
    
    *count = copy_count;
    
    // Reset the events array
    app_switch_count = 0;
}

// Initialize app monitoring
int startAppMonitoring() {
    // Initialize current app
    updateCurrentApp();
    return 1;
}