#include "keyboard.h"
#include <CoreGraphics/CoreGraphics.h>
#include <CoreFoundation/CoreFoundation.h>
#include <string.h>
#include <time.h>

// Global variables for keystroke tracking only
static KeystrokeStats keystroke_stats = {0, 0, 0};

// Keyboard event callback - only tracks keystrokes
CGEventRef keyboardCallback(CGEventTapProxy proxy, CGEventType type, CGEventRef event, void *refcon) {
    (void)proxy;
    (void)refcon;
    
    if (type == kCGEventKeyDown) {
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