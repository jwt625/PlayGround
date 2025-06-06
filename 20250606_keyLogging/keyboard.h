#ifndef KEYBOARD_H
#define KEYBOARD_H

#include <CoreGraphics/CoreGraphics.h>
#include <CoreFoundation/CoreFoundation.h>

// Structure to track different key types
typedef struct {
    int letters;
    int numbers;
    int special;
} KeystrokeStats;

// Structure to track app information
typedef struct {
    char app_name[256];
    double timestamp;
} AppInfo;

// Keyboard tracking functions
int startEventTap(void);
int getAndResetLetters(void);
int getAndResetNumbers(void);
int getAndResetSpecial(void);

// App tracking functions
int startAppMonitoring(void);
AppInfo getCurrentApp(void);
void getAppSwitchEvents(AppInfo* events, int* count, int max_events);

#endif // KEYBOARD_H