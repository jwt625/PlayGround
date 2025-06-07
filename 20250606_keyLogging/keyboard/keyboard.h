#ifndef KEYBOARD_H
#define KEYBOARD_H

#include <CoreGraphics/CoreGraphics.h>

// Structure to track keystroke types
typedef struct {
    int letters;
    int numbers;
    int special;
} KeystrokeStats;

// Keyboard tracking functions
int startEventTap(void);
int getAndResetLetters(void);
int getAndResetNumbers(void);
int getAndResetSpecial(void);

#endif // KEYBOARD_H