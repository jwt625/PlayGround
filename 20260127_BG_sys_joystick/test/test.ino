#include "USB.h"
#include "USBHIDMouse.h"

USBHIDMouse Mouse;

void setup() {
  USB.begin();
  Mouse.begin();
  delay(2000);
}

void loop() {
  Mouse.move(0, 0, 1);  // scroll down slowly
  delay(200);
}
