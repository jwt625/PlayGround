/*
 * HID Mouse Test - Seeed XIAO ESP32S3 Sense
 *
 * This board has NATIVE USB - no CP2102!
 *
 * Arduino IDE Settings:
 * - Board: XIAO_ESP32S3
 * - USB CDC On Boot: Enabled
 * - USB Mode: USB-OTG (TinyUSB)
 *
 * Expected: Cursor moves in a square pattern every 500ms
 */

#include "USB.h"
#include "USBHIDMouse.h"

USBHIDMouse Mouse;

int step = 0;
unsigned long lastMove = 0;

void setup() {
  USB.begin();
  Mouse.begin();
  delay(2000);  // Wait for USB enumeration

  Serial.begin(115200);
  Serial.println("XIAO HID Test - cursor should be moving!");
}

void loop() {
  if (millis() - lastMove >= 500) {
    lastMove = millis();

    switch (step % 4) {
      case 0: Mouse.move(30, 0);  Serial.println("RIGHT"); break;
      case 1: Mouse.move(0, 30);  Serial.println("DOWN");  break;
      case 2: Mouse.move(-30, 0); Serial.println("LEFT");  break;
      case 3: Mouse.move(0, -30); Serial.println("UP");    break;
    }
    step++;
  }
}
