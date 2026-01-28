/*
 * HID Debug Test for Heltec WiFi LoRa 32 V3 (ESP32-S3)
 *
 * Purpose: Verify HID enumeration with visual cursor movement
 * Uses Composite CDC+HID for debug output via Serial
 *
 * Expected behavior:
 * - Mouse cursor should move in a small square pattern
 * - Serial output at 115200 baud (if CDC enumerates)
 *
 * Arduino IDE Settings:
 * - Board: Heltec WiFi LoRa 32 (V3)
 * - USB Mode: USB-OTG (TinyUSB)  <-- CRITICAL
 * - USB CDC On Boot: Enabled (for Serial debug)
 */

#include "USB.h"
#include "USBHIDMouse.h"

USBHIDMouse Mouse;

int step = 0;
unsigned long lastMove = 0;
const int MOVE_INTERVAL = 500;  // ms between moves
const int MOVE_AMOUNT = 20;     // pixels to move

void setup() {
  // Initialize USB with both CDC (Serial) and HID
  USB.begin();
  Mouse.begin();

  // Wait for USB to enumerate
  delay(3000);

  // Debug output (will show on native USB CDC if enabled)
  Serial.begin(115200);
  Serial.println();
  Serial.println("=== HID Debug Test Started ===");
  Serial.println("If you see this, CDC is working.");
  Serial.println("Cursor should now move in a square pattern.");
  Serial.println();
}

void loop() {
  unsigned long now = millis();

  if (now - lastMove >= MOVE_INTERVAL) {
    lastMove = now;

    // Move cursor in a square pattern
    switch (step % 4) {
      case 0:
        Mouse.move(MOVE_AMOUNT, 0);  // Right
        Serial.println("Moving cursor: RIGHT");
        break;
      case 1:
        Mouse.move(0, MOVE_AMOUNT);  // Down
        Serial.println("Moving cursor: DOWN");
        break;
      case 2:
        Mouse.move(-MOVE_AMOUNT, 0); // Left
        Serial.println("Moving cursor: LEFT");
        break;
      case 3:
        Mouse.move(0, -MOVE_AMOUNT); // Up
        Serial.println("Moving cursor: UP");
        break;
    }
    step++;
  }
}
