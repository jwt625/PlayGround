/*
 * Simple LED Blink Test - Heltec WiFi LoRa 32 V3
 *
 * The onboard LED is on GPIO35 for this board.
 * If it blinks, uploads work fine.
 */

#define LED_PIN 35

void setup() {
  pinMode(LED_PIN, OUTPUT);
  Serial.begin(115200);
  Serial.println("Blink test started!");
}

void loop() {
  digitalWrite(LED_PIN, HIGH);
  Serial.println("LED ON");
  delay(500);

  digitalWrite(LED_PIN, LOW);
  Serial.println("LED OFF");
  delay(500);
}
