void setup() {
  Serial.begin(115200);
  delay(1000);
  Serial.println("ESP32 Version: " + String(ESP.getChipModel()));
  Serial.println("SDK Version: " + String(ESP.getSdkVersion()));
}

void loop() {
  // Nothing to do here
}
