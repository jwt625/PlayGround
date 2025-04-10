#include "config.h"
#define CAMERA_MODEL_XIAO_ESP32S3
#include "esp_camera.h"
#include <WiFi.h>
#include "esp_sleep.h"
// Include your camera pins definition (usually provided in camera_pins.h)
#include "camera_pins.h"


// Constants
const uint64_t SLEEP_DURATION = 60 * 1000000; // 60 seconds in microseconds
const int MAX_CONNECTION_ATTEMPTS = 3;
const int MAX_SEND_ATTEMPTS = 3;
const char* SERVER_URL = "http://192.168.12.135/capture";

// Function declarations
bool initCamera();
bool setupWiFi();
bool captureAndSendImage();
void goToSleep();

void setup() {
  Serial.begin(115200);
  delay(100); // Brief delay for serial stability
  // Wait for 5 seconds at startup to allow time for potential uploads
  for(int i = 5; i > 0; i--) {
    Serial.printf("Upload window: %d seconds remaining...\n", i);
    delay(1000);
  }
  
  
  // Initialize camera with the same config as before
  if (!initCamera()) {
    Serial.println("Camera init failed - going to sleep");
    goToSleep();
    return;
  }

  // Try to connect to WiFi
  if (!setupWiFi()) {
    Serial.println("WiFi connection failed - going to sleep");
    goToSleep();
    return;
  }

  // Capture and send image
  if (!captureAndSendImage()) {
    Serial.println("Image capture/send failed - going to sleep");
  }

  // Go to sleep after everything (success or failure)
  goToSleep();
}

void loop() {
  // Nothing here - everything is done in setup() before sleeping
}

bool initCamera() {
  camera_config_t config;
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer = LEDC_TIMER_0;
  
  // These pin assignments come from the camera_pins.h for XIAO ESP32S3 Sense
  config.pin_d0     = Y2_GPIO_NUM;
  config.pin_d1     = Y3_GPIO_NUM;
  config.pin_d2     = Y4_GPIO_NUM;
  config.pin_d3     = Y5_GPIO_NUM;
  config.pin_d4     = Y6_GPIO_NUM;
  config.pin_d5     = Y7_GPIO_NUM;
  config.pin_d6     = Y8_GPIO_NUM;
  config.pin_d7     = Y9_GPIO_NUM;
  config.pin_xclk   = XCLK_GPIO_NUM;
  config.pin_pclk   = PCLK_GPIO_NUM;
  config.pin_vsync  = VSYNC_GPIO_NUM;
  config.pin_href   = HREF_GPIO_NUM;
  config.pin_sscb_sda = SIOD_GPIO_NUM;
  config.pin_sscb_scl = SIOC_GPIO_NUM;
  config.pin_pwdn   = PWDN_GPIO_NUM;
  config.pin_reset  = RESET_GPIO_NUM;

  config.xclk_freq_hz = 20000000;
  config.pixel_format = PIXFORMAT_JPEG;
  config.frame_size = FRAMESIZE_SVGA;
  config.jpeg_quality = 12;
  config.fb_count = 1;
  
  pinMode(PWDN_GPIO_NUM, OUTPUT);
  digitalWrite(PWDN_GPIO_NUM, 0); // Ensure sensor is active
  delay(10); // Allow power-up time

  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    Serial.printf("Camera init failed with error 0x%x\n", err);
    return false;
  }

  // Configure camera settings
  sensor_t* s = esp_camera_sensor_get();
  if (s) {
    s->set_brightness(s, 1);
    s->set_exposure_ctrl(s, 1);
    s->set_gain_ctrl(s, 1);
    s->set_whitebal(s, 1);
  }

  return true;
}

bool setupWiFi() {
  WiFi.mode(WIFI_STA);
  WiFi.begin(ssid, password);

  int attempts = 0;
  while (WiFi.status() != WL_CONNECTED && attempts < MAX_CONNECTION_ATTEMPTS) {
    delay(1000);
    Serial.print(".");
    attempts++;
  }

  if (WiFi.status() != WL_CONNECTED) {
    Serial.println("\nWiFi connection failed");
    return false;
  }

  Serial.println("\nWiFi connected");
  Serial.print("IP Address: ");
  Serial.println(WiFi.localIP());
  return true;
}

bool captureAndSendImage() {
  camera_fb_t* fb = esp_camera_fb_get();
  if (!fb) {
    Serial.println("Camera capture failed");
    return false;
  }

  // Create a WiFiClient object
  WiFiClient client;
  // Connect to your Python server
  if (!client.connect(SERVER_IP, SERVER_PORT)) {
    Serial.println("Connection to server failed");
    esp_camera_fb_return(fb);
    return false;
  }

  // Send HTTP POST request
  client.println("POST /capture HTTP/1.1");
  client.println("Host: 192.168.12.135");
  client.println("Content-Type: image/jpeg");
  client.print("Content-Length: ");
  client.println(fb->len);
  client.println();
  
  // Send the image data
  client.write(fb->buf, fb->len);
  
  // Wait for server response
  unsigned long timeout = millis();
  while (client.available() == 0) {
    if (millis() - timeout > 5000) {
      Serial.println("Server response timeout");
      client.stop();
      esp_camera_fb_return(fb);
      return false;
    }
  }

  // Read server response
  while (client.available()) {
    String line = client.readStringUntil('\r');
    Serial.println(line);
  }

  // Cleanup
  client.stop();
  esp_camera_fb_return(fb);
  return true;
}

void goToSleep() {
  // Disconnect WiFi
  WiFi.disconnect(true);
  WiFi.mode(WIFI_OFF);
  
  // Power down the camera
  esp_camera_deinit();
  pinMode(PWDN_GPIO_NUM, OUTPUT);
  digitalWrite(PWDN_GPIO_NUM, 1); // Drive high to power down sensor

  
  // Configure wake-up timer
  esp_sleep_enable_timer_wakeup(SLEEP_DURATION);
  
  Serial.println("Going to sleep...");
  Serial.flush();
  
  // Enter deep sleep
  esp_deep_sleep_start();
} 