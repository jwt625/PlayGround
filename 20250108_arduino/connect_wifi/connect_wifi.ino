
// Replace with your network credentials

#include "config.h"
#define CAMERA_MODEL_XIAO_ESP32S3
#include "esp_camera.h"
#include <WiFi.h>
#include <WebServer.h>

// Replace these with your network credentials
// const char* ssid     = "YOUR_SSID";
// const char* password = "YOUR_PASSWORD";

// Create a web server on port 80
WebServer server(80);

// Include your camera pins definition (usually provided in camera_pins.h)
#include "camera_pins.h"

// HTTP handler to capture and send the JPEG image
void handleCapture() {
  camera_fb_t *fb = esp_camera_fb_get();
  if (!fb) {
    Serial.println("Camera capture failed");
    server.send(500, "text/plain", "Camera capture failed");
    return;
  }
  
  // Get the client connected to the server
  WiFiClient client = server.client();
  
  // Manually send the HTTP response headers
  client.print("HTTP/1.1 200 OK\r\n");
  client.print("Content-Type: image/jpeg\r\n");
  client.print("Content-Length: ");
  client.print(fb->len);
  client.print("\r\n");
  client.print("Access-Control-Allow-Origin: *\r\n");
  client.print("\r\n");
  
  // Send the JPEG image data directly to the client
  client.write(fb->buf, fb->len);
  
  // Return the frame buffer back to the driver
  esp_camera_fb_return(fb);
}



void setup() {
  Serial.begin(115200);
  delay(1000);

  // --- Camera configuration ---
  camera_config_t config;
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer   = LEDC_TIMER_0;
  
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
  
  // Choose an appropriate frame size based on your module’s PSRAM availability.
  // For modules with PSRAM (like the XIAO ESP32S3 Sense), you can use higher resolutions.
  config.frame_size = FRAMESIZE_SVGA;
  config.jpeg_quality = 12; // 0-63, lower means better quality
  config.fb_count = 1;      // Use one frame buffer

  // Initialize the camera
  esp_err_t err = esp_camera_init(&config);
  if(err != ESP_OK) {
    Serial.printf("Camera init failed with error 0x%x", err);
    return;
  }
  Serial.println("Camera initialized");

  // --- Connect to WiFi ---
  WiFi.mode(WIFI_STA);
  WiFi.begin(ssid, password);
  Serial.print("Connecting to WiFi");
  while(WiFi.status() != WL_CONNECTED) {
    Serial.print(".");
    delay(500);
  }
  Serial.println();
  Serial.print("Connected! IP Address: ");
  Serial.println(WiFi.localIP());

  // --- Set up the web server ---
  server.on("/capture", HTTP_GET, handleCapture);
  server.begin();
  Serial.println("HTTP server started");
}

void loop() {
  server.handleClient();
  delay(1);
}

