# DevLog-000-Heltec-V3-Analog-Joystick-HID

## General Context
Goal: Use an old **3-DOF analog joystick** (X, Y, Yaw) as a **USB HID input device** on a computer, primarily for:
- X/Y → horizontal/vertical scrolling
- Yaw → zoom (Ctrl + scroll)

Target platform:
- **Heltec WiFi LoRa 32 (V3)**
- MCU: **ESP32-S3**
- Development environment: **Arduino IDE 2.3.6**
- Host OS: **macOS**

Key design choice:
- Use the ESP32-S3 **native USB (TinyUSB)** to emulate a **USB HID mouse/keyboard**.
- No external ADC required.

---

## Hardware & Electrical Decisions

### Joystick Characteristics
- Analog joystick with potentiometer-based axes.
- Original reference: ~0–5 V.
- Neutral positions measured around mid-scale (~2.5 V originally).

### Power & Scaling Decision
- Joystick pots were successfully re-powered from **3.3 V** instead of 5 V.
- This ensured all wiper outputs stayed within ADC-safe range.

Measured voltages after re-powering:
- **Blue (X axis)**:  
  - Left: 0.0 V  
  - Neutral: ~1.25 V  
  - Right max: ~2.90 V
- **Yellow (Y axis)**:  
  - Backward: 0.0 V  
  - Neutral: ~1.38 V  
  - Forward max: ~3.10 V
- **White (Yaw)**:  
  - CCW: 0.0 V  
  - Neutral: ~1.51 V  
  - CW max: ~3.15 V

Conclusion:
- All channels remain below 3.3 V.
- **No voltage dividers required**.
- Direct ADC wiring is safe.

### Wiring
- Joystick Vref → Heltec **3V3**
- Joystick GND → Heltec **GND**
- Joystick signals:
  - Blue (X) → **GPIO5 (ADC1_CH4)**
  - Yellow (Y) → **GPIO6 (ADC1_CH5)**
  - White (Yaw) → **GPIO7 (ADC1_CH6)**

Optional recommendation (not yet confirmed installed):
- 100 nF capacitor from each ADC pin to GND for noise suppression.

---

## Software Strategy

### Input Processing
- High-precision analog not required.
- Plan: quantize each axis into **5–7 discrete levels** (rate-based control).
- Neutral offset handled via **software calibration** (center measured at boot).

### HID Behavior
- Device should enumerate as:
  - USB HID Mouse (scroll + movement)
  - USB HID Keyboard (Ctrl modifier for zoom)
- Scrolling and zoom implemented as **rate-based wheel events**, not absolute position.

---

## USB / Arduino / Upload Findings

### Board Identification
Confirmed board:
- **Heltec WiFi LoRa 32 (V3)** (with LoRa + OLED, ESP32-S3)

### Buttons
- **PRG** = BOOT (GPIO0)
- **RST** = Reset / EN

### LED Behavior
- Yellow LED staying on is **normal**.
- It is a power/system indicator, **not an error or reset-stuck signal**.

### USB Interfaces Observed on macOS
Command:
```bash
ls /dev/cu.*
```

Observed:
- `/dev/cu.usbserial-0001`
- No `/dev/cu.usbmodemXXXX` present during HID sketch runtime.

Interpretation:
- Board exposes a **USB-to-UART bridge** (`usbserial`) used for:
  - Flashing
  - Recovery
- Native ESP32-S3 USB CDC (`usbmodem`) disappears when HID-only firmware runs.
- This is expected TinyUSB / HID behavior.

### Upload Workflow
- Reliable uploads achieved using:
  - **Port:** `/dev/cu.usbserial-0001`
- HID sketches can remove the CDC port at runtime.
- Board recovery always possible via PRG (BOOT) + reset sequence.

---

## Test Sketches & Results

### Trivial Upload Test
- Simple sketch uploaded successfully.
- Confirms:
  - Board is not held in reset.
  - Flash and boot are functional.

### HID Scroll Test
Sketch characteristics:
- Uses `USB.begin()`
- Uses `USBHIDMouse`
- Sends scroll events via:
  - `Mouse.move(0, 0, 1)` or `Mouse.scroll(...)`

Results:
- Sketch uploads successfully.
- No visible scrolling observed in macOS.

### Current Unknown
- HID device may not be enumerating as expected on macOS.
- Need confirmation via:
  - `system_profiler SPUSBDataType`
  - Cursor-movement HID test (more visually obvious than scroll)

No confirmed HID cursor movement observed yet.

---

## What Has Been Ruled Out
- Hardware damage (uploads succeed).
- ADC overvoltage (signals < 3.3 V).
- Board stuck in reset (RST LED behavior normal).
- Wrong board selection in Arduino IDE.

---

## Open Questions / Next Debug Steps
1. Confirm HID enumeration via `system_profiler SPUSBDataType`.
2. Test obvious HID action (cursor movement) instead of scroll.
3. Verify Arduino ESP32 core TinyUSB HID API compatibility.
4. Decide between:
   - HID-only
   - Composite CDC + HID (for easier debugging).
5. Once HID confirmed, integrate joystick ADC + 7-level quantizer.

---

## Status
- Hardware: **Ready**
- ADC wiring: **Verified**
- Flashing: **Reliable**
- HID functionality: **Not yet visually confirmed**


```bash
 Confirmed: Heltec V3 cannot do native USB through its USB-C port.                                            
                                                                                                               
  Here's the full picture:                                                                                     
  ┌──────────────────────┬──────────────────────────┐                                                          
  │         Item         │          Status          │                                                          
  ├──────────────────────┼──────────────────────────┤                                                          
  │ GPIO19 (USB D-)      │ Exposed on header as A18 │                                                          
  ├──────────────────────┼──────────────────────────┤                                                          
  │ GPIO20 (USB D+)      │ Exposed on header as A19 │                                                          
  ├──────────────────────┼──────────────────────────┤                                                          
  │ USB-C port           │ Connected to CP2102 only │                                                          
  ├──────────────────────┼──────────────────────────┤                                                          
  │ Native USB via USB-C │ No                       │                                                          
  └──────────────────────┴──────────────────────────┘                                                          
  The Verdict:                                                                                                 
  - The ESP32-S3 chip supports native USB                                                                      
  - GPIO19/20 are on the header pins                                                                           
  - But the USB-C port only connects to the CP2102 serial chip                                                 
  - Native USB HID won't work without hardware modification                                                    
                                                                                                               
  If you really want to use the Heltec V3 for USB HID:                                                         
  You'd need to solder a USB cable directly to GPIO19/20 on the header (plus GND), bypassing the USB-C port    
  entirely. That's messy.                                                                                      
                                                                                                               
  Recommendation: Use the XIAO ESP32S3 Sense - it has native USB connected properly.                           
                                                                                                               
  Sources:                                                                                                     
  - https://wiki.heltec.org/docs/devices/open-source-hardware/esp32-series/lora-32/wifi-lora-32-v3/Pin-diagram-
  guidance                                                                                                     
  - https://www.espboards.dev/esp32/heltec-wifi-lora-32-v3/                                                    
  - http://community.heltec.cn/t/usb-native-port-on-heltec-wifi-kit-32-v3/19278     
```


## Switch to XIAO ESP32S3 Sense

XIAO ESP32S3 specs for your joystick project:                                                                
  ┌────────────────┬────────────────────────────┬────────┐                                                     
  │  Requirement   │        XIAO ESP32S3        │ Status │                                                     
  ├────────────────┼────────────────────────────┼────────┤                                                     
  │ 3 ADC channels │ 9 ADC pins (A0-A5, A8-A10) │ Yes    │                                                     
  ├────────────────┼────────────────────────────┼────────┤                                                     
  │ 3.3V supply    │ 700mA available on 3V3 pin │ Yes    │                                                     
  ├────────────────┼────────────────────────────┼────────┤                                                     
  │ Native USB HID │ Yes, direct USB-C          │ Yes    │                                                     
  └────────────────┴────────────────────────────┴────────┘                                                     
  Suggested wiring:                                                                                            
  ┌─────────────┬──────────┐                                                                                   
  │  Joystick   │ XIAO Pin │                                                                                   
  ├─────────────┼──────────┤                                                                                   
  │ X (Blue)    │ A0       │                                                                                   
  ├─────────────┼──────────┤                                                                                   
  │ Y (Yellow)  │ A1       │                                                                                   
  ├─────────────┼──────────┤                                                                                   
  │ Yaw (White) │ A2       │                                                                                   
  ├─────────────┼──────────┤                                                                                   
  │ Vref        │ 3V3      │                                                                                   
  ├─────────────┼──────────┤                                                                                   
  │ GND         │ GND      │                                                                                   
  └─────────────┴──────────┘                                             

  