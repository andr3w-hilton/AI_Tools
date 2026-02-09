# M5Stack CardComputer v1.1 Projects

## Hardware
- **Main Unit:** M5Stack CardComputer v1.1 with ESP32-S3 STAMP S3A
- **Display:** 1.14" LCD (240x135)
- **Keyboard:** Built-in QWERTY keyboard
- **Connectivity:** WiFi, Bluetooth

## Ordered Modules (Pending Arrival)

### I/O Hub 1-to-6 Expansion Unit
- Expands single port to 6 ports
- Enables connection of multiple modules simultaneously

### RF 433MHz Transmitter/Receiver Modules
- Wireless communication at 433MHz frequency
- Common for home automation, remote controls
- **Potential Projects:**
  - Home automation controller
  - Wireless sensor network
  - RF remote control system
  - RF signal scanner/analyzer

### GPS/BDS v1.1 Module
- GPS (Global Positioning System)
- BDS (BeiDou Navigation Satellite System)
- **Potential Projects:**
  - Location tracker
  - Geocaching tool
  - Navigation device
  - GPS logger
  - Speed/altitude monitor

### RFID Unit 2
- Radio-frequency identification reader/writer
- **Potential Projects:**
  - Access control system
  - Asset tracking
  - RFID tag reader/writer
  - NFC payment/authentication tool
  - Inventory management

## CircuitPython Setup

**Recommended Version:** CircuitPython 9.2.x (stable)
**Download:** https://circuitpython.org/board/m5stack_cardputer/

**Note:** Avoid CircuitPython 10.x beta versions due to bootloop issues with this hardware.

## Project Ideas

### Combined Module Projects
1. **Smart Home Controller** - RF 433MHz + RFID for home automation with access control
2. **GPS Asset Tracker** - GPS + RFID for tracking tagged items with location
3. **Portable Security System** - RFID + RF to monitor and control access points
4. **Field Data Collection Device** - GPS + RFID for inventory/survey work with location tagging

## Development Notes

This directory will contain CircuitPython applications and libraries for the M5Stack CardComputer.

### Typical Project Structure
```
project_name/
├── code.py           # Main CircuitPython script
├── lib/              # Required CircuitPython libraries
├── config.py         # Configuration settings
└── README.md         # Project documentation
```
