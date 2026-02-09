# I2C Diagnostics Tool

Diagnostic script to troubleshoot the 1-to-6 I/O Hub and connected modules on M5Stack CardComputer.

## Purpose

This tool helps identify why modules aren't working when connected through the I/O hub by:
- Scanning the I2C bus for connected devices
- Testing communication with GPS, RFID, and RF modules
- Providing troubleshooting guidance for common issues

## Hardware Setup

1. **M5Stack CardComputer v1.1** with CircuitPython installed
2. **1-to-6 I/O Hub** connected to Port A of CardComputer
3. **Modules to test:**
   - GPS/BDS v1.1 Module
   - RFID Unit 2
   - RF 433MHz Transmitter/Receiver

## Installation

1. Copy `code.py` to your CardComputer's CIRCUITPY drive
2. The script will run automatically on boot
3. View output via serial console (115200 baud)

## Using Serial Console

Connect to serial monitor to see diagnostic output:

**Windows (PowerShell):**
```powershell
# Find COM port
mode

# Use PuTTY, TeraTerm, or Windows Terminal
# Baud rate: 115200
```

**Thonny IDE:**
- Install Thonny
- View → Shell
- Select CircuitPython interpreter

## What the Diagnostic Tests

### 1. I2C Bus Scan
- Scans addresses 0x08 to 0x77
- Lists all detected I2C devices
- Identifies common module addresses

### 2. GPS Module Test
- Checks for I2C GPS at address 0x42
- Notes: Most GPS modules use UART/serial, not I2C
- If not found on I2C, GPS likely uses TX/RX pins

### 3. RFID Module Test
- Checks common RFID addresses (0x24, 0x48, 0x28, 0x29)
- RFID Unit 2 typically uses PN532 chip via I2C

### 4. RF 433MHz Test
- Notes that RF modules use GPIO, not I2C
- These won't appear on I2C scan (expected behavior)

### 5. Connection Checklist
- Provides troubleshooting steps
- Power supply considerations
- Common issue explanations

## Expected Results

### If Working Correctly:
```
Found 1-3 device(s):
  • Address: 0x24 (36) - RFID Reader (PN532 possible)
  • Address: 0x42 (66) - GPS Module (typical)
```

### If No Devices Found:
This indicates a problem with:
- Hub connection to CardComputer
- Module connections to hub
- Power supply
- Wrong I2C pins configured

## Important Notes

### Module Communication Types:

**GPS Module:**
- Most GPS modules use UART (serial communication)
- If it doesn't appear on I2C scan, this is likely normal
- You'll need to use TX/RX pins instead

**RFID Unit 2:**
- Uses I2C communication (PN532 chip)
- Should appear on I2C scan

**RF 433MHz:**
- Uses GPIO digital pins (transmit/receive)
- Does NOT use I2C
- Won't appear on I2C scan (this is normal)

### Power Limitations:

The 1-to-6 I/O Hub:
- Extends I2C communication bus only
- Does NOT amplify or distribute power
- All modules draw power from the single CardComputer port
- Multiple modules may exceed available current

**Solution:** Consider external power supply for modules

## Troubleshooting Steps

1. **Start Simple:** Test one module at a time directly on Port A (no hub)
2. **Check Cables:** Ensure all connections are secure
3. **Verify Power:** CardComputer should be powered via USB or fully charged battery
4. **Pin Configuration:** Verify I2C pins in code match your CardComputer model
5. **Module Type:** Confirm whether module uses I2C, UART, or GPIO

## Pin Configuration

Default pins for CardComputer Port A:
- **G1:** I2C SDA (data)
- **G2:** I2C SCL (clock)

If diagnostics fail to initialize I2C, you may need to adjust pins in code.py:
```python
i2c = busio.I2C(board.G2, board.G1)  # SCL, SDA
```

## Next Steps

After running diagnostics:

1. **If devices found:** Note which addresses responded
2. **If no devices found:** Try modules individually without hub
3. **For GPS:** Look for UART/serial examples instead of I2C
4. **For RFID:** Should work via I2C, check power and connections
5. **For RF 433MHz:** Look for GPIO-based examples

## Additional Resources

- [M5Stack CardComputer Documentation](https://docs.m5stack.com/en/core/cardputer)
- [CircuitPython I2C Guide](https://learn.adafruit.com/circuitpython-essentials/circuitpython-i2c)
- [M5Stack Module Specifications](https://docs.m5stack.com/en/unit)
