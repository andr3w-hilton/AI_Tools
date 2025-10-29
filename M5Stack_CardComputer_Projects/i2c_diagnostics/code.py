"""
I2C Diagnostics Tool for M5Stack CardComputer with 1-to-6 I/O Hub
Tests GPS, RFID, and RF modules connected through the hub

Hardware Setup:
- M5Stack CardComputer v1.1
- 1-to-6 I/O Hub connected to Port A
- GPS/BDS v1.1 Module
- RFID Unit 2
- RF 433MHz Module

Author: Diagnostic Script
Date: 2025-10-24
"""

import board
import busio
import time
import digitalio

# Display setup
try:
    import displayio
    import terminalio
    from adafruit_display_text import label
    display = board.DISPLAY
    HAS_DISPLAY = True
except:
    HAS_DISPLAY = False
    print("Display not available, using serial only")

def print_both(text):
    """Print to both serial and display if available"""
    print(text)
    # Could add display output here if needed

def scan_i2c_bus(i2c):
    """Scan I2C bus and return list of addresses found"""
    print_both("\n" + "="*40)
    print_both("Scanning I2C Bus...")
    print_both("="*40)

    devices_found = []

    while not i2c.try_lock():
        pass

    try:
        devices = i2c.scan()
        print_both(f"\nFound {len(devices)} device(s):")

        for device_address in devices:
            hex_addr = hex(device_address)
            devices_found.append(device_address)

            # Identify common module addresses
            device_name = "Unknown"
            if device_address == 0x42:
                device_name = "GPS Module (typical)"
            elif device_address in [0x28, 0x29]:
                device_name = "RFID Reader (PN532 possible)"
            elif device_address == 0x55:
                device_name = "ENV Sensor (possible)"
            elif device_address in range(0x08, 0x78):
                device_name = "I2C Device"

            print_both(f"  • Address: {hex_addr} ({device_address}) - {device_name}")

        if len(devices) == 0:
            print_both("  ⚠ WARNING: No I2C devices detected!")
            print_both("  Check connections and power supply")

    finally:
        i2c.unlock()

    return devices_found

def test_gps_module(i2c):
    """Test GPS/BDS module communication"""
    print_both("\n" + "="*40)
    print_both("Testing GPS Module...")
    print_both("="*40)

    # GPS modules typically use UART, not I2C
    print_both("Note: GPS modules usually use UART/serial")
    print_both("Checking for I2C GPS at 0x42...")

    while not i2c.try_lock():
        pass

    try:
        if 0x42 in i2c.scan():
            print_both("✓ GPS I2C device found at 0x42")
            return True
        else:
            print_both("✗ No GPS device at 0x42")
            print_both("  → GPS may use UART (TX/RX pins)")
            return False
    finally:
        i2c.unlock()

def test_rfid_module(i2c):
    """Test RFID Unit 2 communication"""
    print_both("\n" + "="*40)
    print_both("Testing RFID Module...")
    print_both("="*40)

    # RFID Unit 2 typically uses PN532 chip at 0x24 or 0x48
    common_addresses = [0x24, 0x48, 0x28, 0x29]

    while not i2c.try_lock():
        pass

    try:
        found_devices = i2c.scan()
        rfid_found = False

        for addr in common_addresses:
            if addr in found_devices:
                print_both(f"✓ Possible RFID device at {hex(addr)}")
                rfid_found = True

        if not rfid_found:
            print_both("✗ No RFID device detected")
            print_both(f"  Expected at: {[hex(a) for a in common_addresses]}")

        return rfid_found
    finally:
        i2c.unlock()

def test_rf_module():
    """Test RF 433MHz module"""
    print_both("\n" + "="*40)
    print_both("Testing RF 433MHz Module...")
    print_both("="*40)

    # RF modules typically use GPIO pins, not I2C
    print_both("Note: RF 433MHz modules use GPIO pins")
    print_both("  → Transmitter: Digital output pin")
    print_both("  → Receiver: Digital input pin")
    print_both("  → These don't appear on I2C bus")
    print_both("✓ RF modules don't use I2C (this is normal)")
    return True

def check_power_and_connections():
    """Provide guidance on power and connection issues"""
    print_both("\n" + "="*40)
    print_both("Power & Connection Checklist")
    print_both("="*40)

    print_both("\n1. I/O Hub Connection:")
    print_both("   □ Hub firmly connected to CardComputer Port A")
    print_both("   □ All module cables securely connected to hub")

    print_both("\n2. Power Supply:")
    print_both("   □ CardComputer has sufficient power (USB or battery)")
    print_both("   □ Consider external power for multiple modules")
    print_both("   ⚠ Hub does NOT distribute power - shares data only")

    print_both("\n3. Module Types:")
    print_both("   • GPS: Usually UART (TX/RX), may have I2C variant")
    print_both("   • RFID: I2C communication (PN532 chip)")
    print_both("   • RF 433MHz: GPIO digital pins (not I2C)")

    print_both("\n4. Common Issues:")
    print_both("   • Wrong port on hub")
    print_both("   • Insufficient current for all modules")
    print_both("   • Cable/connector issues")
    print_both("   • Module needs specific initialization")

def main():
    """Main diagnostic routine"""
    print_both("="*40)
    print_both("M5Stack I/O Hub Diagnostics")
    print_both("CardComputer v1.1")
    print_both("="*40)

    try:
        # Initialize I2C bus (Port A)
        # CardComputer Port A typically uses G1 (SDA) and G2 (SCL)
        print_both("\nInitializing I2C bus on Port A...")

        # Try common I2C pin configurations for CardComputer
        try:
            i2c = busio.I2C(board.G2, board.G1)  # SCL, SDA
            print_both("✓ I2C initialized on G2 (SCL) / G1 (SDA)")
        except Exception as e:
            print_both(f"✗ I2C init failed: {e}")
            print_both("Check pin configuration for your CardComputer model")
            return

        # Wait for I2C to stabilize
        time.sleep(1)

        # Run diagnostics
        devices = scan_i2c_bus(i2c)

        test_gps_module(i2c)
        test_rfid_module(i2c)
        test_rf_module()

        check_power_and_connections()

        # Summary
        print_both("\n" + "="*40)
        print_both("Diagnostic Summary")
        print_both("="*40)
        print_both(f"Total I2C devices found: {len(devices)}")

        if len(devices) == 0:
            print_both("\n⚠ TROUBLESHOOTING STEPS:")
            print_both("1. Check if hub is connected to correct port")
            print_both("2. Try connecting one module directly (no hub)")
            print_both("3. Verify module is receiving power")
            print_both("4. Check cable connections")
            print_both("5. Some modules may use UART, not I2C")
        else:
            print_both("\n✓ I2C devices detected - checking module types")

        print_both("\n" + "="*40)
        print_both("Diagnostic complete!")
        print_both("="*40)

        i2c.deinit()

    except Exception as e:
        print_both(f"\n✗ Error during diagnostics: {e}")
        import traceback
        traceback.print_exception(type(e), e, e.__traceback__)

# Run diagnostics
if __name__ == "__main__":
    main()
    print_both("\nDiagnostics finished. Check output above.")
