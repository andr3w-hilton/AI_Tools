@echo off
echo Compiling Keylogger.cs to Keylogger.exe...
echo.

REM Find .NET Framework compiler (should be available on all Windows machines)
set "csc=%SystemRoot%\Microsoft.NET\Framework\v4.0.30319\csc.exe"

if not exist "%csc%" (
    echo Trying 64-bit Framework path...
    set "csc=%SystemRoot%\Microsoft.NET\Framework64\v4.0.30319\csc.exe"
)

if not exist "%csc%" (
    echo [!] C# compiler not found. Trying newer .NET SDK...
    where csc.exe >nul 2>&1
    if errorlevel 1 (
        echo [!] Error: C# compiler not found on this system.
        echo [!] Please install .NET Framework or .NET SDK
        pause
        exit /b 1
    )
    set "csc=csc.exe"
)

echo Using compiler: %csc%
echo.

REM Compile the keylogger
"%csc%" /out:Keylogger.exe /target:exe /optimize+ Keylogger.cs

if errorlevel 1 (
    echo.
    echo [!] Compilation failed!
    pause
    exit /b 1
)

echo.
echo [*] Success! Keylogger.exe created
echo [*] You can now run: Keylogger.exe
echo.
pause
