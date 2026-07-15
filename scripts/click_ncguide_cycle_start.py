from __future__ import annotations

import argparse
import ctypes
import json
import time
from ctypes import wintypes


user32 = ctypes.WinDLL("user32", use_last_error=True)

EnumChildProc = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)

BM_CLICK = 0x00F5
INPUT_MOUSE = 0
MOUSEEVENTF_LEFTDOWN = 0x0002
MOUSEEVENTF_LEFTUP = 0x0004


class POINT(ctypes.Structure):
    _fields_ = [("x", wintypes.LONG), ("y", wintypes.LONG)]


class MOUSEINPUT(ctypes.Structure):
    _fields_ = [
        ("dx", wintypes.LONG),
        ("dy", wintypes.LONG),
        ("mouseData", wintypes.DWORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong)),
    ]


class INPUT_UNION(ctypes.Union):
    _fields_ = [("mi", MOUSEINPUT)]


class INPUT(ctypes.Structure):
    _fields_ = [("type", wintypes.DWORD), ("union", INPUT_UNION)]


user32.FindWindowW.argtypes = [wintypes.LPCWSTR, wintypes.LPCWSTR]
user32.FindWindowW.restype = wintypes.HWND
user32.EnumChildWindows.argtypes = [wintypes.HWND, EnumChildProc, wintypes.LPARAM]
user32.EnumChildWindows.restype = wintypes.BOOL
user32.GetWindowTextLengthW.argtypes = [wintypes.HWND]
user32.GetWindowTextLengthW.restype = ctypes.c_int
user32.GetWindowTextW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
user32.GetWindowTextW.restype = ctypes.c_int
user32.GetClassNameW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
user32.GetClassNameW.restype = ctypes.c_int
user32.PostMessageW.argtypes = [wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM]
user32.PostMessageW.restype = wintypes.BOOL
user32.SetForegroundWindow.argtypes = [wintypes.HWND]
user32.SetForegroundWindow.restype = wintypes.BOOL
user32.ClientToScreen.argtypes = [wintypes.HWND, ctypes.POINTER(POINT)]
user32.ClientToScreen.restype = wintypes.BOOL
user32.SetCursorPos.argtypes = [ctypes.c_int, ctypes.c_int]
user32.SetCursorPos.restype = wintypes.BOOL
user32.SendInput.argtypes = [wintypes.UINT, ctypes.POINTER(INPUT), ctypes.c_int]
user32.SendInput.restype = wintypes.UINT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Click NCGuide Cycle Start/Run control and report success.")
    parser.add_argument("--window-title", default="Machine Signal Simulator")
    parser.add_argument("--button-text", default="Run")
    parser.add_argument("--no-button-text", action="store_true")
    parser.add_argument("--screen-coordinate", action="store_true", help="Treat --x/--y as absolute screen coordinates.")
    parser.add_argument("--x", type=int, default=0, help="Optional client-area X coordinate fallback.")
    parser.add_argument("--y", type=int, default=0, help="Optional client-area Y coordinate fallback.")
    return parser.parse_args()


def text_of(hwnd: int) -> str:
    length = user32.GetWindowTextLengthW(hwnd)
    if length <= 0:
        return ""
    buffer = ctypes.create_unicode_buffer(length + 1)
    user32.GetWindowTextW(hwnd, buffer, length + 1)
    return buffer.value


def class_of(hwnd: int) -> str:
    buffer = ctypes.create_unicode_buffer(256)
    user32.GetClassNameW(hwnd, buffer, 256)
    return buffer.value


def find_child_button(parent: int, button_text: str) -> int:
    result = {"hwnd": 0}

    def callback(hwnd: int, lparam: int) -> bool:
        if text_of(hwnd) == button_text and "BUTTON" in class_of(hwnd).upper():
            result["hwnd"] = int(hwnd)
            return False
        return True

    user32.EnumChildWindows(parent, EnumChildProc(callback), 0)
    return result["hwnd"]


def click_child_button(parent: int, button: int) -> bool:
    user32.SetForegroundWindow(parent)
    time.sleep(0.1)
    if not user32.PostMessageW(button, BM_CLICK, 0, 0):
        return False
    time.sleep(0.2)
    return True


def click_client_coordinate(parent: int, x: int, y: int) -> bool:
    point = POINT(x, y)
    if not user32.ClientToScreen(parent, ctypes.byref(point)):
        return False
    user32.SetForegroundWindow(parent)
    time.sleep(0.1)
    if not user32.SetCursorPos(point.x, point.y):
        return False
    inputs = (INPUT * 2)()
    inputs[0].type = INPUT_MOUSE
    inputs[0].union.mi.dwFlags = MOUSEEVENTF_LEFTDOWN
    inputs[1].type = INPUT_MOUSE
    inputs[1].union.mi.dwFlags = MOUSEEVENTF_LEFTUP
    sent = user32.SendInput(2, inputs, ctypes.sizeof(INPUT))
    time.sleep(0.2)
    return sent == 2


def click_screen_coordinate(parent: int, x: int, y: int) -> bool:
    user32.SetForegroundWindow(parent)
    time.sleep(0.1)
    if not user32.SetCursorPos(x, y):
        return False
    inputs = (INPUT * 2)()
    inputs[0].type = INPUT_MOUSE
    inputs[0].union.mi.dwFlags = MOUSEEVENTF_LEFTDOWN
    inputs[1].type = INPUT_MOUSE
    inputs[1].union.mi.dwFlags = MOUSEEVENTF_LEFTUP
    sent = user32.SendInput(2, inputs, ctypes.sizeof(INPUT))
    time.sleep(0.2)
    return sent == 2


def result(success: bool, **payload: object) -> int:
    print(json.dumps({"success": success, **payload}, ensure_ascii=False))
    return 0 if success else 1


def main() -> int:
    args = parse_args()
    if args.no_button_text:
        args.button_text = ""
    parent = user32.FindWindowW(None, args.window_title)
    if not parent:
        return result(False, error="window_not_found", window_title=args.window_title)

    button = find_child_button(parent, args.button_text) if args.button_text else 0
    if button:
        if click_child_button(parent, button):
            return result(
                True,
                method="button_text",
                window_title=args.window_title,
                button_text=args.button_text,
                button_hwnd=button,
            )
        return result(False, error="button_click_failed", window_title=args.window_title, button_text=args.button_text)

    if args.x > 0 and args.y > 0:
        if args.screen_coordinate:
            if click_screen_coordinate(parent, args.x, args.y):
                return result(True, method="screen_coordinate", window_title=args.window_title, x=args.x, y=args.y)
            return result(False, error="coordinate_click_failed", window_title=args.window_title, x=args.x, y=args.y)
        if click_client_coordinate(parent, args.x, args.y):
            return result(True, method="client_coordinate", window_title=args.window_title, x=args.x, y=args.y)
        return result(False, error="coordinate_click_failed", window_title=args.window_title, x=args.x, y=args.y)

    return result(
        False,
        error="button_not_found_and_no_coordinate",
        window_title=args.window_title,
        button_text=args.button_text,
    )


if __name__ == "__main__":
    raise SystemExit(main())
