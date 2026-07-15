from __future__ import annotations

import argparse
import ctypes
from ctypes import wintypes


user32 = ctypes.WinDLL("user32", use_last_error=True)

EnumWindowsProc = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
EnumChildProc = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)

user32.EnumWindows.argtypes = [EnumWindowsProc, wintypes.LPARAM]
user32.EnumWindows.restype = wintypes.BOOL
user32.EnumChildWindows.argtypes = [wintypes.HWND, EnumChildProc, wintypes.LPARAM]
user32.EnumChildWindows.restype = wintypes.BOOL
user32.GetWindowTextLengthW.argtypes = [wintypes.HWND]
user32.GetWindowTextLengthW.restype = ctypes.c_int
user32.GetWindowTextW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
user32.GetWindowTextW.restype = ctypes.c_int
user32.GetClassNameW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
user32.GetClassNameW.restype = ctypes.c_int
user32.IsWindowVisible.argtypes = [wintypes.HWND]
user32.IsWindowVisible.restype = wintypes.BOOL
user32.FindWindowW.argtypes = [wintypes.LPCWSTR, wintypes.LPCWSTR]
user32.FindWindowW.restype = wintypes.HWND
user32.GetWindowRect.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.RECT)]
user32.GetWindowRect.restype = wintypes.BOOL
user32.ScreenToClient.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.POINT)]
user32.ScreenToClient.restype = wintypes.BOOL


def get_text(hwnd: int) -> str:
    length = user32.GetWindowTextLengthW(hwnd)
    if length <= 0:
        return ""
    buffer = ctypes.create_unicode_buffer(length + 1)
    user32.GetWindowTextW(hwnd, buffer, length + 1)
    return buffer.value


def get_class(hwnd: int) -> str:
    buffer = ctypes.create_unicode_buffer(256)
    user32.GetClassNameW(hwnd, buffer, 256)
    return buffer.value


def list_top_windows() -> list[tuple[int, str, str]]:
    rows: list[tuple[int, str, str]] = []

    def callback(hwnd: int, lparam: int) -> bool:
        if user32.IsWindowVisible(hwnd):
            text = get_text(hwnd).strip()
            if text:
                rows.append((int(hwnd), get_class(hwnd), text))
        return True

    user32.EnumWindows(EnumWindowsProc(callback), 0)
    return sorted(rows, key=lambda item: item[2].lower())


def list_child_windows(parent: int) -> list[tuple[int, str, str]]:
    rows: list[tuple[int, str, str, tuple[int, int, int, int]]] = []

    def callback(hwnd: int, lparam: int) -> bool:
        rect = wintypes.RECT()
        bounds = (0, 0, 0, 0)
        if user32.GetWindowRect(hwnd, ctypes.byref(rect)):
            top_left = wintypes.POINT(rect.left, rect.top)
            bottom_right = wintypes.POINT(rect.right, rect.bottom)
            user32.ScreenToClient(parent, ctypes.byref(top_left))
            user32.ScreenToClient(parent, ctypes.byref(bottom_right))
            bounds = (top_left.x, top_left.y, bottom_right.x, bottom_right.y)
        rows.append((int(hwnd), get_class(hwnd), get_text(hwnd).strip(), bounds))
        return True

    user32.EnumChildWindows(parent, EnumChildProc(callback), 0)
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect Win32 windows and child controls.")
    parser.add_argument("--window-title", default="")
    parser.add_argument("--filter", default="FANUC|Machine|Signal|NCGuide|Simulator")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.window_title:
        import re

        pattern = re.compile(args.filter, re.I)
        for hwnd, class_name, text in list_top_windows():
            if pattern.search(f"{class_name} {text}"):
                print(f"{hwnd}: class={class_name!r} title={text!r}")
        return 0

    parent = user32.FindWindowW(None, args.window_title)
    if not parent:
        print(f"找不到窗口: {args.window_title}")
        return 1
    print(f"parent={parent} title={args.window_title!r} class={get_class(parent)!r}")
    children = list_child_windows(parent)
    if not children:
        print("没有枚举到子控件。该窗口可能是自绘界面，需要坐标/图像方式。")
        return 0
    for hwnd, class_name, text, bounds in children:
        x1, y1, x2, y2 = bounds
        print(f"{hwnd}: class={class_name!r} text={text!r} rect=({x1},{y1},{x2},{y2}) size=({x2-x1}x{y2-y1})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
