"""Test Windows cursor injection without clicking any window."""
from __future__ import annotations

import argparse
import ctypes
import json
import os
import sys
import time
from ctypes import wintypes


if os.name != "nt":
    raise SystemExit("This diagnostic requires Windows.")

user32 = ctypes.WinDLL("user32", use_last_error=True)


class POINT(ctypes.Structure):
    _fields_ = [("x", wintypes.LONG), ("y", wintypes.LONG)]


user32.GetCursorPos.argtypes = [ctypes.POINTER(POINT)]
user32.GetCursorPos.restype = wintypes.BOOL
user32.SetCursorPos.argtypes = [ctypes.c_int, ctypes.c_int]
user32.SetCursorPos.restype = wintypes.BOOL
user32.GetForegroundWindow.restype = wintypes.HWND
user32.WindowFromPoint.argtypes = [POINT]
user32.WindowFromPoint.restype = wintypes.HWND
user32.GetWindowTextLengthW.argtypes = [wintypes.HWND]
user32.GetWindowTextLengthW.restype = ctypes.c_int
user32.GetWindowTextW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
user32.GetWindowTextW.restype = ctypes.c_int
user32.GetClassNameW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
user32.GetClassNameW.restype = ctypes.c_int
user32.GetAncestor.argtypes = [wintypes.HWND, wintypes.UINT]
user32.GetAncestor.restype = wintypes.HWND
user32.GetSystemMetrics.argtypes = [ctypes.c_int]
user32.GetSystemMetrics.restype = ctypes.c_int


def window_title(hwnd: int) -> str:
    size = user32.GetWindowTextLengthW(hwnd)
    buf = ctypes.create_unicode_buffer(max(size + 1, 2))
    user32.GetWindowTextW(hwnd, buf, len(buf))
    return buf.value


def window_class(hwnd: int) -> str:
    buf = ctypes.create_unicode_buffer(256)
    user32.GetClassNameW(hwnd, buf, len(buf))
    return buf.value


def window_info(hwnd: int) -> dict[str, object]:
    hwnd = int(hwnd or 0)
    root = int(user32.GetAncestor(hwnd, 2) or hwnd) if hwnd else 0  # GA_ROOT
    return {"hwnd": hwnd, "title": window_title(hwnd), "class": window_class(hwnd),
            "root_hwnd": root, "root_title": window_title(root), "root_class": window_class(root)}


def cursor() -> tuple[bool, int, int, int]:
    point = POINT()
    ctypes.set_last_error(0)
    ok = bool(user32.GetCursorPos(ctypes.byref(point)))
    return ok, point.x, point.y, ctypes.get_last_error()


def main() -> int:
    parser = argparse.ArgumentParser(description="Move-test the Windows cursor; this never clicks.")
    parser.add_argument("--x", type=int, help="screen X; default is current X + 80")
    parser.add_argument("--y", type=int, help="screen Y; default is current Y + 40")
    parser.add_argument("--repeat", type=int, default=3)
    args = parser.parse_args()

    before_ok, before_x, before_y, before_error = cursor()
    x = before_x + 80 if args.x is None else args.x
    y = before_y + 40 if args.y is None else args.y
    # SM_XVIRTUALSCREEN...SM_CYVIRTUALSCREEN avoids testing outside a multi-monitor desktop.
    bounds = {
        "left": user32.GetSystemMetrics(76), "top": user32.GetSystemMetrics(77),
        "width": user32.GetSystemMetrics(78), "height": user32.GetSystemMetrics(79),
    }
    results = []
    for _ in range(max(1, args.repeat)):
        ctypes.set_last_error(0)
        moved = bool(user32.SetCursorPos(x, y))
        set_error = ctypes.get_last_error()
        time.sleep(0.08)
        read_ok, actual_x, actual_y, read_error = cursor()
        results.append({"set_cursor_ok": moved, "set_cursor_error": set_error,
                        "get_cursor_ok": read_ok, "get_cursor_error": read_error,
                        "actual_x": actual_x, "actual_y": actual_y,
                        "matches_target": read_ok and actual_x == x and actual_y == y})

    payload = {
        "before": {"ok": before_ok, "x": before_x, "y": before_y, "error": before_error},
        "target": {"x": x, "y": y}, "virtual_screen": bounds,
        "foreground": window_info(user32.GetForegroundWindow()),
        "target": window_info(user32.WindowFromPoint(POINT(x, y))),
        "results": results,
        "all_matches": bool(results) and all(row["matches_target"] for row in results),
        "note": "No mouse button event was sent.",
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if payload["all_matches"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
