from __future__ import annotations

import argparse
import ctypes
import time
from ctypes import wintypes


user32 = ctypes.WinDLL("user32", use_last_error=True)


class POINT(ctypes.Structure):
    _fields_ = [("x", wintypes.LONG), ("y", wintypes.LONG)]


class RECT(ctypes.Structure):
    _fields_ = [
        ("left", wintypes.LONG),
        ("top", wintypes.LONG),
        ("right", wintypes.LONG),
        ("bottom", wintypes.LONG),
    ]


user32.FindWindowW.argtypes = [wintypes.LPCWSTR, wintypes.LPCWSTR]
user32.FindWindowW.restype = wintypes.HWND
user32.EnumWindows.argtypes = [ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM), wintypes.LPARAM]
user32.EnumWindows.restype = wintypes.BOOL
user32.GetWindowTextLengthW.argtypes = [wintypes.HWND]
user32.GetWindowTextLengthW.restype = ctypes.c_int
user32.GetWindowTextW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
user32.GetWindowTextW.restype = ctypes.c_int
user32.IsWindowVisible.argtypes = [wintypes.HWND]
user32.IsWindowVisible.restype = wintypes.BOOL
user32.GetCursorPos.argtypes = [ctypes.POINTER(POINT)]
user32.GetCursorPos.restype = wintypes.BOOL
user32.ScreenToClient.argtypes = [wintypes.HWND, ctypes.POINTER(POINT)]
user32.ScreenToClient.restype = wintypes.BOOL
user32.GetClientRect.argtypes = [wintypes.HWND, ctypes.POINTER(RECT)]
user32.GetClientRect.restype = wintypes.BOOL


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Print mouse coordinates relative to the FANUC NCGuide window."
    )
    parser.add_argument("--window-title", default="FANUC CNC GUIDE")
    parser.add_argument("--list-windows", action="store_true")
    parser.add_argument("--interval", type=float, default=0.2)
    parser.add_argument("--once", action="store_true")
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Capture mode and cycle-start coordinates by pressing Enter while the mouse is over each button.",
    )
    return parser.parse_args()


def get_last_error_message() -> str:
    code = ctypes.get_last_error()
    return f"Win32 error {code}" if code else "unknown Win32 error"


def main() -> int:
    args = parse_args()
    if args.list_windows:
        return list_windows()

    hwnd = user32.FindWindowW(None, args.window_title)
    if not hwnd:
        print(f'找不到窗口: "{args.window_title}"')
        print("请先打开 FANUC NCGuide，或用 --window-title 指定实际窗口标题。")
        return 1

    rect = RECT()
    if not user32.GetClientRect(hwnd, ctypes.byref(rect)):
        print(f"读取窗口客户区失败: {get_last_error_message()}")
        return 1

    width = rect.right - rect.left
    height = rect.bottom - rect.top
    print(f'已找到窗口: "{args.window_title}" hwnd={hwnd}')
    print(f"窗口客户区大小: width={width}, height={height}")
    if args.interactive:
        return interactive_capture(hwnd, args.window_title)

    print("把鼠标移动到 MEM/AUTO 或 Cycle Start 按钮上，读取 client_x/client_y。按 Ctrl+C 结束。")
    print()

    while True:
        screen = POINT()
        if not user32.GetCursorPos(ctypes.byref(screen)):
            print(f"读取鼠标位置失败: {get_last_error_message()}")
            return 1

        client = POINT(screen.x, screen.y)
        if not user32.ScreenToClient(hwnd, ctypes.byref(client)):
            print(f"坐标转换失败: {get_last_error_message()}")
            return 1

        inside = 0 <= client.x < width and 0 <= client.y < height
        print(
            f"screen=({screen.x:4d},{screen.y:4d}) "
            f"client=({client.x:4d},{client.y:4d}) "
            f"inside={inside}",
            flush=True,
        )
        if args.once:
            return 0
        time.sleep(max(args.interval, 0.05))


def current_client_point(hwnd: int) -> tuple[int, int]:
    screen = POINT()
    if not user32.GetCursorPos(ctypes.byref(screen)):
        raise RuntimeError(f"读取鼠标位置失败: {get_last_error_message()}")
    client = POINT(screen.x, screen.y)
    if not user32.ScreenToClient(hwnd, ctypes.byref(client)):
        raise RuntimeError(f"坐标转换失败: {get_last_error_message()}")
    return int(client.x), int(client.y)


def wait_for_point(hwnd: int, label: str) -> tuple[int, int]:
    input(f"把鼠标移动到 {label} 上，然后按 Enter ...")
    x, y = current_client_point(hwnd)
    print(f"{label}: x={x}, y={y}")
    return x, y


def interactive_capture(hwnd: int, window_title: str) -> int:
    print()
    print("交互校准模式")
    print("如果不需要模式按钮，可以在第一步直接按 Enter，然后后续命令里不使用 mode 参数。")
    mode_x, mode_y = wait_for_point(hwnd, "MEM/AUTO 模式按钮")
    start_x, start_y = wait_for_point(hwnd, "Cycle Start 按钮")
    print()
    print("可复制运行命令：")
    print(
        '& ".venv\\Scripts\\python.exe" main.py `\n'
        "  --llm-provider disabled `\n"
        "  --trigger-ncguide-ui `\n"
        f'  --ncguide-window-title "{window_title}" `\n'
        f"  --ncguide-mode-x {mode_x} `\n"
        f"  --ncguide-mode-y {mode_y} `\n"
        f"  --ncguide-cycle-start-x {start_x} `\n"
        f"  --ncguide-cycle-start-y {start_y}"
    )
    return 0


def list_windows() -> int:
    rows: list[tuple[int, str]] = []

    callback_type = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)

    def callback(hwnd: int, lparam: int) -> bool:
        if not user32.IsWindowVisible(hwnd):
            return True
        length = user32.GetWindowTextLengthW(hwnd)
        if length <= 0:
            return True
        buffer = ctypes.create_unicode_buffer(length + 1)
        user32.GetWindowTextW(hwnd, buffer, length + 1)
        title = buffer.value.strip()
        if title:
            rows.append((int(hwnd), title))
        return True

    user32.EnumWindows(callback_type(callback), 0)
    for hwnd, title in sorted(rows, key=lambda item: item[1].lower()):
        print(f"{hwnd}: {title}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
