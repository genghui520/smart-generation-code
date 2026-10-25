from __future__ import annotations

import ctypes
from ctypes import wintypes


user32 = ctypes.windll.user32
target_pids: set[int] | None = None


def title(hwnd: int) -> str:
    length = user32.GetWindowTextLengthW(hwnd)
    buffer = ctypes.create_unicode_buffer(max(length + 1, 2))
    user32.GetWindowTextW(hwnd, buffer, len(buffer))
    return buffer.value


def class_name(hwnd: int) -> str:
    buffer = ctypes.create_unicode_buffer(256)
    user32.GetClassNameW(hwnd, buffer, len(buffer))
    return buffer.value


records: list[tuple[int, int, bool, str, str]] = []


@ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
def collect(hwnd: int, _param: int) -> bool:
    pid = wintypes.DWORD()
    user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
    window_title = title(hwnd)
    if (target_pids is None and (user32.IsWindowVisible(hwnd) or window_title)) or (
        target_pids is not None and pid.value in target_pids
    ):
        records.append(
            (pid.value, int(hwnd), bool(user32.IsWindowVisible(hwnd)), class_name(hwnd), window_title)
        )
    return True


user32.EnumWindows(collect, 0)
for record in records:
    print(record)
