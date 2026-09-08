"""Upload one small NC program to FANUC NCGuide through FOCAS."""

from __future__ import annotations

import argparse
import ctypes
import os
from pathlib import Path


DEFAULT_INSTALL_DIR = Path(r"D:\Program Files (x86)\FANUC\NCGuide FS0i-D")
DEFAULT_PROGRAM = Path(__file__).resolve().parents[1] / "nc_programs" / "O3471.nc"


class ODBST(ctypes.Structure):
    _fields_ = [
        ("hdck", ctypes.c_short),
        ("tmmode", ctypes.c_short),
        ("aut", ctypes.c_short),
        ("run", ctypes.c_short),
        ("motion", ctypes.c_short),
        ("mstb", ctypes.c_short),
        ("emergency", ctypes.c_short),
        ("alarm", ctypes.c_short),
        ("edit", ctypes.c_short),
    ]


def focas_function(dll: ctypes.WinDLL, name: str, restype, argtypes):
    function = getattr(dll, name)
    function.restype = restype
    function.argtypes = argtypes
    return function


def ret_text(value: int) -> str:
    return "EW_OK" if value == 0 else f"FOCAS_RET_{value}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--install-dir", type=Path, default=DEFAULT_INSTALL_DIR)
    parser.add_argument("--program", type=Path, default=DEFAULT_PROGRAM)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8193)
    parser.add_argument("--timeout", type=int, default=10)
    parser.add_argument("--status-only", action="store_true")
    args = parser.parse_args()

    dll_path = args.install_dir / "Fwlib32.dll"
    if not dll_path.exists():
        print(f"[ERROR] Fwlib32.dll not found: {dll_path}")
        return 2
    if not args.program.exists():
        print(f"[ERROR] NC program not found: {args.program}")
        return 2

    if hasattr(os, "add_dll_directory"):
        os.add_dll_directory(str(args.install_dir))
    dll = ctypes.WinDLL(str(dll_path))

    connect = focas_function(
        dll,
        "cnc_allclibhndl3",
        ctypes.c_short,
        [ctypes.c_char_p, ctypes.c_ushort, ctypes.c_long, ctypes.POINTER(ctypes.c_ushort)],
    )
    free_handle = focas_function(dll, "cnc_freelibhndl", ctypes.c_short, [ctypes.c_ushort])
    statinfo = focas_function(dll, "cnc_statinfo", ctypes.c_short, [ctypes.c_ushort, ctypes.POINTER(ODBST)])
    dwnstart3 = focas_function(dll, "cnc_dwnstart3", ctypes.c_short, [ctypes.c_ushort, ctypes.c_short])
    download3 = focas_function(
        dll,
        "cnc_download3",
        ctypes.c_short,
        [ctypes.c_ushort, ctypes.POINTER(ctypes.c_long), ctypes.POINTER(ctypes.c_char)],
    )
    dwnend3 = focas_function(dll, "cnc_dwnend3", ctypes.c_short, [ctypes.c_ushort])

    handle = ctypes.c_ushort()
    result = int(connect(args.host.encode("ascii"), args.port, args.timeout, ctypes.byref(handle)))
    print(f"[CONNECT] ret={result} {ret_text(result)} handle={handle.value}")
    if result != 0:
        return 3

    try:
        status = ODBST()
        status_result = int(statinfo(handle, ctypes.byref(status)))
        print(
            f"[STATUS] ret={status_result} {ret_text(status_result)} "
            f"hdck={status.hdck} tmmode={status.tmmode} aut={status.aut} "
            f"run={status.run} motion={status.motion} mstb={status.mstb} "
            f"emergency={status.emergency} alarm={status.alarm} edit={status.edit}"
        )
        if status_result != 0:
            return 4
        if args.status_only:
            return 0
        if status.alarm != 0 or status.emergency != 0:
            print("[ABORT] alarm or emergency is active")
            return 5
        if status.run == 3 or status.motion != 0:
            print("[ABORT] NCGuide is still running; stop/reset it before upload")
            return 6

        # FOCAS expects one final '%' character; tolerate a text editor's
        # trailing CR/LF without changing the NC content.
        payload = args.program.read_bytes().rstrip(b"\r\n")
        if not payload.startswith(b"\n") or not payload.endswith(b"%"):
            print("[ERROR] payload must start with LF and end with '%'")
            return 7
        payload_buffer = ctypes.create_string_buffer(payload)

        result = int(dwnstart3(handle, 0))
        print(f"[UPLOAD START] ret={result} {ret_text(result)}")
        if result != 0:
            return 8

        remaining = len(payload)
        offset = 0
        while remaining:
            requested = ctypes.c_long(remaining)
            data_ptr = ctypes.cast(ctypes.byref(payload_buffer, offset), ctypes.POINTER(ctypes.c_char))
            result = int(download3(handle, ctypes.byref(requested), data_ptr))
            print(f"[UPLOAD DATA] ret={result} {ret_text(result)} requested={remaining} accepted={requested.value}")
            if result != 0:
                return 9
            if requested.value <= 0 or requested.value > remaining:
                print("[ERROR] cnc_download3 returned an invalid accepted length")
                return 10
            offset += requested.value
            remaining -= requested.value

        result = int(dwnend3(handle))
        print(f"[UPLOAD END] ret={result} {ret_text(result)}")
        return 0 if result == 0 else 11
    finally:
        free_result = int(free_handle(handle))
        print(f"[DISCONNECT] ret={free_result} {ret_text(free_result)}")


if __name__ == "__main__":
    raise SystemExit(main())
