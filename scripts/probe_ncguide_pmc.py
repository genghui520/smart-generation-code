from __future__ import annotations

import ctypes
import os
from pathlib import Path


INSTALL_DIR = Path(r"D:\Program Files (x86)\FANUC\NCGuide FS0i-D")


class IODBPMC(ctypes.Structure):
    _fields_ = [
        ("type_a", ctypes.c_short),
        ("type_d", ctypes.c_short),
        ("datano_s", ctypes.c_short),
        ("datano_e", ctypes.c_short),
        ("cdata", ctypes.c_char * 5),
    ]


def main() -> int:
    if hasattr(os, "add_dll_directory"):
        os.add_dll_directory(str(INSTALL_DIR))
    dll = ctypes.WinDLL(str(INSTALL_DIR / "Fwlib32.dll"))
    connect = dll.cnc_allclibhndl3
    connect.argtypes = [ctypes.c_char_p, ctypes.c_ushort, ctypes.c_long, ctypes.POINTER(ctypes.c_ushort)]
    connect.restype = ctypes.c_short
    free_handle = dll.cnc_freelibhndl
    free_handle.argtypes = [ctypes.c_ushort]
    free_handle.restype = ctypes.c_short
    read_pmc = dll.pmc_rdpmcrng
    read_pmc.argtypes = [ctypes.c_ushort, ctypes.c_short, ctypes.c_short, ctypes.c_ushort, ctypes.c_ushort, ctypes.c_ushort, ctypes.POINTER(IODBPMC)]
    read_pmc.restype = ctypes.c_short

    handle = ctypes.c_ushort()
    ret = int(connect(b"127.0.0.1", 8193, 10, ctypes.byref(handle)))
    print(f"connect ret={ret} handle={handle.value}")
    if ret != 0:
        return 1
    try:
        for name, address_type, address in (("X0000", 3, 0), ("X0008", 3, 8), ("X0010", 3, 16), ("G0007", 0, 7)):
            data = IODBPMC()
            ret = int(read_pmc(handle, address_type, 0, address, address, 9, ctypes.byref(data)))
            value = int.from_bytes(bytes(data.cdata[:1]), byteorder="little", signed=False) if ret == 0 else None
            print(f"{name} ret={ret} value=0x{value:02X} bits=" + (format(value, "08b") if value is not None else "-"))
    finally:
        print(f"disconnect ret={int(free_handle(handle))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
