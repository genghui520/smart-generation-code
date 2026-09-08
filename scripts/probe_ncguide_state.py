from __future__ import annotations

import ctypes
import os
from pathlib import Path


INSTALL_DIR = Path(r"D:\Program Files (x86)\FANUC\NCGuide FS0i-D")


class ODBST(ctypes.Structure):
    _fields_ = [
        ("hdck", ctypes.c_short), ("tmmode", ctypes.c_short),
        ("aut", ctypes.c_short), ("run", ctypes.c_short),
        ("motion", ctypes.c_short), ("mstb", ctypes.c_short),
        ("emergency", ctypes.c_short), ("alarm", ctypes.c_short),
        ("edit", ctypes.c_short),
    ]


class IODBSGNL(ctypes.Structure):
    _fields_ = [
        ("datano", ctypes.c_short), ("type", ctypes.c_short),
        ("mode", ctypes.c_short), ("hndl_ax", ctypes.c_short),
        ("hndl_mv", ctypes.c_short), ("rpd_ovrd", ctypes.c_short),
        ("jog_ovrd", ctypes.c_short), ("feed_ovrd", ctypes.c_short),
        ("spdl_ovrd", ctypes.c_short), ("blck_del", ctypes.c_short),
        ("sngl_blck", ctypes.c_short), ("machn_lock", ctypes.c_short),
        ("dry_run", ctypes.c_short), ("mem_prtct", ctypes.c_short),
        ("feed_hold", ctypes.c_short),
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
    statinfo = dll.cnc_statinfo
    statinfo.argtypes = [ctypes.c_ushort, ctypes.POINTER(ODBST)]
    statinfo.restype = ctypes.c_short
    rdopnlsgnl = dll.cnc_rdopnlsgnl
    rdopnlsgnl.argtypes = [ctypes.c_ushort, ctypes.c_short, ctypes.POINTER(IODBSGNL)]
    rdopnlsgnl.restype = ctypes.c_short

    handle = ctypes.c_ushort()
    ret = int(connect(b"127.0.0.1", 8193, 10, ctypes.byref(handle)))
    print(f"connect ret={ret} handle={handle.value}")
    if ret != 0:
        return 1
    try:
        status = ODBST()
        ret = int(statinfo(handle, ctypes.byref(status)))
        print(
            f"cnc_statinfo ret={ret} aut={status.aut} run={status.run} "
            f"motion={status.motion} emergency={status.emergency} alarm={status.alarm} edit={status.edit}"
        )
        panel = IODBSGNL()
        ret = int(rdopnlsgnl(handle, -1, ctypes.byref(panel)))
        print(
            f"cnc_rdopnlsgnl(-1) ret={ret} type={panel.type} mode={panel.mode} "
            f"sngl_blck={panel.sngl_blck} machn_lock={panel.machn_lock} "
            f"dry_run={panel.dry_run} mem_prtct={panel.mem_prtct} feed_hold={panel.feed_hold}"
        )
    finally:
        print(f"disconnect ret={int(free_handle(handle))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
