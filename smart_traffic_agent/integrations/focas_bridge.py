from __future__ import annotations

import argparse
import ctypes
import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


DEFAULT_INSTALL_DIR = Path(r"D:\Program Files (x86)\FANUC\NCGuide FS0i-F")


@dataclass(slots=True)
class BridgeConfig:
    install_dir: Path = DEFAULT_INSTALL_DIR
    host: str = "127.0.0.1"
    port: int = 8193
    timeout_seconds: int = 10


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


def python_bits() -> int:
    return 64 if sys.maxsize > 2**32 else 32


def make_response(status_code: int, **payload: Any) -> dict[str, Any]:
    return {"status_code": status_code, **payload}


class FocasBridge:
    """Small 32-bit helper for FANUC FOCAS calls.

    This module is intentionally dependency-light so it can run under a separate
    32-bit Python installation while the main SMPAgent process remains 64-bit.
    """

    def __init__(self, config: BridgeConfig) -> None:
        self.config = config
        self.dll: Any | None = None
        self.handle: ctypes.c_ushort | None = None

    def load_library(self) -> None:
        dll_path = self.config.install_dir / "Fwlib32.dll"
        if not dll_path.exists():
            raise FileNotFoundError(f"FOCAS DLL not found: {dll_path}")
        if hasattr(os, "add_dll_directory"):
            os.add_dll_directory(str(self.config.install_dir))
        self.dll = ctypes.WinDLL(str(dll_path))
        self.dll.cnc_allclibhndl3.argtypes = [
            ctypes.c_char_p,
            ctypes.c_ushort,
            ctypes.c_long,
            ctypes.POINTER(ctypes.c_ushort),
        ]
        self.dll.cnc_allclibhndl3.restype = ctypes.c_short
        self.dll.cnc_freelibhndl.argtypes = [ctypes.c_ushort]
        self.dll.cnc_freelibhndl.restype = ctypes.c_short
        self.dll.cnc_statinfo.argtypes = [ctypes.c_ushort, ctypes.POINTER(ODBST)]
        self.dll.cnc_statinfo.restype = ctypes.c_short

    def connect(self) -> dict[str, Any]:
        if self.dll is None:
            self.load_library()
        assert self.dll is not None
        handle = ctypes.c_ushort()
        result = self.dll.cnc_allclibhndl3(
            self.config.host.encode("ascii"),
            ctypes.c_ushort(self.config.port),
            ctypes.c_long(self.config.timeout_seconds),
            ctypes.byref(handle),
        )
        if result != 0:
            return make_response(
                int(result),
                error="cnc_allclibhndl3 failed",
                host=self.config.host,
                port=self.config.port,
            )
        self.handle = handle
        return make_response(0, handle=int(handle.value), host=self.config.host, port=self.config.port)

    def disconnect(self) -> dict[str, Any]:
        if self.dll is None or self.handle is None:
            return make_response(0, disconnected=False)
        result = self.dll.cnc_freelibhndl(self.handle)
        self.handle = None
        return make_response(int(result), disconnected=result == 0)

    def read_run_status(self) -> dict[str, Any]:
        connected_here = False
        if self.handle is None:
            connected = self.connect()
            if connected["status_code"] != 0:
                return connected
            connected_here = True
        assert self.dll is not None and self.handle is not None
        status = ODBST()
        result = self.dll.cnc_statinfo(self.handle, ctypes.byref(status))
        payload = make_response(
            int(result),
            function="cnc_statinfo",
            statinfo=asdict_odb_status(status),
        )
        if connected_here:
            payload["disconnect"] = self.disconnect()
        return payload


def asdict_odb_status(status: ODBST) -> dict[str, int]:
    return {name: int(getattr(status, name)) for name, _ in status._fields_}


def run_once(action: str, config: BridgeConfig) -> dict[str, Any]:
    if python_bits() != 32:
        return make_response(
            400,
            error="FOCAS bridge must be run with 32-bit Python for the installed NCGuide DLLs.",
            python_bits=python_bits(),
        )
    bridge = FocasBridge(config)
    try:
        if action == "probe":
            bridge.load_library()
            return make_response(0, loaded=True, python_bits=python_bits(), install_dir=str(config.install_dir))
        if action == "connect":
            result = bridge.connect()
            bridge.disconnect()
            return result
        if action == "read_run_status":
            return bridge.read_run_status()
        return make_response(400, error=f"Unsupported action: {action}")
    except Exception as exc:
        return make_response(500, error=str(exc), python_bits=python_bits())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="focas-bridge")
    parser.add_argument("--install-dir", type=Path, default=DEFAULT_INSTALL_DIR)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8193)
    parser.add_argument("--timeout", type=int, default=10)
    parser.add_argument("--action", choices=["probe", "connect", "read_run_status"], default="read_run_status")
    args = parser.parse_args(argv)

    result = run_once(
        args.action,
        BridgeConfig(
            install_dir=args.install_dir,
            host=args.host,
            port=args.port,
            timeout_seconds=args.timeout,
        ),
    )
    print(json.dumps(result, ensure_ascii=False))
    return 0 if result.get("status_code") == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
