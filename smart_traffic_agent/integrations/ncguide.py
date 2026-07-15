from __future__ import annotations

import ctypes
import json
import os
import socket
import struct
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


NCGUIDE_FS0I_D_DIR = Path(r"D:\Program Files (x86)\FANUC\NCGuide FS0i-D")
NCGUIDE_FS0I_F_DIR = Path(r"D:\Program Files (x86)\FANUC\NCGuide FS0i-F")
DEFAULT_NCGUIDE_DIR = NCGUIDE_FS0I_D_DIR
DEFAULT_FOCAS_RUNTIME_DIR = Path(r"E:\机床资料\三菱\DNC_demo1\DNC_demo1\Debug")
DEFAULT_NCGUIDE_HOST = "127.0.0.1"
DEFAULT_FOCAS_PORTS = [8193, 8194]
FOCAS_DLL_NAMES = ["Fwlib32.dll", "fwlibNCG.dll"]
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FOCAS_HEADER_DIR = Path(r"C:\Lib\FOCAS2 Library\Fwlib\0iD")


def default_focas_runtime_dir() -> Path:
    """Return the FOCAS DLL directory that matches the working NCGuide demo."""
    for candidate in (PROJECT_ROOT, Path.cwd()):
        if (candidate / "Fwlib32.dll").exists():
            return candidate
    if DEFAULT_FOCAS_RUNTIME_DIR.exists():
        return DEFAULT_FOCAS_RUNTIME_DIR
    return DEFAULT_NCGUIDE_DIR


def default_focas_header_dir() -> Path:
    """Return the FANUC Series 0i-D SDK include directory used by generated C++."""
    configured = os.environ.get("FOCAS_HEADER_DIR", "").strip()
    if configured:
        return Path(configured)
    return DEFAULT_FOCAS_HEADER_DIR


@dataclass(slots=True)
class NcGuideProbeResult:
    install_dir: str
    install_dir_exists: bool
    python_bits: int
    dlls: dict[str, dict[str, Any]]
    host: str
    ports: dict[int, bool]
    can_load_focas_dll_in_current_python: bool
    notes: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def probe_ncguide(
    install_dir: Path = DEFAULT_NCGUIDE_DIR,
    host: str = DEFAULT_NCGUIDE_HOST,
    ports: list[int] | None = None,
    timeout_seconds: float = 1.0,
) -> NcGuideProbeResult:
    ports = ports or DEFAULT_FOCAS_PORTS
    install_dir = Path(install_dir)
    python_bits = 64 if sys.maxsize > 2**32 else 32
    dlls = {name: inspect_dll(install_dir / name) for name in FOCAS_DLL_NAMES}
    port_results = {port: can_connect_tcp(host, port, timeout_seconds) for port in ports}

    has_usable_dll = any(
        item.get("exists") and item.get("bits") == python_bits
        for item in dlls.values()
    )
    notes = build_probe_notes(
        install_dir_exists=install_dir.exists(),
        python_bits=python_bits,
        dlls=dlls,
        ports=port_results,
    )
    return NcGuideProbeResult(
        install_dir=str(install_dir),
        install_dir_exists=install_dir.exists(),
        python_bits=python_bits,
        dlls=dlls,
        host=host,
        ports=port_results,
        can_load_focas_dll_in_current_python=has_usable_dll,
        notes=notes,
    )


def inspect_dll(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"path": str(path), "exists": False, "bits": None, "machine": None}
    machine = read_pe_machine(path)
    bits = pe_machine_bits(machine)
    return {
        "path": str(path),
        "exists": True,
        "bits": bits,
        "machine": machine,
    }


def read_pe_machine(path: Path) -> str:
    data = path.read_bytes()
    if len(data) < 0x40 or data[:2] != b"MZ":
        return "unknown"
    pe_offset = struct.unpack_from("<I", data, 0x3C)[0]
    if len(data) < pe_offset + 6 or data[pe_offset : pe_offset + 4] != b"PE\0\0":
        return "unknown"
    machine = struct.unpack_from("<H", data, pe_offset + 4)[0]
    return {
        0x014C: "x86",
        0x8664: "x64",
        0x01C0: "arm",
        0xAA64: "arm64",
    }.get(machine, hex(machine))


def pe_machine_bits(machine: str) -> int | None:
    if machine == "x86":
        return 32
    if machine in {"x64", "arm64"}:
        return 64
    return None


def can_connect_tcp(host: str, port: int, timeout_seconds: float) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout_seconds):
            return True
    except OSError:
        return False


def build_probe_notes(
    *,
    install_dir_exists: bool,
    python_bits: int,
    dlls: dict[str, dict[str, Any]],
    ports: dict[int, bool],
) -> list[str]:
    notes: list[str] = []
    if not install_dir_exists:
        notes.append("NCGuide install directory was not found.")
    if not any(item.get("exists") for item in dlls.values()):
        notes.append("No FANUC FOCAS DLL was found in the NCGuide directory.")
    dll_bits = sorted({item.get("bits") for item in dlls.values() if item.get("bits")})
    if dll_bits and python_bits not in dll_bits:
        notes.append(
            f"Current Python is {python_bits}-bit but available FOCAS DLLs are {dll_bits}-bit; "
            "use a matching Python process or a bridge helper."
        )
    if any(ports.values()):
        open_ports = [str(port) for port, ok in ports.items() if ok]
        notes.append(f"NCGuide appears reachable on TCP port(s): {', '.join(open_ports)}.")
    else:
        notes.append("No configured FOCAS TCP port is reachable; start NCGuide or check the port.")
    return notes


class FocasNcGuideClient:
    """Read-only FOCAS client skeleton for FANUC NCGuide.

    Direct DLL access only works when Python and FANUC FOCAS DLL bitness match.
    The installed NCGuide in Program Files (x86) commonly ships 32-bit DLLs, so a
    64-bit Python workflow should call this through a 32-bit bridge process.
    """

    def __init__(
        self,
        install_dir: Path = DEFAULT_NCGUIDE_DIR,
        host: str = DEFAULT_NCGUIDE_HOST,
        port: int = 8193,
        timeout_seconds: int = 10,
    ) -> None:
        self.install_dir = Path(install_dir)
        self.host = host
        self.port = port
        self.timeout_seconds = timeout_seconds
        self._dll: Any | None = None
        self._handle: ctypes.c_ushort | None = None

    def load_library(self) -> None:
        probe = probe_ncguide(self.install_dir, self.host, [self.port])
        if not probe.can_load_focas_dll_in_current_python:
            details = "; ".join(probe.notes)
            raise RuntimeError(f"Cannot load FOCAS DLL in current Python process. {details}")
        dll_path = self.install_dir / "Fwlib32.dll"
        self._dll = ctypes.WinDLL(str(dll_path))

    def connect(self) -> dict[str, Any]:
        if self._dll is None:
            self.load_library()
        assert self._dll is not None
        handle = ctypes.c_ushort()
        result = self._dll.cnc_allclibhndl3(
            self.host.encode("ascii"),
            ctypes.c_ushort(self.port),
            ctypes.c_long(self.timeout_seconds),
            ctypes.byref(handle),
        )
        if result != 0:
            raise RuntimeError(f"cnc_allclibhndl3 failed with FOCAS code {result}")
        self._handle = handle
        return {"status_code": 0, "handle": int(handle.value), "host": self.host, "port": self.port}

    def disconnect(self) -> dict[str, Any]:
        if self._dll is None or self._handle is None:
            return {"status_code": 0, "disconnected": False}
        result = self._dll.cnc_freelibhndl(self._handle)
        self._handle = None
        return {"status_code": int(result), "disconnected": result == 0}


class FocasBridgeClient:
    """Client for a separate 32-bit Python FOCAS bridge process."""

    def __init__(
        self,
        python_exe: Path,
        install_dir: Path = DEFAULT_NCGUIDE_DIR,
        host: str = DEFAULT_NCGUIDE_HOST,
        port: int = 8193,
        timeout_seconds: int = 10,
    ) -> None:
        self.python_exe = Path(python_exe)
        self.install_dir = Path(install_dir)
        self.host = host
        self.port = port
        self.timeout_seconds = timeout_seconds

    def call(self, interface_name: str, parameters: dict[str, Any] | None = None) -> dict[str, Any]:
        action = {
            "Connect": "connect",
            "ReadRunStatus": "read_run_status",
            "ReadPosition": "read_position",
            "ReadFeedSpeed": "read_feed_speed",
            "ReadSpindleSpeed": "read_spindle_speed",
            "ReadAlarm": "read_alarm",
        }.get(interface_name)
        if action is None:
            return {
                "status_code": 400,
                "error": f"Bridge does not support interface: {interface_name}",
            }
        return self.run_bridge(action)

    def run_bridge(self, action: str) -> dict[str, Any]:
        if not self.python_exe.exists():
            return {
                "status_code": 400,
                "error": f"Bridge Python executable not found: {self.python_exe}",
            }
        command = [
            str(self.python_exe),
            str(Path(__file__).with_name("focas_bridge.py")),
            "--install-dir",
            str(self.install_dir),
            "--host",
            self.host,
            "--port",
            str(self.port),
            "--timeout",
            str(self.timeout_seconds),
            "--action",
            action,
        ]
        completed = subprocess.run(
            command,
            cwd=Path(__file__).resolve().parents[2],
            capture_output=True,
            text=True,
            timeout=self.timeout_seconds + 5,
        )
        output = completed.stdout.strip()
        if not output:
            return {
                "status_code": completed.returncode or 500,
                "error": completed.stderr.strip() or "bridge returned no output",
            }
        try:
            result = json.loads(output.splitlines()[-1])
        except json.JSONDecodeError as exc:
            return {
                "status_code": 500,
                "error": f"bridge returned invalid JSON: {exc}",
                "stdout": output,
                "stderr": completed.stderr.strip(),
            }
        if completed.returncode and result.get("status_code") == 0:
            result["status_code"] = completed.returncode
        if completed.stderr.strip():
            result["stderr"] = completed.stderr.strip()
        return result


class FocasCppBridgeClient:
    """Client for the compiled 32-bit C++ FOCAS bridge executable."""

    def __init__(
        self,
        bridge_exe: Path,
        install_dir: Path = DEFAULT_NCGUIDE_DIR,
        host: str = DEFAULT_NCGUIDE_HOST,
        port: int = 8193,
        timeout_seconds: int = 10,
    ) -> None:
        self.bridge_exe = Path(bridge_exe)
        self.install_dir = Path(install_dir)
        self.host = host
        self.port = port
        self.timeout_seconds = timeout_seconds

    def call(self, interface_name: str, parameters: dict[str, Any] | None = None) -> dict[str, Any]:
        action = {
            "Connect": "connect",
            "ReadRunStatus": "read_run_status",
            "ReadPosition": "read_position",
            "ReadFeedSpeed": "read_feed_speed",
            "ReadSpindleSpeed": "read_spindle_speed",
            "ReadAlarm": "read_alarm",
        }.get(interface_name)
        if action is None:
            return {
                "status_code": 400,
                "error": f"C++ bridge does not support interface: {interface_name}",
            }
        return self.run_bridge(action)

    def run_bridge(self, action: str) -> dict[str, Any]:
        if not self.bridge_exe.exists():
            return {
                "status_code": 400,
                "error": f"C++ bridge executable not found: {self.bridge_exe}",
            }
        command = [
            str(self.bridge_exe),
            "--install-dir",
            str(self.install_dir),
            "--host",
            self.host,
            "--port",
            str(self.port),
            "--timeout",
            str(self.timeout_seconds),
            "--action",
            action,
        ]
        completed = subprocess.run(
            command,
            cwd=Path(__file__).resolve().parents[2],
            capture_output=True,
            text=True,
            timeout=self.timeout_seconds + 5,
        )
        output = completed.stdout.strip()
        if not output:
            return {
                "status_code": completed.returncode or 500,
                "error": completed.stderr.strip() or "C++ bridge returned no output",
            }
        try:
            result = json.loads(output.splitlines()[-1])
        except json.JSONDecodeError as exc:
            return {
                "status_code": 500,
                "error": f"C++ bridge returned invalid JSON: {exc}",
                "stdout": output,
                "stderr": completed.stderr.strip(),
            }
        if completed.returncode and result.get("status_code") == 0:
            result["status_code"] = completed.returncode
        if completed.stderr.strip():
            result["stderr"] = completed.stderr.strip()
        return result
