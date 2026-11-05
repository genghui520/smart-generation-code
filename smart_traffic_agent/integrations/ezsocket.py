"""Read-only connection and optional OLE/COM adapter for Mitsubishi EZSocket.

The default probe only opens TCP/683 and sends no CNC command.  When the
Mitsubishi OLE/COM runtime and ``pywin32`` are installed, the explicit COM test
path performs the documented ``SetTCPIPProtocol`` → ``Open2`` →
``Status_GetRunStatus`` → ``Close`` sequence without writing machine data.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import socket
from pathlib import Path
from typing import Any


class EzSocketError(ConnectionError):
    """Raised when the Mitsubishi EZSocket endpoint cannot be reached."""


class EzSocketComError(EzSocketError):
    """Raised when the optional Mitsubishi OLE/COM client cannot be used."""


EZSOCKET_PROGID = "EZNcAut.DispEZNcCommunication"
DEFAULT_SYSTEM_TYPE = 9  # EZNC_SYS_MELDAS800M, as used by the SimCNC sample.
DEFAULT_MACHINE = 1
DEFAULT_COM_TIMEOUT = 10  # Open2 timeout units are 100 ms.


def load_ezsocket_manifest(path: str | Path | None = None) -> dict[str, dict[str, Any]]:
    """Load the generated API manifest without importing the embedding stack."""

    manifest_path = Path(path) if path else (
        Path(__file__).resolve().parents[2]
        / "rag_indexes" / "ezsocket" / "function_manifest.jsonl"
    )
    if not manifest_path.is_file():
        return {}
    rows: dict[str, dict[str, Any]] = {}
    try:
        for line in manifest_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                name = str(row.get("function", "")).strip()
                if name:
                    rows[name] = row
    except (OSError, json.JSONDecodeError):
        return {}
    return rows


def test_ezsocket_api_contract(function: str = "Status_GetRunStatus") -> dict[str, Any]:
    """Validate one read-only API against the generated manual/source index."""

    row = load_ezsocket_manifest().get(function)
    if not row:
        raise EzSocketError(f"EZSocket API 知识库中不存在：{function}")
    if not bool(row.get("read_only")):
        raise EzSocketError(f"系统测试只允许只读 EZSocket API：{function}")
    return {
        "function": function,
        "category": row.get("category", ""),
        "read_only": True,
        "source_file": row.get("source_file", ""),
        "contract_verified": True,
    }


def _numeric_values(value: Any) -> list[int]:
    """Return integer-like values from a pywin32 result tuple."""

    values = value if isinstance(value, (tuple, list)) else (value,)
    result: list[int] = []
    for item in values:
        if isinstance(item, bool):
            result.append(int(item))
        elif isinstance(item, int):
            result.append(item)
        elif isinstance(item, float) and item.is_integer():
            result.append(int(item))
    return result


def _com_return_code(value: Any) -> int:
    """Extract the HRESULT-like LONG returned by an automation method.

    Different pywin32 versions expose an automation method with out
    parameters as either ``(return_code, out_value)`` or
    ``(out_value, return_code)``.  A successful zero is unambiguous; for a
    failure we retain the first numeric value so the diagnostic still shows
    the vendor return code.
    """

    values = _numeric_values(value)
    if not values:
        raise EzSocketComError(f"EZSocket COM 返回值不是整数：{value!r}")
    return 0 if 0 in values else values[0]


def _com_out_value(value: Any, *, return_code: int) -> int | None:
    values = _numeric_values(value)
    if len(values) < 2:
        return None
    # Prefer the value that is not the return code.  This handles both output
    # orderings used by dynamic and generated pywin32 dispatch wrappers.
    for item in values:
        if item != return_code:
            return item
    return values[-1]


def _status_result(value: Any) -> tuple[int, int | None]:
    """Decode ``Status_GetRunStatus``'s LONG return and out status.

    The generated pywin32 wrapper and the dynamic wrapper have historically
    exposed these two values in opposite orders.  The status is documented as
    0/1, while a nonzero communication error is normally outside that range;
    use those constraints to disambiguate both forms.
    """

    values = _numeric_values(value)
    if len(values) < 2:
        return _com_return_code(value), None
    first, second = values[0], values[1]
    if first not in (0, 1) and second in (0, 1):
        return first, second
    if second not in (0, 1) and first in (0, 1):
        return second, first
    if first == 0 and second in (0, 1):
        return first, second
    if second == 0 and first in (0, 1):
        return second, first
    # Both values are 1.  Treat the first as the return code so a failed
    # call is never reported as a successful status read.
    return first, second


def _dispatch_ezsocket_com(factory: Any = None, *, machine: str | None = None) -> Any:
    if factory is not None:
        return factory()
    try:
        from win32com.client import Dispatch, DispatchEx
    except ImportError as exc:  # pragma: no cover - depends on host runtime
        raise EzSocketComError(
            "未安装 pywin32，无法创建 Mitsubishi EZSocket COM 对象。"
            "请在运行 COM 测试的 Windows 环境安装 pywin32。"
        ) from exc
    try:
        if machine:
            return DispatchEx(EZSOCKET_PROGID, machine=machine)
        return Dispatch(EZSOCKET_PROGID)
    except Exception as exc:  # pywintypes.com_error is optional here
        code = exc.args[0] if getattr(exc, "args", None) else None
        if code == -2147023174:  # RPC_S_SERVER_UNAVAILABLE
            detail = "RPC 服务器不可用；请确认远程计算机运行库、RPC 服务和防火墙设置"
        elif code == -2147024891:  # E_ACCESSDENIED
            detail = "DCOM 访问被拒绝；请为当前账户配置远程启动/激活权限"
        else:
            detail = "请安装并注册 Mitsubishi EZSocket/FCSB1224W000 运行库"
        raise EzSocketComError(
            f"无法创建 {EZSOCKET_PROGID}：{type(exc).__name__}: {exc}。{detail}。"
        ) from exc


def call_ezsocket_com_api(
    host: str,
    port: int = 683,
    *,
    function: str = "Status_GetRunStatus",
    system_type: int = DEFAULT_SYSTEM_TYPE,
    machine: int = DEFAULT_MACHINE,
    timeout_units: int = DEFAULT_COM_TIMEOUT,
    timeout: float = 5.0,
    com_machine: str | None = None,
    com_factory: Any = None,
) -> dict[str, Any]:
    """Call one read-only EZSocket API through the Mitsubishi OLE interface.

    The COM object owns the EZSocket framing; this adapter never writes a CNC
    value.  ``com_factory`` is intentionally injectable for deterministic unit
    tests and for hosts that wrap the COM object themselves.
    """

    contract = test_ezsocket_api_contract(function)
    if function != "Status_GetRunStatus":
        raise EzSocketComError(
            f"当前 COM 系统测试只实现只读 API Status_GetRunStatus，收到：{function}"
        )
    endpoint = test_ezsocket_connection(host, port, timeout=timeout)
    try:
        system_type = int(system_type)
        machine = int(machine)
        timeout_units = int(timeout_units)
    except (TypeError, ValueError) as exc:
        raise EzSocketComError("COM 连接参数必须是整数") from exc
    if not 1 <= machine <= 255:
        raise EzSocketComError(f"NC 控制单元编号必须在 1–255 之间：{machine}")
    if not 1 <= timeout_units <= 250:
        raise EzSocketComError(f"Open2 超时值必须在 1–250 之间：{timeout_units}")

    com = _dispatch_ezsocket_com(com_factory, machine=(str(com_machine).strip() or None))
    opened = False
    try:
        setup_raw = com.SetTCPIPProtocol(str(host).strip(), int(port))
        setup_code = _com_return_code(setup_raw)
        if setup_code != 0:
            raise EzSocketComError(f"SetTCPIPProtocol 失败，返回码：{setup_code}")

        # For M700/M800, the sample and the reference manual use
        # EZNC_LOCALHOST in Open2 after SetTCPIPProtocol selects the remote
        # TCP target.
        open_raw = com.Open2(system_type, machine, timeout_units, "EZNC_LOCALHOST")
        open_code = _com_return_code(open_raw)
        if open_code != 0:
            raise EzSocketComError(f"Open2 失败，返回码：{open_code}")
        opened = True

        status_raw = com.Status_GetRunStatus(1)
        status_code, status = _status_result(status_raw)
        if status_code != 0:
            raise EzSocketComError(
                f"Status_GetRunStatus(1) 失败，返回码：{status_code}，原始返回：{status_raw!r}"
            )
        if status is None or status not in (0, 1):
            raise EzSocketComError(
                f"Status_GetRunStatus(1) 未返回 0/1 状态，原始返回：{status_raw!r}"
            )
        return {
            "endpoint": endpoint,
            "api": contract,
            "transport": "ezsocket-com",
            "system_type": system_type,
            "machine": machine,
            "status_index": 1,
            "run_status": status,
            "return_code": status_code,
            "com_progid": EZSOCKET_PROGID,
            "com_machine": str(com_machine).strip() or "local",
        }
    except EzSocketComError:
        raise
    except Exception as exc:
        raise EzSocketComError(
            f"EZSocket COM 调用失败：{type(exc).__name__}: {exc}"
        ) from exc
    finally:
        if opened:
            try:
                com.Close()
            except Exception:
                # Preserve the API error; cleanup is best effort because some
                # EZSocket versions release the line when the COM object dies.
                pass


@dataclass(frozen=True)
class EzSocketConnection:
    host: str
    port: int = 683
    timeout: float = 5.0

    def probe(self) -> dict[str, Any]:
        host = str(self.host or "").strip()
        if not host:
            raise EzSocketError("未填写三菱仿真器 IP 地址")
        try:
            port = int(self.port)
        except (TypeError, ValueError) as exc:
            raise EzSocketError(f"EZSocket 端口无效：{self.port!r}") from exc
        if not 1 <= port <= 65535:
            raise EzSocketError(f"EZSocket 端口必须在 1–65535 之间：{port}")
        try:
            timeout = max(0.1, min(float(self.timeout), 30.0))
        except (TypeError, ValueError) as exc:
            raise EzSocketError(f"EZSocket 超时无效：{self.timeout!r}") from exc
        try:
            with socket.create_connection((host, port), timeout=timeout) as connection:
                peer = connection.getpeername()
                local = connection.getsockname()
        except (OSError, ValueError) as exc:
            raise EzSocketError(
                f"无法连接三菱 EZSocket {host}:{port}：{type(exc).__name__}: {exc}"
            ) from exc
        return {
            "protocol": "EZSocket",
            "host": host,
            "port": port,
            "peer": f"{peer[0]}:{peer[1]}",
            "local": f"{local[0]}:{local[1]}",
            "transport": "tcp",
            "read_only": True,
        }


def test_ezsocket_connection(host: str, port: int = 683, *, timeout: float = 5.0) -> dict[str, Any]:
    """Probe a Mitsubishi simulator without sending a command frame."""
    return EzSocketConnection(host, int(port), timeout).probe()
