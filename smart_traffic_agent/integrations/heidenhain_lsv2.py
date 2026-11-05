"""Safe, read-only Heidenhain LSV2 connection adapter.

The third-party pyLSV2 package is imported lazily so the rest of the desktop
application remains usable when the optional protocol dependency is absent.
"""
from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Callable, Iterable


class HeidenhainLsv2Error(RuntimeError):
    """Normalized LSV2 adapter error."""


def _plain(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, (bytes, bytearray)):
        return bytes(value).hex()
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if value.__class__.__name__ == "DirectoryEntry":
        return {
            "path": str(getattr(value, "path", "")),
            "free_size": int(getattr(value, "free_size", -1)),
            "dir_attributes": list(getattr(value, "dir_attributes", ()) or ()),
        }
    if value.__class__.__name__ == "VersionInfo":
        return {
            name: _plain(getattr(value, name))
            for name in ("control", "nc_sw", "plc", "splc", "option_bits", "id_number", "release", "type")
            if hasattr(value, name)
        }
    if hasattr(value, "value") and isinstance(value.value, (str, int, float)):
        return value.value
    if hasattr(value, "__dict__"):
        return {str(key): _plain(item) for key, item in vars(value).items() if not str(key).startswith("_")}
    return str(value)


class HeidenhainLsv2Client:
    """Read-only LSV2 client for connection probes and TNC640 collection."""

    def __init__(self, host: str, port: int = 19000, timeout: float = 5.0) -> None:
        self.host = str(host).strip()
        self.port = int(port)
        self.timeout = float(timeout)
        self._connection = None

    def _new_connection(self):
        try:
            import pyLSV2
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise HeidenhainLsv2Error(
                "未安装 pyLSV2，请先执行：python -m pip install -e ."
            ) from exc
        return pyLSV2.LSV2(self.host, port=self.port, timeout=self.timeout, safe_mode=True)

    def read_directory_info(self, remote_directory: str = "TNC:") -> dict[str, Any]:
        """Connect and perform one safe FILETRANSFER read API call."""
        if not self.host:
            raise HeidenhainLsv2Error("海德汉地址不能为空")
        connection = self._new_connection()
        self._connection = connection
        try:
            connection.connect()
            directory = connection.directory_info(remote_directory)
            if directory is False or directory is None:
                detail = _plain(getattr(connection, "last_error", None))
                raise HeidenhainLsv2Error(f"LSV2 directory_info 读取失败：{detail}")
            directory_data = _plain(directory)
            if isinstance(directory_data, dict) and not str(directory_data.get("path", "")).strip():
                detail = _plain(getattr(connection, "last_error", None))
                raise HeidenhainLsv2Error(f"LSV2 directory_info 返回空目录，读取未确认：{detail}")
            return {
                "protocol": "LSV2",
                "operation": "directory_info",
                "host": self.host,
                "port": self.port,
                "directory": directory_data,
                "versions": _plain(connection.versions),
            }
        except HeidenhainLsv2Error:
            raise
        except Exception as exc:
            detail = _plain(getattr(connection, "last_error", None))
            suffix = f"；last_error={detail}" if detail else ""
            raise HeidenhainLsv2Error(f"LSV2 读取目录信息失败：{exc}{suffix}") from exc
        finally:
            try:
                connection.disconnect()
            except Exception:
                pass

    def collect_read_only(
        self,
        operations: Iterable[str],
        *,
        output_dir: str | Path | None = None,
        samples: int = 1,
        interval_seconds: float = 0.0,
        on_record: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        """Collect selected TNC640 operations over one LSV2 session.

        The collector deliberately exposes only read operations.  Every
        selected operation gets its own record, including failures caused by
        an optional login level or a controller option.  This keeps a full
        collection useful when one optional API is unavailable and prevents
        a failed API from being mistaken for a successful call.
        """
        if not self.host:
            raise HeidenhainLsv2Error("海德汉地址不能为空")
        try:
            import pyLSV2
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise HeidenhainLsv2Error(
                "未安装 pyLSV2，请先执行：python -m pip install -e ."
            ) from exc

        try:
            from pyLSV2.const import MemoryType
        except ImportError:  # pragma: no cover - old pyLSV2 compatibility
            MemoryType = None

        requested = [str(name).strip() for name in operations if str(name).strip()]
        repeat = max(1, min(int(samples), 100))
        delay = max(0.0, min(float(interval_seconds), 60.0))
        output_root = Path(output_dir) if output_dir else None
        if output_root:
            output_root.mkdir(parents=True, exist_ok=True)
        blocked = sorted(set(requested) - set(READ_ONLY_OPERATION_CALLS))
        if blocked:
            raise HeidenhainLsv2Error(
                "LSV2 采集只允许只读 API，以下目标被拒绝：" + ", ".join(blocked)
            )
        # pyLSV2's safe_mode also blocks read-only privilege elevation
        # (DNC/PLCDEBUG/DIAGNOSTICS).  The adapter remains safe because its
        # dispatch table contains no write or control methods; disabling that
        # package guard is necessary to collect the documented read APIs.
        connection = pyLSV2.LSV2(self.host, port=self.port, timeout=self.timeout, safe_mode=False)
        self._connection = connection
        records: list[dict[str, Any]] = []
        try:
            connection.connect()
            for operation in requested:
                for sample_index in range(1, repeat + 1):
                    started = datetime.now().astimezone()
                    from ..heidenhain_catalog import heidenhain_operation_metadata
                    library_metadata = heidenhain_operation_metadata(operation)
                    record: dict[str, Any] = {
                        "protocol": "LSV2",
                        "protocol_function": operation,
                        "interface_name": operation,
                        "step_id": f"{operation}_{sample_index:03d}",
                        "sample_index": sample_index,
                        "input_parameters": {
                            "host": self.host,
                            "port": self.port,
                            **READ_ONLY_OPERATION_INPUTS.get(operation, {}),
                        },
                        "started_at": started.isoformat(),
                        "traffic_status": "read_only",
                        **library_metadata,
                    }
                    try:
                        value = _invoke_read_only(connection, operation, MemoryType, output_root)
                        _require_meaningful_read(operation, value)
                        record.update(
                            status_code=0,
                            response={"data": _plain(value), "return_text": "OK"},
                            error="",
                        )
                    except Exception as exc:  # retain the failed target and continue
                        detail = _plain(getattr(connection, "last_error", None))
                        record.update(
                            status_code=-1,
                            response={"data": None, "return_text": "ERROR"},
                            error=f"{type(exc).__name__}: {exc}" + (f"；last_error={detail}" if detail else ""),
                        )
                    record["finished_at"] = datetime.now().astimezone().isoformat()
                    record["success"] = record["status_code"] == 0
                    records.append(record)
                    if on_record:
                        on_record(record)
                    if delay and sample_index < repeat:
                        import time
                        time.sleep(delay)
        except HeidenhainLsv2Error:
            raise
        except Exception as exc:
            detail = _plain(getattr(connection, "last_error", None))
            suffix = f"；last_error={detail}" if detail else ""
            raise HeidenhainLsv2Error(f"LSV2 采集连接失败：{exc}{suffix}") from exc
        finally:
            try:
                connection.disconnect()
            except Exception:
                pass
            self._connection = None

        return {
            "protocol": "LSV2",
            "host": self.host,
            "port": self.port,
            "requested_operations": requested,
            "records": records,
            "successful_operations": sorted({r["protocol_function"] for r in records if r["success"]}),
            "failed_operations": sorted({r["protocol_function"] for r in records if not r["success"]}),
        }


def test_lsv2_directory_info(host: str, port: int = 19000, timeout: float = 5.0) -> dict[str, Any]:
    return HeidenhainLsv2Client(host, port, timeout).read_directory_info()


READ_ONLY_OPERATION_INPUTS: dict[str, dict[str, Any]] = {
    "directory_info": {"remote_directory": "TNC:"},
    "file_info": {"remote_file_path": "TNC:"},
    "get_file_list": {"path": "TNC:", "descend": False, "pattern": ""},
    "read_plc_memory": {"first_element": 0, "memory_type": "MARKER", "number_of_elements": 1},
    "read_plc_address": {"address": "M0"},
    "read_data_path": {"path": ""},
    "get_machine_parameter": {"name": "MP1"},
    "grab_screen_dump": {"local_path": "artifacts/screen_dump.png"},
    "real_time_readings": {"signals": [], "duration": 0, "interval": 600},
}


def _invoke_read_only(connection: Any, operation: str, memory_type: Any, output_root: Path | None) -> Any:
    """Call one pyLSV2 read API with conservative, documented arguments."""
    if operation == "versions":
        return connection.versions
    if operation == "parameters":
        return connection.parameters
    if operation == "program_status":
        return connection.program_status()
    if operation == "program_stack":
        return connection.program_stack()
    if operation == "directory_info":
        return connection.directory_info("TNC:")
    if operation == "directory_content":
        return connection.directory_content()
    if operation == "drive_info":
        return connection.drive_info()
    if operation == "file_info":
        return connection.file_info("TNC:")
    if operation == "get_file_list":
        return connection.get_file_list("TNC:", descend=False, pattern="")
    if operation == "read_plc_memory":
        if memory_type is None:
            raise HeidenhainLsv2Error("当前 pyLSV2 未提供 MemoryType")
        return connection.read_plc_memory(0, memory_type.MARKER, 1)
    if operation == "read_plc_address":
        return connection.read_plc_address("M0")
    if operation == "read_data_path":
        return connection.read_data_path("")
    if operation == "axes_location":
        return connection.axes_location()
    if operation == "override_state":
        return connection.override_state()
    if operation == "spindle_tool_status":
        return connection.spindle_tool_status()
    if operation == "get_remote_datetime":
        return connection.get_remote_datetime()
    if operation == "get_error_messages":
        return connection.get_error_messages()
    if operation == "read_scope_signals":
        return connection.read_scope_signals()
    if operation == "grab_screen_dump":
        if output_root is None:
            raise HeidenhainLsv2Error("屏幕截图需要采集输出目录")
        path = output_root / "artifacts" / "screen_dump.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        return {"path": str(path), "saved": bool(connection.grab_screen_dump(path))}
    if operation == "get_machine_parameter":
        return connection.get_machine_parameter("MP1")
    if operation == "recive_file":
        raise HeidenhainLsv2Error("接收文件需要明确的远程文件和本地路径，当前默认采集不猜测文件")
    if operation == "real_time_readings":
        raise HeidenhainLsv2Error("real_time_readings 仅适用于 iTNC530，TNC640 默认跳过")
    raise HeidenhainLsv2Error(f"未登记的只读 LSV2 API：{operation}")


def _require_meaningful_read(operation: str, value: Any) -> None:
    """Reject permission/unsupported responses that pyLSV2 represents as empty values."""
    if operation in {"axes_location", "override_state", "spindle_tool_status", "read_scope_signals"}:
        if value is None or value == []:
            raise HeidenhainLsv2Error(f"{operation} 返回空数据，未确认控制器支持或权限已授予")
    elif operation in {"read_plc_memory", "read_plc_address"}:
        if value is None or value == []:
            raise HeidenhainLsv2Error(f"{operation} 返回空数据，PLCDEBUG 权限或地址不可用")
    elif operation == "read_data_path" and value is None:
        raise HeidenhainLsv2Error("read_data_path 返回空值，TNC640 不支持该 Data Path 或需要 DATA 权限")
    elif operation == "get_machine_parameter" and value == "":
        raise HeidenhainLsv2Error("get_machine_parameter 返回空值，机床参数名称或 INSPECT 权限不可用")


READ_ONLY_OPERATION_CALLS = {
    "versions", "parameters", "program_status", "program_stack", "directory_info",
    "directory_content", "drive_info", "file_info", "get_file_list", "read_plc_memory",
    "read_plc_address", "read_data_path", "axes_location", "override_state",
    "spindle_tool_status", "get_remote_datetime", "get_error_messages", "read_scope_signals",
    "grab_screen_dump", "recive_file", "get_machine_parameter", "real_time_readings",
}
