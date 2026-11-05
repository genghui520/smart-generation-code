"""TNC640 LSV2 operation catalog used by the protocol-specific UI."""
from __future__ import annotations


_ROWS = (
    ("versions", "系统信息", "读取控制器、NC 软件和 PLC 版本信息", "只读"),
    ("parameters", "系统信息", "读取 LSV2 连接参数和能力信息", "只读"),
    ("program_status", "程序状态", "读取当前程序执行状态", "只读"),
    ("program_stack", "程序状态", "读取当前程序调用栈", "只读"),
    ("directory_info", "文件系统", "读取 TNC 目录信息", "只读"),
    ("directory_content", "文件系统", "读取目录内容", "只读"),
    ("drive_info", "文件系统", "读取 TNC/PLC 等驱动器信息", "只读"),
    ("file_info", "文件系统", "读取远程文件信息", "只读"),
    ("get_file_list", "文件系统", "读取远程文件列表", "只读"),
    ("read_plc_memory", "PLC/数据", "读取 PLC 内存区域", "只读"),
    ("read_plc_address", "PLC/数据", "按地址读取 PLC 数据", "只读"),
    ("read_data_path", "PLC/数据", "按 Data Path 读取控制器数据", "只读"),
    ("axes_location", "轴与状态", "读取各轴当前位置", "只读"),
    ("override_state", "轴与状态", "读取进给、主轴和快速倍率", "只读"),
    ("spindle_tool_status", "轴与状态", "读取主轴和刀具状态", "只读"),
    ("get_remote_datetime", "系统信息", "读取控制器远程时间", "只读"),
    ("get_error_messages", "诊断", "读取控制器错误信息", "只读"),
    ("real_time_readings", "实时数据", "读取实时过程数据；pyLSV2 文档注明仅适用于 iTNC530，不作为 TNC640 默认目标", "TNC640不适用"),
    ("read_scope_signals", "实时数据", "读取 Remote Scope 信号", "需选件/权限"),
    ("grab_screen_dump", "诊断", "读取控制器屏幕截图", "需选件/权限"),
    ("change_directory", "文件系统", "切换远程工作目录", "改变会话状态"),
    ("make_directory", "文件系统", "创建远程目录", "写入"),
    ("delete_empty_directory", "文件系统", "删除空远程目录", "写入"),
    ("delete_file", "文件系统", "删除远程文件", "写入"),
    ("copy_remote_file", "文件系统", "复制远程文件", "写入"),
    ("move_file", "文件系统", "移动远程文件", "写入"),
    ("send_file", "文件传输", "向控制器发送文件", "写入"),
    ("recive_file", "文件传输", "从控制器接收文件；需要明确远程文件路径和本地目标路径", "需指定参数"),
    ("get_machine_parameter", "机床参数", "读取机床参数", "只读"),
    ("set_machine_parameter", "机床参数", "写入机床参数", "高风险"),
    ("set_keyboard_access", "远程控制", "设置键盘访问权限", "高风险"),
    ("send_key_code", "远程控制", "发送控制器按键", "高风险"),
)

# The UI/runtime operation names above are pyLSV2 names.  These mappings keep
# the official RemoTools LSV-2 Library symbol and header category alongside
# them so the catalog is grounded in the vendor API rather than the Python
# wrapper's convenience names.
_LIBRARY_MAPPING = {
    "versions": ("LSV2ReceiveVersions", "Basic / system information"),
    "parameters": ("LSV2ReceivePara", "Basic / system information"),
    "program_status": ("LSV2ReceiveRunInfo", "DNC / run information"),
    "program_stack": ("LSV2ReceiveRunInfo", "DNC / run information"),
    "directory_info": ("LSV2ReceiveDirInfo", "File system"),
    "directory_content": ("LSV2ReceiveDir", "File system"),
    "drive_info": ("LSV2ReceiveDrives", "File system"),
    "file_info": ("LSV2ReceiveFileInfo", "File system"),
    "get_file_list": ("LSV2GetFileList", "Backup / file scan"),
    "read_plc_memory": ("LSV2ReceiveMem", "Memory / PLC"),
    "read_plc_address": ("LSV2ReceiveMem", "Memory / PLC"),
    "read_data_path": ("LSV2ReceiveDataEntryInfo", "Data access"),
    "axes_location": ("LSV2ReceiveAxesData", "Axes"),
    "override_state": ("LSV2ReceiveDncInfo", "DNC / run information"),
    "spindle_tool_status": ("LSV2ReceiveDncInfo", "DNC / run information"),
    "get_remote_datetime": ("LSV2ReceiveTime", "Time"),
    "get_error_messages": ("LSV2ReceiveLogFile", "File / log"),
    "read_scope_signals": ("LSV2ReceiveScopeInfo; LSV2ReceiveScopeData", "Scope"),
    "grab_screen_dump": ("LSV2ReceiveScreenDump", "Screen"),
    "recive_file": ("LSV2ReceiveFile", "File system"),
    "get_machine_parameter": ("LSV2ReceiveMachineConstant", "Machine constants"),
    "real_time_readings": ("LSV2ReceiveScopeData", "Scope"),
    "change_directory": ("LSV2ChangeDir", "File system"),
    "make_directory": ("LSV2MakeDir", "File system"),
    "delete_empty_directory": ("LSV2DeleteDir", "File system"),
    "delete_file": ("LSV2DeleteFile", "File system"),
    "copy_remote_file": ("LSV2CopyFile", "File system"),
    "move_file": ("LSV2RenameFile", "File system"),
    "send_file": ("LSV2TransmitFile", "File system"),
    "set_machine_parameter": ("LSV2SetMachineConstant", "Machine constants"),
    "set_keyboard_access": ("LSV2LockKeyboard", "Keyboard / control"),
    "send_key_code": ("LSV2KeyPress", "Keyboard / control"),
}

HEIDENHAIN_READ_ONLY_API_NAMES = frozenset(
    name for name, _category, _description, risk in _ROWS
    if risk in {"只读", "需选件/权限"}
)


def heidenhain_tnc640_catalog() -> list[dict[str, str]]:
    return [
        {
            "function": name,
            "category": category,
            "description": f"TNC640 LSV2：{description}；权限/风险：{risk}",
            "protocol": "lsv2",
            "controller_model": "TNC640",
            "risk": risk,
            "read_only": risk in {"只读", "需选件/权限"},
            "library_function": _LIBRARY_MAPPING.get(name, ("", ""))[0],
            "library_category": _LIBRARY_MAPPING.get(name, ("", ""))[1],
            "library_header": "RemoTools/lsv2lib/includes/lsv2",
        }
        for name, category, description, risk in _ROWS
    ]


def heidenhain_operation_metadata(function: str) -> dict[str, str]:
    library_function, library_category = _LIBRARY_MAPPING.get(function, ("", ""))
    return {
        "library_function": library_function,
        "library_category": library_category,
        "library_header": "RemoTools/lsv2lib/includes/lsv2",
    }
