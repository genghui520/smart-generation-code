"""Desktop protocol capabilities and connection/capture availability."""
PROTOCOL_PROFILES = {
    "focas": {"label": "FANUC · FOCAS", "available": True, "connection_testable": True, "capture_available": True, "port": "8192", "target": "FANUC CNC / NCGuide"},
    # Mitsubishi CNC simulators expose the EZSocket endpoint on TCP/683 by
    # default.  The protocol is connection-testable now; packet/API capture
    # remains disabled until the command catalog is configured for this
    # controller family.
    "ezsocket": {
        "label": "三菱 · EZSocket",
        "available": False,
        "connection_testable": True,
        "capture_available": False,
        "host": "192.168.139.130",
        "port": "683",
        "target": "三菱数控仿真器 · EZSocket",
        "knowledge_collection": "ezsocket_knowledge",
    },
    "s7comm": {"label": "西门子 · S7Comm", "available": False, "connection_testable": False, "capture_available": False, "port": "", "target": "西门子数控机床"},
    "lsv2": {"label": "海德汉 · LSV/2", "available": True, "connection_testable": True, "capture_available": True, "host": "192.168.56.102", "port": "19000", "samples": "1", "interval": "0.2", "target": "海德汉 TNC640", "knowledge_collection": "heidenhain_lsv2_knowledge"},
}


def protocol_profile(config):
    key = config.get("protocol", "focas")
    return PROTOCOL_PROFILES.get(key, {"label": str(key), "available": False, "port": "", "target": "设备"})


def require_capture_adapter(config):
    profile = protocol_profile(config)
    if not profile.get("capture_available", profile.get("available", False)):
        raise ValueError(f"{profile['label']} 采集适配器尚未接入，当前仅支持连接测试。")


def require_connection_adapter(config):
    profile = protocol_profile(config)
    if not profile.get("connection_testable", profile.get("available", False)):
        raise ValueError(f"{profile['label']} 连接适配器尚未接入。")
