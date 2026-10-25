"""Desktop protocol capabilities; selectable profiles do not imply live adapters."""
PROTOCOL_PROFILES = {
    "focas": {"label": "FANUC · FOCAS", "available": True, "port": "8193", "target": "FANUC CNC / NCGuide"},
    "ezsocket": {"label": "三菱 · EZSocket", "available": False, "port": "", "target": "三菱数控机床"},
    "s7comm": {"label": "西门子 · S7Comm", "available": False, "port": "", "target": "西门子数控机床"},
    "lsv2": {"label": "海德汉 · LSV/2", "available": False, "port": "", "target": "海德汉数控机床"},
}


def protocol_profile(config):
    key = config.get("protocol", "focas")
    return PROTOCOL_PROFILES.get(key, {"label": str(key), "available": False, "port": "", "target": "设备"})


def require_capture_adapter(config):
    profile = protocol_profile(config)
    if not profile["available"]:
        raise ValueError(f"{profile['label']} 采集适配器尚未接入，当前仅支持配置预览。")
