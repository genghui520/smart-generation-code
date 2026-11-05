"""Offline model eligibility from the FOCAS function-list matrix.

These are documentation capabilities, not proof that an NCGuide option is enabled.
The desktop capture transport is Ethernet, so use the site's `e` columns.
"""
import json
import re
from functools import lru_cache
from pathlib import Path

UNSELECTED = "未选择型号（显示全部）"
SDK_UNKNOWN = "SDK 无法唯一识别型号（显示全部）"


def detect_sdk_model(directory):
    path = Path(directory) / "Fwlib32.h"
    try:
        stat = path.stat()
        return _detect_sdk_model(str(path), stat.st_mtime_ns, stat.st_size)
    except OSError:
        return SDK_UNKNOWN


@lru_cache(maxsize=16)
def _detect_sdk_model(path, mtime, size):
    text = Path(path).read_text(encoding="latin-1")
    # Match actual definitions, not mentions in conditional branches/comments.
    text = re.sub(r"/\*.*?\*/|//[^\n]*", "", text, flags=re.S)
    definitions = set(re.findall(r"^\s*#\s*define\s+(\w+)\b", text, re.M))
    mapping = {"FS0ID": "0i-A", "FS0IBD": "0i-B/C", "FS0IDD": "0i-D",
               "FS15BD": "15-B", "FS15D": "15i-A/B", "FS16WD": "16i/18i-W"}
    models = {model for macro, model in mapping.items() if macro in definitions}
    # FS30D, FS16D and PMID cover several documented models, not one.
    if definitions & {"FS30D", "FS16D", "PMID", "PMD"}:
        return SDK_UNKNOWN
    return next(iter(models)) if len(models) == 1 else SDK_UNKNOWN
CONTROLLER_MODELS = {
    UNSELECTED: "",
    "0i-A": "0ia", "0i-B/C": "0ib", "0i-D": "0id", "0i-F": "0if",
    "0i-PD": "0ipd", "0i-PF": "0ipf", "15-B": "15b", "15i-A/B": "15iab",
    "16/18/21-B/C": "16bc", "16i/18i/21i-A": "16ia", "16i/18i/21i-B": "16ib",
    "16i/18i-P": "16ip", "16i/18i-L": "16il", "16i/18i-W": "16iw",
    "30i/31i/32i-A": "30ia", "30i/31i/32i-B": "30ib",
    "30i-P": "30ip", "30i-L": "30il", "30i-WA": "30iwa", "30i-WB": "30iwb",
    "Power Mate i-H": "pmih", "Power Mate i-D": "pmid", "Power Motion i-A": "pmia",
}

# F0i-D scope is deliberately derived from the two authoritative local
# sources.  The compatibility matrix contains APIs that are not in the
# current Manifest (for example Ethernet/PMC families), while the Manifest
# also contains APIs for other controller families.  The intersection is the
# only valid target universe for this device in the capture workflow.
F0ID_MODEL = "0i-D"
F0ID_SCOPE_ID = "fanuc:0i-D"
F0ID_COMPATIBILITY_COLUMN = "e0id"


@lru_cache(maxsize=1)
def compatibility_data():
    path = Path(__file__).resolve().parents[1] / "rag_indexes/focas/api_compatibility.json"
    return json.loads(path.read_text(encoding="utf-8"))["functions"]


def _manifest_function_names(manifest_rows=None):
    if manifest_rows is None:
        path = Path(__file__).resolve().parents[1] / "rag_indexes/focas/function_manifest.jsonl"
        try:
            rows = (json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
        except (OSError, json.JSONDecodeError):
            return set()
    else:
        rows = manifest_rows
    return {
        str(row.get("function", "")).strip()
        for row in rows
        if isinstance(row, dict) and str(row.get("function", "")).strip()
    }


@lru_cache(maxsize=1)
def _default_f0id_api_names():
    return _f0id_api_names(None)


def _f0id_api_names(manifest_rows=None):
    """Return the exact F0i-D target set: Manifest functions with e0id=O.

    ``manifest_rows`` is accepted for callers/tests that already loaded a
    custom manifest.  The cached default path is used by production callers.
    Compatibility is never inferred from an API name or from header symbols.
    """
    manifest_names = _manifest_function_names(manifest_rows)
    compatible = {
        str(name).strip()
        for name, row in compatibility_data().items()
        if isinstance(row, dict) and str(row.get(F0ID_COMPATIBILITY_COLUMN, "")).upper() == "O"
    }
    return frozenset(manifest_names.intersection(compatible))


def f0id_api_names(manifest_rows=None):
    return _default_f0id_api_names() if manifest_rows is None else _f0id_api_names(manifest_rows)


def f0id_scope_metadata(manifest_rows=None):
    names = f0id_api_names(manifest_rows)
    return {
        "scope_id": F0ID_SCOPE_ID,
        "device_model": F0ID_MODEL,
        "compatibility_column": F0ID_COMPATIBILITY_COLUMN,
        "source_manifest": "rag_indexes/focas/function_manifest.jsonl",
        "source_compatibility": "rag_indexes/focas/api_compatibility.json",
        "api_count": len(names),
        "api_names": sorted(names),
    }


def unavailable_reason(function, config):
    if config.get("protocol", "focas") != "focas":
        return ""
    model = detect_sdk_model(config["header"]) if "header" in config else config.get("controller_model", UNSELECTED)
    if model in (UNSELECTED, SDK_UNKNOWN):
        return ""
    if model not in CONTROLLER_MODELS:
        return f"未知控制器型号：{model}"
    mark = compatibility_data().get(function, {}).get("e" + CONTROLLER_MODELS[model])
    if mark == "O":
        return ""
    if mark == "X":
        return f"{function} 不适用于 {model} 的 Ethernet 连接，无法选择"
    return f"{function} 在 {model} 的 Ethernet 连接下适用性尚未确认，暂不可选"
