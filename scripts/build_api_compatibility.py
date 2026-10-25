"""Refresh the offline FOCAS model matrix from the user-specified reference."""
import json
from datetime import datetime, timezone, timedelta
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from smart_traffic_agent.rag.focas_loader import fetch_bytes, parse_xml

SOURCE = "https://www.woody.vip/fanuc/All/flist_All.xml"


def parse_matrix(data):
    functions = {}
    for item in parse_xml(data).findall(".//chapter/item"):
        name = (item.findtext("fname") or "").strip()
        if not name:
            continue
        functions[name] = {cell.tag: (cell.text or "").strip() for cell in item
                           if cell.tag.startswith(("e", "h")) and (cell.text or "").strip() in {"O", "X"}}
    if not functions or not any(functions.values()):
        raise ValueError("Source contains no compatibility matrix; existing snapshot preserved")
    return functions


if __name__ == "__main__":
    functions = parse_matrix(fetch_bytes(SOURCE))
    destination = Path(__file__).resolve().parents[1] / "rag_indexes/focas/api_compatibility.json"
    payload = {"source": SOURCE, "retrieved_on": datetime.now(timezone(timedelta(hours=8))).date().isoformat(),
               "functions": functions}
    temporary = destination.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(destination)
    print(f"Saved {len(functions)} function compatibility records to {destination}")
