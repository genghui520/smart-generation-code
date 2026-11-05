"""Deterministic retrieval of verified historical FOCAS call examples.

Only runtime records with successful return codes and matched PCAP evidence are
eligible.  Failed compilations, unmatched calls, and partial batches remain
available to the normal error memory but are never presented as success
examples to CodeGenerationAgent.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _rows(path: Path) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                result.append(value)
    except (OSError, UnicodeError):
        pass
    return result


def _successful_batch(collection: dict[str, Any]) -> bool:
    status = collection.get("status_fields", {})
    return (
        collection.get("status") in {"collected", "success"}
        and status.get("workflow_status", "completed") == "completed"
        and status.get("api_execution_status") == "success"
        and status.get("capture_status") == "matched"
        and status.get("collection_status") in {"collected", "success"}
    )


def _snippet(fragment: str, api_name: str) -> str:
    lines = fragment.splitlines()
    selected: list[str] = []
    for index, line in enumerate(lines):
        if api_name in line or "ctx.call" in line:
            selected.extend(lines[max(0, index - 2): min(len(lines), index + 2)])
    # Stable de-duplication keeps the prompt compact while retaining call order.
    seen: set[str] = set()
    return "\n".join(line for line in selected if not (line in seen or seen.add(line)))[:12000]


def _candidate_roots(output_dir: Path | None) -> list[Path]:
    roots: list[Path] = []
    if output_dir is not None:
        if output_dir.name.startswith("batch_"):
            roots.extend([output_dir.parent, output_dir.parent.parent])
        else:
            roots.append(output_dir)
    else:
        roots.extend([Path("runs/desktop"), Path("runs")])
    result: list[Path] = []
    seen: set[Path] = set()
    for root in roots:
        resolved = root.resolve()
        if resolved not in seen and resolved.is_dir():
            seen.add(resolved)
            result.append(resolved)
    return result


def retrieve_verified_api_examples(
    api_name: str,
    *,
    output_dir: Path | None = None,
    limit: int = 3,
    sdk_identity: str = "",
    controller_model: str = "",
) -> list[dict[str, Any]]:
    """Return recent, wire-verified examples for ``api_name``.

    The result is evidence for generation, not an executable plan.  Values are
    retained for comparison, while the active SDK contract remains authoritative.
    """
    target = str(api_name or "").strip()
    if not target:
        return []
    candidates: list[dict[str, Any]] = []
    seen_batches: set[Path] = set()
    for root in _candidate_roots(output_dir):
        for collection_path in root.glob("**/batch_*/execution/collection_result.json"):
            batch = collection_path.parent.parent
            if batch in seen_batches:
                continue
            seen_batches.add(batch)
            collection = _read_json(collection_path)
            if not _successful_batch(collection):
                continue
            wire_targets = set(collection.get("evidence", {}).get("target_apis_with_wire_evidence", []))
            if target not in wire_targets:
                continue
            calls_path = batch / "execution" / "data" / "api_calls.jsonl"
            calls = [row for row in _rows(calls_path)
                     if row.get("api_name") == target
                     and row.get("return_code") == 0
                     and row.get("traffic_status") == "matched"
                     and row.get("transaction_id")]
            if not calls:
                continue
            plan = _read_json(batch / "plan.json")
            record_sdk_identity = str(
                plan.get("sdk_environment_identity")
                or plan.get("sdk_identity")
                or ""
            )
            fragment_path = batch / "generated" / "scenario_fragment.inc"
            if not fragment_path.is_file():
                fragment_path = batch / "execution" / "scenario_fragment.inc"
            try:
                fragment = fragment_path.read_text(encoding="utf-8") if fragment_path.is_file() else ""
            except (OSError, UnicodeError):
                fragment = ""
            order: list[str] = []
            for row in _rows(calls_path):
                name = str(row.get("api_name", ""))
                if name and name not in order:
                    order.append(name)
                if name == target:
                    break
            candidates.append({
                "api_name": target,
                "source_batch": str(batch),
                "controller_model": controller_model or plan.get("target_environment", ""),
                "sdk_identity": record_sdk_identity,
                "verified": True,
                "pcap_matched": True,
                "successful_calls": len(calls),
                "calls": [{
                    "parameters": row.get("input", {}).get("params", ""),
                    "return_code": row.get("return_code"),
                    "return_text": row.get("output", {}).get("return_desc", ""),
                    "output": row.get("output", {}).get("success_data", ""),
                } for row in calls[:8]],
                "preceding_call_order": order,
                "cpp_call_snippet": _snippet(fragment, target),
                "log_format": next((row.get("input", {}).get("params", "") for row in calls), ""),
            })
    candidates.sort(key=lambda item: (
        item.get("controller_model") != controller_model,
        bool(sdk_identity) and item.get("sdk_identity") != sdk_identity,
        item["source_batch"],
    ), reverse=False)
    return candidates[: max(0, int(limit))]
