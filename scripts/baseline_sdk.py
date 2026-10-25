"""Isolated, shared FANUC FOCAS SDK resources for RQ1 baselines.

This module stages only the official header and import library. It deliberately
does not copy project examples, generated artifacts, DLLs, RAG indexes, or
Ground Truth files into a baseline workspace.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path


DEFAULT_HEADER = Path(r"C:\Lib\FOCAS2 Library\Fwlib\0iD\Fwlib32.h")
DEFAULT_LIBRARY = Path(r"C:\Lib\FOCAS2 Library\Fwlib\Fwlib32.lib")


def sdk_source_paths() -> tuple[Path, Path]:
    root = os.environ.get("RQ1_BASELINE_SDK_SOURCE", "").strip()
    if root:
        base = Path(root)
        return base / "Fwlib32.h", base / "Fwlib32.lib"
    return DEFAULT_HEADER, DEFAULT_LIBRARY


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stage_sdk(destination: Path) -> dict[str, object]:
    """Copy the same official header/library into an isolated baseline workspace."""
    header, library = sdk_source_paths()
    missing = [str(path) for path in (header, library) if not path.is_file()]
    if missing:
        raise FileNotFoundError("Official FOCAS SDK resource missing: " + ", ".join(missing))
    destination.mkdir(parents=True, exist_ok=True)
    staged_header = destination / "Fwlib32.h"
    staged_library = destination / "Fwlib32.lib"
    shutil.copy2(header, staged_header)
    shutil.copy2(library, staged_library)
    manifest = {
        "source_header": str(header),
        "source_library": str(library),
        "header": str(staged_header),
        "library": str(staged_library),
        "header_sha256": sha256(staged_header),
        "library_sha256": sha256(staged_library),
        "architecture": "x86",
        "compiler_standard": "C++17",
        "compiler": "MSVC",
        "resources_only": ["Fwlib32.h", "Fwlib32.lib"],
    }
    (destination / "sdk_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest
