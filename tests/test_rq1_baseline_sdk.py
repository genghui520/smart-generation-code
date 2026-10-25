from __future__ import annotations

import hashlib
import tempfile
from pathlib import Path

from scripts.baseline_sdk import DEFAULT_HEADER, DEFAULT_LIBRARY, stage_sdk
from scripts.rq1_scenario_tasks import RQ1_BASELINE_ENGINEERING_FRAMEWORK


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_official_sdk_sources_exist_and_are_single_version():
    assert DEFAULT_HEADER.is_file()
    assert DEFAULT_LIBRARY.is_file()
    assert DEFAULT_HEADER.parent.name == "0iD"
    assert _digest(DEFAULT_HEADER) == _digest(DEFAULT_HEADER)
    assert _digest(DEFAULT_LIBRARY) == _digest(DEFAULT_LIBRARY)


def test_stage_contains_only_header_and_library():
    with tempfile.TemporaryDirectory(prefix="rq1_sdk_test_") as raw:
        tmp_path = Path(raw)
        manifest = stage_sdk(tmp_path / "sdk")
        assert (tmp_path / "sdk" / "Fwlib32.h").is_file()
        assert (tmp_path / "sdk" / "Fwlib32.lib").is_file()
        assert manifest["architecture"] == "x86"
        assert manifest["compiler_standard"] == "C++17"
        assert manifest["header_sha256"] == _digest(tmp_path / "sdk" / "Fwlib32.h")
        assert manifest["library_sha256"] == _digest(tmp_path / "sdk" / "Fwlib32.lib")
        assert sorted(p.name for p in (tmp_path / "sdk").iterdir()) == [
            "Fwlib32.h",
            "Fwlib32.lib",
            "sdk_manifest.json",
        ]


def test_framework_exposes_relative_sdk_access_without_benchmark_answers():
    assert "official FOCAS SDK available in the benchmark environment" in RQ1_BASELINE_ENGINEERING_FRAMEWORK
    assert "cnc_" not in RQ1_BASELINE_ENGINEERING_FRAMEWORK
    assert "Ground Truth operation lists" in RQ1_BASELINE_ENGINEERING_FRAMEWORK
