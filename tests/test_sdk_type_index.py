from __future__ import annotations

import json
from pathlib import Path

import pytest
from unittest.mock import patch

from smart_traffic_agent.sdk_type_index import (
    build_sdk_type_index,
    load_or_build_sdk_type_index,
    sdk_environment_identity,
    validate_struct_member_access,
)


SDK_HEADER = Path(r"C:\Lib\FOCAS2 Library\Fwlib\0iD\Fwlib32.h")


@pytest.mark.parametrize("field", ["manual", "write", "labelskip", "warning", "battery"])
def test_rejects_nonexistent_odbst_members(field: str) -> None:
    index = build_sdk_type_index(SDK_HEADER)
    errors = validate_struct_member_access(
        f"ODBST status{{}};\nuse(status.{field});", index
    )
    assert len(errors) == 1
    assert "SDK_TYPE_MISMATCH" in errors[0]
    assert "struct=ODBST" in errors[0]
    assert f"field={field}" in errors[0]
    assert "scenario_fragment.inc:2" in errors[0]
    assert "valid_members=" in errors[0]
    assert "aut" in errors[0]


def test_accepts_active_sdk_members_and_arrays() -> None:
    index = build_sdk_type_index(SDK_HEADER)
    source = (
        "const ODBST& status = initial; ODBPOS positions[3]{}; "
        "ODBACT* feed = &value; "
        "use(status.aut, status.run, status.motion, positions[0].abs.data, feed->data);"
    )
    assert validate_struct_member_access(source, index) == []
    assert index.structs["ODBACT"]["fields"]["dummy"]["array"] == "2"
    assert index.functions["cnc_statinfo"].startswith("FWLIBAPI short WINAPI cnc_statinfo(")


def test_other_sdk_type_uses_same_validation(tmp_path: Path) -> None:
    header = tmp_path / "Fwlib32.h"
    header.write_text("typedef struct example { short real; long data[4][8]; } EXAMPLE;", encoding="ascii")
    index = build_sdk_type_index(header)
    assert index.structs["EXAMPLE"]["fields"]["data"]["array_dimensions"] == ["4", "8"]
    errors = validate_struct_member_access("EXAMPLE value{}; value.fake = 1;", index)
    assert len(errors) == 1
    assert "struct=EXAMPLE field=fake" in errors[0]
    assert "real" in errors[0] and "data" in errors[0]


def test_index_rebuilds_when_header_changes(tmp_path: Path) -> None:
    header = tmp_path / "Fwlib32.h"
    cache = tmp_path / "sdk_index.json"
    header.write_text("typedef struct sample { short first; } SAMPLE;", encoding="ascii")
    first = load_or_build_sdk_type_index(header, cache)
    header.write_text("typedef struct sample { short second; } SAMPLE;", encoding="ascii")
    second = load_or_build_sdk_type_index(header, cache)
    assert first.header_sha256 != second.header_sha256
    assert "second" in second.structs["SAMPLE"]["fields"]
    assert "first" not in second.structs["SAMPLE"]["fields"]
    assert json.loads(cache.read_text(encoding="utf-8"))["schema_version"] == 5


def test_uses_active_conditional_definition(tmp_path: Path) -> None:
    header = tmp_path / "Fwlib32.h"
    header.write_text(
        "#if defined(PMD)\ntypedef struct sample { short pmd; } SAMPLE;\n"
        "#else\ntypedef struct sample { short normal; } SAMPLE;\n#endif\n",
        encoding="ascii",
    )
    index = build_sdk_type_index(header)
    assert set(index.structs["SAMPLE"]["fields"]) == {"normal"}


def test_validator_blocks_type_error_before_production_compile(tmp_path: Path) -> None:
    from smart_traffic_agent.agent_tools.validate_focas_cpp import (
        ValidateFocasCppInput, ValidateFocasCppTool,
    )

    shell = tmp_path / "api_script.cpp"
    fragment = tmp_path / "scenario_fragment.inc"
    shell.write_text("#include <Fwlib32.h>\n", encoding="utf-8")
    fragment.write_text("void read() { ODBST st{}; use(st.battery); }", encoding="utf-8")
    with patch("smart_traffic_agent.focas_capture_runtime.validate_production_source", return_value=[]):
        result = ValidateFocasCppTool().invoke(ValidateFocasCppInput(
            cpp_path=shell, fragment_path=fragment, production_contract=object(),
        ))
    assert not result.success
    assert any("SDK_TYPE_MISMATCH" in error and "field=battery" in error for error in result.errors)


def test_real_msvc_accepts_legal_sdk_fields(tmp_path: Path) -> None:
    from smart_traffic_agent.agent_tools.cpp_execution import (
        CompileCppInput, CompileGeneratedCppTool, find_msvc_x86_toolchain,
    )

    if find_msvc_x86_toolchain() is None:
        pytest.skip("MSVC x86 toolchain unavailable")
    source = tmp_path / "legal_sdk_fields.cpp"
    source.write_text(
        "#include <Fwlib32.h>\n"
        "int main() { ODBST st{}; ODBPOS pos{}; ODBACT feed{}; "
        "return st.aut + st.run + st.motion + pos.abs.data + feed.data; }\n",
        encoding="ascii",
    )
    output = CompileGeneratedCppTool().invoke(CompileCppInput(
        source_path=source, executable_path=tmp_path / "legal_sdk_fields.exe", work_dir=tmp_path,
    ))
    assert output.success, output.stdout + output.stderr


def test_environment_identity_records_actual_files(tmp_path: Path) -> None:
    identity = sdk_environment_identity(SDK_HEADER, tmp_path / "missing.dll")
    assert identity["header"]["exists"]
    assert identity["header"]["sha256"]
    assert identity["runtime_dll"]["exists"] is False
    assert identity["abi_release_match"] == "unverified"
