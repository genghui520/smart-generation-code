from __future__ import annotations

import csv
import json
from pathlib import Path

from smart_traffic_agent.api_log_templates import (
    INPUT_LOG_FIELDS,
    OUTPUT_LOG_FIELDS,
    api_metadata,
    fixed_input_row,
    fixed_output_row,
    write_fixed_logs,
)


def test_sdk_metadata_counts_real_input_and_inout_parameters() -> None:
    statinfo = api_metadata("cnc_statinfo")
    assert statinfo.parameter_count == 1
    assert statinfo.business_input_count == 0
    assert [item.name for item in statinfo.output_parameters] == ["statinfo"]
    assert {"aut", "run", "alarm"}.issubset(statinfo.parameters[1].fields)

    position = api_metadata("cnc_rdposition")
    assert "fields" in position.parameters[-1].fields["abs"]
    assert "data" in position.parameters[-1].fields["abs"]["fields"]

    directory = api_metadata("cnc_rdprogdir3")
    assert directory.parameter_count == 4
    assert [item.direction for item in directory.parameters] == ["in", "in", "inout", "inout", "out"]

    path = api_metadata("cnc_getpath")
    assert path.parameter_count == 1
    assert [item.direction for item in path.parameters] == ["in", "out", "out"]

    download = api_metadata("cnc_download4")
    assert download.parameters[1].direction == "inout"
    assert download.parameters[2].valid_length_rule == "runtime_bound_by=length"


def test_cnc_alarm2_uses_the_active_header_contract() -> None:
    alarm = api_metadata("cnc_alarm2")

    assert alarm.complete
    assert [item.name for item in alarm.parameters] == ["FlibHndl", "alarm"]
    assert [item.type for item in alarm.parameters] == ["unsigned short", "long *"]
    assert [item.direction for item in alarm.parameters] == ["in", "out"]


def test_api_metadata_reports_sdk_header_arity_mismatch(monkeypatch) -> None:
    import smart_traffic_agent.api_log_templates as templates
    from types import SimpleNamespace

    templates.api_metadata.cache_clear()
    monkeypatch.setattr(
        templates,
        "named_prototype",
        lambda _name: "FWLIBAPI short WINAPI cnc_alarm2(unsigned short FlibHndl, long *alarm);",
    )
    monkeypatch.setattr(
        templates,
        "_sdk_index",
        lambda: SimpleNamespace(
            functions={"cnc_alarm2": "FWLIBAPI short WINAPI cnc_alarm2(unsigned short, long *, short);"},
            structs={},
            unions={},
        ),
    )
    metadata = templates.api_metadata("cnc_alarm2")
    try:
        assert not metadata.complete
        assert any("SDK_PROTOTYPE_MISMATCH" in item for item in metadata.diagnostics)
        assert any("documented_parameters=2" in item for item in metadata.diagnostics)
        assert any("sdk_header_parameters=3" in item for item in metadata.diagnostics)
    finally:
        monkeypatch.undo()
        templates.api_metadata.cache_clear()


def test_api_metadata_reports_sdk_header_type_mismatch(monkeypatch) -> None:
    import smart_traffic_agent.api_log_templates as templates
    from types import SimpleNamespace

    templates.api_metadata.cache_clear()
    monkeypatch.setattr(
        templates,
        "named_prototype",
        lambda _name: "FWLIBAPI short WINAPI cnc_alarm2(unsigned short FlibHndl, long *alarm);",
    )
    monkeypatch.setattr(
        templates,
        "_sdk_index",
        lambda: SimpleNamespace(
            functions={"cnc_alarm2": "FWLIBAPI short WINAPI cnc_alarm2(unsigned short, short *);"},
            structs={},
            unions={},
        ),
    )
    metadata = templates.api_metadata("cnc_alarm2")
    try:
        assert not metadata.complete
        assert any("documented_types" in item and "sdk_header_types" in item for item in metadata.diagnostics)
    finally:
        monkeypatch.undo()
        templates.api_metadata.cache_clear()


def test_fixed_rows_keep_handle_and_only_emit_success_output() -> None:
    input_row = fixed_input_row({
        "timestamp": "2026-10-28T12:00:00Z",
        "protocol_function": "cnc_rdmacro",
        "parameters": "FlibHndl=32769;number=100;length=10",
    })
    assert "api_parameter_count" not in input_row
    assert input_row["params"].startswith("FlibHndl=32769")

    success = fixed_output_row({
        "timestamp": "2026-10-28T12:00:01Z",
        "api_name": "cnc_rdmacro",
        "return_code": "0",
        "return_text": "EW_OK",
        "data": "datano=100;mcr_val=12;dec_val=0",
    })
    assert success["return_desc"] == "EW_OK"
    assert "mcr_val=12" in success["success_data"]
    assert "macro.mcr_val" not in success["success_data"]
    alarm = fixed_output_row({
        "api_name": "cnc_rdalminfo",
        "return_code": 0,
        "data": "alarminfo.alm1.alm[0].axis=-1;alarminfo.alm1.data_end=0",
    })
    assert "alm[0].axis=-1" in alarm["success_data"]
    assert "alarminfo." not in alarm["success_data"]
    assert fixed_output_row({"api_name": "cnc_statinfo", "return_code": 0, "return_text": "EW_OK", "data": "run=0"})["return_code"] == 0

    failure = fixed_output_row({
        "timestamp": "2026-10-28T12:00:02Z",
        "api_name": "cnc_upstart3",
        "return_code": "5",
        "return_text": "EW_DATA",
        "data": "output_valid=false",
    })
    assert failure["return_desc"] == "EW_DATA"
    assert failure["success_data"] == ""


def test_timer_union_logs_keep_only_selected_branch() -> None:
    date_row = fixed_output_row({
        "api_name": "cnc_gettimer",
        "return_code": 0,
        "return_text": "EW_OK",
        "data": (
            "buf.type=0;buf.data.date.year=2026;buf.data.date.month=10;"
            "buf.data.date.date=30;buf.data.time.hour=2026;"
            "buf.data.time.minute=10;buf.data.time.second=30"
        ),
    })
    assert "data.date.year=2026" in date_row["success_data"]
    assert "data.time.hour" not in date_row["success_data"]

    time_row = fixed_output_row({
        "api_name": "cnc_gettimer",
        "return_code": 0,
        "return_text": "EW_OK",
        "data": (
            "buf.type=1;buf.data.date.year=2026;buf.data.date.month=10;"
            "buf.data.date.date=30;buf.data.time.hour=12;"
            "buf.data.time.minute=34;buf.data.time.second=56"
        ),
    })
    assert "data.time.hour=12" in time_row["success_data"]
    assert "data.date.year" not in time_row["success_data"]


def test_fixed_logs_have_stable_columns_and_invocation_alignment(tmp_path: Path) -> None:
    inputs = [
        {
            "index": "1", "timestamp": "t1", "step_id": "S1",
            "protocol_function": "cnc_statinfo", "parameters": "FlibHndl=1",
            "invocation_id": "inv-1",
        },
        {
            "index": "2", "timestamp": "t2", "step_id": "S2",
            "protocol_function": "cnc_statinfo", "parameters": "FlibHndl=1",
            "invocation_id": "inv-2",
        },
    ]
    outputs = [
        {
            "index": "1", "timestamp": "u1", "step_id": "S1",
            "api_name": "cnc_statinfo", "return_code": "0", "return_text": "EW_OK",
            "data": "aut=3;run=0", "invocation_id": "inv-1",
        },
        {
            "index": "2", "timestamp": "u2", "step_id": "S2",
            "api_name": "cnc_statinfo", "return_code": "5", "return_text": "EW_DATA",
            "data": "output_valid=false", "invocation_id": "inv-2",
        },
    ]
    input_path, output_path, calls_path = write_fixed_logs(tmp_path, inputs, outputs)
    with input_path.open(encoding="utf-8-sig", newline="") as stream:
        input_rows = list(csv.DictReader(stream))
        assert stream.seekable()
    with output_path.open(encoding="utf-8-sig", newline="") as stream:
        output_rows = list(csv.DictReader(stream))
    assert list(input_rows[0]) == INPUT_LOG_FIELDS
    assert list(output_rows[0]) == OUTPUT_LOG_FIELDS
    assert len(input_rows) == len(output_rows) == 2
    assert output_rows[1]["success_data"] == ""
    records = [json.loads(line) for line in calls_path.read_text(encoding="utf-8").splitlines()]
    assert [item["invocation_id"] for item in records] == ["inv-1", "inv-2"]
    assert records[0]["input"]["values"]["FlibHndl"] == 1
    assert records[1]["output"]["return_desc"] == "EW_DATA"


def test_missing_return_row_is_explicitly_unavailable(tmp_path: Path) -> None:
    write_fixed_logs(
        tmp_path,
        [{"index": "1", "timestamp": "t", "step_id": "S", "protocol_function": "cnc_statinfo", "parameters": "FlibHndl=1", "invocation_id": "i1"}],
        [],
    )
    with (tmp_path / "output_log.csv").open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert rows[0]["return_code"] == ""
    assert rows[0]["return_desc"] == "UNAVAILABLE"
    assert rows[0]["success_data"] == ""
