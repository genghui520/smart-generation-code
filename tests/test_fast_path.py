from pathlib import Path

from smart_traffic_agent.fast_path import (
    build_fast_path_steps,
    classify_known_api,
    eligible_read_fast_path,
)


SDK_HEADER = Path(r"C:\Lib\FOCAS2 Library\Fwlib\0iD\Fwlib32.h")


def test_fast_path_requires_active_sdk_declaration() -> None:
    facts = classify_known_api("cnc_statinfo", header_path=SDK_HEADER)
    assert facts["status"] == "known"
    missing = classify_known_api("cnc_not_declared", header_path=SDK_HEADER)
    assert missing["status"] == "unknown"


def test_session_and_write_apis_do_not_use_read_fast_path() -> None:
    eligible, facts = eligible_read_fast_path(
        ["cnc_dwnstart4"], header_path=SDK_HEADER
    )
    assert not eligible
    assert facts[0]["status"] == "partially_known"


def test_fast_path_steps_keep_status_target_status_order() -> None:
    steps = build_fast_path_steps("cnc_statinfo")
    assert [step.protocol_function for step in steps] == [
        "cnc_statinfo", "cnc_alarm2", "cnc_statinfo", "cnc_statinfo"
    ]
