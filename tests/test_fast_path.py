from pathlib import Path

from smart_traffic_agent.fast_path import (
    build_fast_path_steps,
    eligible_session_fast_path,
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


def test_output_only_path_apis_are_deterministic_fast_path_candidates() -> None:
    for name in ("cnc_getpath", "cnc_setpath"):
        facts = classify_known_api(name, header_path=SDK_HEADER)
        assert facts["status"] == "known"


def _download4_trcm() -> dict:
    return {
        "apis": {
            "cnc_download4": {
                "available": True,
                "chain": {
                    "paths": [{
                        "relations": [
                            {
                                "source_api": "cnc_dwnstart4",
                                "target_api": "cnc_download4",
                                "relation_type": "SESSION_SEQUENCE",
                                "confidence_status": "documented",
                            },
                            {
                                "source_api": "cnc_download4",
                                "target_api": "cnc_dwnend4",
                                "relation_type": "SESSION_SEQUENCE",
                                "confidence_status": "documented",
                            },
                        ]
                    }]
                },
                "dependencies": {
                    "dependencies": [{
                        "source_api": "cnc_rdprogdir3",
                        "target_api": "cnc_download4",
                        "relation": {
                            "source_api": "cnc_rdprogdir3",
                            "target_api": "cnc_download4",
                            "relation_type": "PROVIDES_INPUT",
                            "confidence_status": "documented",
                        },
                    }]
                },
                "parameter_rules": {
                    "rules": [{
                        "parameter": "program_number",
                        "source_api": "cnc_rdprogdir3",
                    }]
                },
            }
        }
    }


def test_session_fast_path_requires_trcm_and_active_contract() -> None:
    eligible, facts = eligible_session_fast_path(
        ["cnc_download4"], header_path=SDK_HEADER,
        trcm_context=_download4_trcm(),
    )
    assert eligible
    assert facts["recipe"] == "download4_core"
    assert facts["session_calls"] == ["cnc_dwnstart4", "cnc_download4", "cnc_dwnend4"]


def test_session_fast_path_falls_back_when_directory_relation_is_missing() -> None:
    context = _download4_trcm()
    context["apis"]["cnc_download4"]["parameter_rules"]["rules"] = []
    eligible, facts = eligible_session_fast_path(
        ["cnc_download4"], header_path=SDK_HEADER,
        trcm_context=context,
    )
    assert not eligible
    assert any("program-number provider" in reason for reason in facts["reasons"])


def test_session_fast_path_skips_conditional_discovery_when_inputs_are_verified() -> None:
    from smart_traffic_agent.fast_path import build_session_fast_path_steps

    facts = {
        "session_calls": ["cnc_upstart3", "cnc_upload3", "cnc_upend3"],
        "directory_discovery": {
            "api": "cnc_rdprogdir3",
            "required": False,
            "required_if_missing_inputs": ["s_number", "e_number"],
        },
        "directory_discovery_required": False,
        "directory_discovery_required_if_missing_inputs": ["s_number", "e_number"],
        "verified_inputs": {"s_number": 1234, "e_number": 1234},
    }
    steps = build_session_fast_path_steps("cnc_upload3", facts)
    assert [step.protocol_function for step in steps] == [
        "cnc_upstart3", "cnc_upload3", "cnc_upend3"
    ]
