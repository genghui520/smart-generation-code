from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_reset_status_is_not_treated_as_emergency_stop():
    source = (ROOT / "smart_traffic_agent/templates/ncguide_motion_driver.cpp").read_text(encoding="utf-8")
    assert "status.emergency == 1" in source
    assert "status.alarm || status.emergency)" not in source
    assert "emergency=2 means ReSET" in source


def test_generated_runtime_uses_same_reset_rule():
    source = (ROOT / "smart_traffic_agent/focas_capture_runtime.py").read_text(encoding="utf-8")
    assert "status.emergency == 1" in source


def test_alarm_handling_classifies_known_0id_states_without_blind_reset():
    source = (ROOT / "smart_traffic_agent/templates/ncguide_motion_driver.cpp").read_text(encoding="utf-8")
    assert "(1L << 0)" in source
    assert "SW_PARAMETER_WRITE_SWITCH" in source
    assert "disable_parameter_write_switch_then_reset" in source
    assert "(1L << 1)" in source
    assert "PW_POWER_OFF_PARAMETER_SET" in source
    assert "power_cycle_cnc" in source
    assert source.count("automatic_action=not_attempted") >= 3
    assert "use_alarm_number_and_message_specific_procedure" in source
    assert "cnc_reset(" not in source
    assert "cnc_clralm(" not in source
