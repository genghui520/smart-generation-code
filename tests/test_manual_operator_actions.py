from __future__ import annotations

from pathlib import Path

from smart_traffic_agent.desktop_backend import manual_operator_action_timed_out


def test_manual_action_timeout_is_detected_from_runner_log(tmp_path: Path) -> None:
    execution = tmp_path / "execution"
    execution.mkdir()
    stdout_path = execution / "api_script_stdout.txt"

    stdout_path.write_text("@@SMP@@{\"event\":\"operator_action_timeout\"}\n", encoding="utf-8")
    assert not manual_operator_action_timed_out(tmp_path)

    stdout_path.write_text("operator_action -> ret=12 MANUAL_ACTION_TIMEOUT", encoding="utf-8")
    assert manual_operator_action_timed_out(tmp_path)

    stdout_path.write_text("operator_action -> ret=12 MANUAL_ACTION_DEFERRED", encoding="utf-8")
    assert manual_operator_action_timed_out(tmp_path)
