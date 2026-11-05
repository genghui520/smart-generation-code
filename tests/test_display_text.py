from pathlib import Path

from smart_traffic_agent.display_text import (
    format_event,
    human_summary_text,
    localize_technical_text,
    relation_label,
    write_human_summary,
)


def test_event_display_translates_agents_and_keeps_raw_identifiers() -> None:
    line = format_event(
        "agent_start",
        {"agent": "PlanningAgent", "target_api": "cnc_download4", "status": "partially_known"},
    )
    assert "规划智能体" in line
    assert "PlanningAgent" in line
    assert "cnc_download4" in line
    assert "partially_known" in line


def test_technical_display_explains_relation_and_error_without_changing_api_name() -> None:
    line = localize_technical_text("PROVIDES_INPUT cnc_rdprogdir3 EW_BUFFER")
    assert "参数供给" in line
    assert "缓冲区" in line
    assert "cnc_rdprogdir3" in line
    assert "EW_BUFFER" in line


def test_human_summary_is_a_companion_file_and_json_contract_is_untouched(tmp_path: Path) -> None:
    summary = {
        "task_id": "display-test",
        "success": False,
        "stage": "execution",
        "collection_status": "partial",
        "offline_audit_status": "not_run",
        "api_log_count": 2,
        "capture_event_count": 1,
        "plan_steps": 3,
        "repair_attempts": 1,
        "function_coverage": {"captured_function_count": 1, "resolved_function_count": 0, "unavailable_function_count": 0},
        "errors": ["EW_LENGTH from cnc_download4"],
        "relation_types": {"PROVIDES_INPUT": 1},
    }
    path = write_human_summary(tmp_path / "summary_zh.txt", summary)
    text = path.read_text(encoding="utf-8")
    assert "SMPAgent 执行摘要" in text
    assert "部分采集" in text
    assert "参数供给" in text
    assert "EW_LENGTH" in text


def test_relation_label_preserves_database_type() -> None:
    text = relation_label("SESSION_SEQUENCE")
    assert "SESSION_SEQUENCE" in text
    assert "会话顺序" in text
