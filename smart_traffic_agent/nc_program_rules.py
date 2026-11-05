from __future__ import annotations

from typing import Any

from .models import ExecutionPlan, NcProgramSpec, PlanStep


def extract_nc_program_requirement(prerequisites: Any) -> str | None:
    """Return the first explicit NC-program requirement from a node contract."""
    if isinstance(prerequisites, str):
        try:
            import json
            prerequisites = json.loads(prerequisites)
        except (TypeError, ValueError):
            return None
    if not isinstance(prerequisites, dict):
        return None
    entries = prerequisites.get("nc_program", [])
    if isinstance(entries, dict):
        entries = [entries]
    if not isinstance(entries, list):
        return None
    for entry in entries:
        if isinstance(entry, dict):
            value = str(entry.get("requirement", "")).strip().lower()
            if value:
                return value
    return None


def extract_nc_program_requirements(prerequisites: Any) -> list[str]:
    """Return declarative NC payload requirements from the API node contract."""
    if isinstance(prerequisites, str):
        try:
            import json
            prerequisites = json.loads(prerequisites)
        except (TypeError, ValueError):
            return []
    if not isinstance(prerequisites, dict):
        return []
    entries = prerequisites.get("nc_program", [])
    if isinstance(entries, dict):
        entries = [entries]
    if not isinstance(entries, list):
        return []
    requirements: list[str] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        values = entry.get("requirements", [])
        if isinstance(values, str):
            values = [values]
        if not isinstance(values, list):
            continue
        for value in values:
            text = str(value).strip()
            if text and text not in requirements:
                requirements.append(text)
    return requirements


def has_running_nc_program_requirement(prerequisites: Any) -> bool:
    return extract_nc_program_requirement(prerequisites) == "running"


def axis_domain_from_count(axis_count: int) -> list[int]:
    """Build the only legal axis domain from a successful runtime count."""
    count = int(axis_count)
    if count < 1:
        raise ValueError("cnc_sysinfo returned no controlled axes")
    return list(range(1, count + 1))


def build_accdecdly_running_steps() -> list[PlanStep]:
    """Build the deterministic running-program sequence for cnc_accdecdly."""
    axis_binding = {
        "axis_source": "cnc_sysinfo.sysinfo.axes",
        "axis_domain": "1..N",
        "axis_count_source": "runtime_output",
        "length_rule": "axis=-1: 4 + 4 * MAX_AXIS; axis=1..N: 8",
        "sampling_output": "accdecdly.data[]",
    }
    return [
        PlanStep("NC-RUN-001", "before", "upload NC motion program", "UploadProgram", {}, protocol_function="cnc_dwnstart3/cnc_download3/cnc_dwnend3"),
        PlanStep("NC-RUN-002", "before", "select NC motion program", "SelectProgram", {}, protocol_function="cnc_search"),
        PlanStep("NC-RUN-003", "before", "verify selected program", "ReadProgramNumber", {}, protocol_function="cnc_rdprgnum"),
        PlanStep(
            "NC-RUN-004", "before", "record baseline CNC run status", "ReadRunStatus", {},
            expected_state="program is not running before Cycle Start", protocol_function="cnc_statinfo",
        ),
        PlanStep(
            "NC-RUN-005", "before", "read controlled axis count", "ReadSystemInfo", {},
            protocol_function="cnc_sysinfo",
            outputs=[{"path": "sysinfo.axes", "binds": "axis_domain"}],
        ),
        PlanStep(
            "NC-RUN-006", "before", "record baseline acceleration/deceleration delay", "ReadAccelerationDecelerationDelay",
            dict(axis_binding), protocol_function="cnc_accdecdly", role="target",
        ),
        PlanStep(
            "NC-RUN-007", "during", "start NC motion program", "StartProgram", {},
            operation_kind="ncguide_ui",
            expected_state="Cycle Start acknowledged and real CNC motion is observed",
            protocol_function="ncguide_ui_cycle_start",
        ),
        PlanStep(
            "NC-RUN-008", "during", "confirm NC program is running", "ReadRunStatus", {},
            expected_state="cnc_statinfo.run indicates active execution; do not replace with a delay", protocol_function="cnc_statinfo",
        ),
        PlanStep(
            "NC-RUN-009", "during", "sample acceleration/deceleration delay for every legal axis", "ReadAccelerationDecelerationDelay",
            dict(axis_binding), repeat=4, interval_seconds=0.2, expected_state="real accdecdly.data[] may vary during motion",
            protocol_function="cnc_accdecdly", role="target",
        ),
        PlanStep(
            "NC-RUN-010", "after", "confirm NC program completion", "ReadRunStatus", {},
            expected_state="program completion gate observed from cnc_statinfo", protocol_function="cnc_statinfo",
        ),
        PlanStep(
            "NC-RUN-011", "after", "sample final acceleration/deceleration delay for every legal axis", "ReadAccelerationDecelerationDelay",
            dict(axis_binding), protocol_function="cnc_accdecdly", role="target",
        ),
    ]


def _node_prerequisites(facts: dict[str, Any]) -> Any:
    api_info = facts.get("api_info", {})
    if isinstance(api_info, dict):
        node = api_info.get("api", {})
        if isinstance(node, dict) and "prerequisites" in node:
            return node.get("prerequisites")
    node = facts.get("node_contract", {})
    return node.get("prerequisites", {}) if isinstance(node, dict) else {}


def apply_running_nc_program_rule(
    plan: ExecutionPlan,
    trcm_context: dict[str, Any],
    requested_targets: list[str] | tuple[str, ...],
) -> bool:
    """Apply the graph's explicit ``nc_program=requirement=running`` rule."""
    targets = {str(item).strip() for item in requested_targets if str(item).strip()}
    if "cnc_accdecdly" not in targets:
        return False
    facts = (trcm_context.get("apis", {}) or {}).get("cnc_accdecdly", {})
    prerequisites = _node_prerequisites(facts) if isinstance(facts, dict) else {}
    if not isinstance(facts, dict) or not has_running_nc_program_requirement(prerequisites):
        return False
    graph_requirements = extract_nc_program_requirements(prerequisites)
    plan.target_api = "cnc_accdecdly"
    plan.scenario_type = "coordinate_motion"
    plan.scenario = "coordinate_motion"
    plan.scenario_goal = "Capture cnc_accdecdly before, during, and after a verified running NC motion program."
    plan.nc_program_type = "required_for_running_state"
    plan.nc_program_requirements = [
        "NC program must be uploaded and selected before Cycle Start",
        "cnc_statinfo must confirm real running state before dynamic sampling",
        "cnc_sysinfo.sysinfo.axes is the sole source for axis domain 1..N",
        "sample every legal axis before, during, and after motion",
    ]
    plan.nc_program_spec = NcProgramSpec(
        program_name="",
        purpose="Run an existing axis-motion template and capture cnc_accdecdly while the NC program is demonstrably running.",
        block_goals=[
            "perform bounded X-axis motion",
            "perform bounded XY motion with distinct feed phases",
            "return to a safe point and end with M30",
        ],
        constraints=[
            "validate NCGuide execution permission and travel bounds before Cycle Start",
            "do not equate program existence with running state",
            "confirm running state from cnc_statinfo output, never a fixed delay",
            *graph_requirements,
        ],
        generation_notes=[
            "Reuse the coordinate-motion template; choose distinct moderate feeds and distances within verified bounds.",
            "Only successful cnc_sysinfo output may materialize axis numbers.",
            *graph_requirements,
        ],
    )
    plan.steps = build_accdecdly_running_steps()
    plan.rag_context["running_nc_program"] = {
        "requirement": "running",
        "target_api": "cnc_accdecdly",
        "axis_source": "cnc_sysinfo.axes",
        "axis_count_source": "runtime_output",
        "axis_domain": "1..N",
        "length_rule": "axis=-1: 4 + 4 * MAX_AXIS; axis=1..N: 8",
        "status_gate": "cnc_statinfo.run observed after Cycle Start",
        "variation_policy": "mark observed only when real accdecdly.data[] values differ; retain logs and PCAP otherwise",
        "nc_program_requirements": graph_requirements,
    }
    plan.nc_program_requirements = list(dict.fromkeys([*plan.nc_program_requirements, *graph_requirements]))
    plan.rag_context["explicit_target_functions"] = ["cnc_accdecdly"]
    return True
