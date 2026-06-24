from __future__ import annotations

from pathlib import Path

from ..models import GeneratedArtifacts, WorkflowState
from ..utils import ensure_dir


class CodeGenerationAgent:
    def run(self, state: WorkflowState, output_dir: Path) -> WorkflowState:
        if state.plan is None:
            raise ValueError("Cannot generate code before a plan exists.")

        generated_dir = ensure_dir(output_dir / "generated")
        nc_program = render_nc_program(state.plan.scenario_type)
        api_script = render_api_script(state.plan.steps)

        api_script_path = generated_dir / "api_script.py"
        nc_program_path = generated_dir / "program.nc"
        api_script_path.write_text(api_script, encoding="utf-8")
        nc_program_path.write_text(nc_program, encoding="utf-8")

        diagnostics = validate_generated(api_script, nc_program)
        state.artifacts = GeneratedArtifacts(
            api_script=api_script,
            nc_program=nc_program,
            api_script_path=api_script_path,
            nc_program_path=nc_program_path,
            diagnostics=diagnostics,
        )
        state.stage = "execution"
        return state


def render_nc_program(scenario: str) -> str:
    if scenario == "coordinate_motion":
        return "\n".join(
            [
                "O1001",
                "G90 G54",
                "G01 X0 Y0 Z5 F300",
                "G01 X10 Y0 Z5 F300",
                "G01 X10 Y10 Z3 F240",
                "G01 X0 Y10 Z2 F240",
                "G01 X0 Y0 Z5 F300",
                "M30",
                "",
            ]
        )
    if scenario == "spindle_state":
        return "\n".join(["O2001", "G90 G54", "M03 S800", "G04 P1", "S1200", "G04 P1", "M05", "M30", ""])
    if scenario == "program_lifecycle":
        return "\n".join(["O3001", "G90 G54", "G01 X1 Y1 F120", "G04 P1", "M30", ""])
    return "\n".join(["O9001", "G90 G54", "G04 P1", "M30", ""])


def render_api_script(steps) -> str:
    lines = [
        "from smart_traffic_agent.integrations.simulator_client import SimulatedCncClient",
        "",
        "",
        "def run(client: SimulatedCncClient) -> list[dict]:",
        "    logs = []",
    ]
    for step in steps:
        lines.append(f"    # {step.phase}: {step.action}")
        lines.append(f"    for _ in range({step.repeat}):")
        lines.append(f"        logs.append(client.call({step.interface_name!r}, {step.parameters!r}))")
    lines.extend(["    return logs", ""])
    return "\n".join(lines)


def validate_generated(api_script: str, nc_program: str) -> list[str]:
    diagnostics: list[str] = []
    if "client.call" not in api_script:
        diagnostics.append("api script does not contain client calls")
    if "M30" not in nc_program:
        diagnostics.append("NC program does not end with M30")
    if not nc_program.startswith("O"):
        diagnostics.append("NC program does not start with a program number")
    return diagnostics

