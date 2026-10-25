"""Canonical external task descriptions for the RQ1 benchmark.

The scenario strings are the only task source shared by SMPAgent and every
baseline. Baselines additionally receive one common, API-neutral engineering
framework; SMPAgent does not.
"""
from __future__ import annotations

from pathlib import Path


RQ1_SCENARIO_TASKS = {
    "coordinate_motion": (
        "Generate a complete FOCAS client for the Coordinate Motion scenario. "
        "The client must connect to 127.0.0.1:8193, read CNC run status, actual feed, "
        "and absolute X/Y/Z position, and include a documented bounded NC program "
        "with G90, G54, G00, multiple G01 blocks with multiple feed values, and M30."
    ),
    "program_management": (
        "Generate a complete FOCAS client for the Program Management scenario. "
        "The client must connect to 127.0.0.1:8193, safely upload or document a bounded "
        "NC program, select and verify its O-number, read CNC run status and program "
        "number, and include complete return-code handling and cleanup. Use only documented "
        "FANUC FOCAS APIs and a valid NC program payload ending in M30."
    ),
    "tool_management": (
        "Generate a complete FOCAS client for the Tool Management scenario. "
        "The client must connect to 127.0.0.1:8193, read relevant CNC status and tool or "
        "offset information using documented FANUC FOCAS APIs, perform only bounded and "
        "explicitly safe operations, check every return code, and cleanly release the "
        "FOCAS handle. Include a complete documented NC context when required by the API."
    ),
}


RQ1_BASELINE_ENGINEERING_FRAMEWORK = """Generate a complete FANUC FOCAS code solution for the scenario above.

Use the scenario task description exactly as provided. Do not assume or inject any additional FOCAS API names, ABI definitions, structure layouts, Ground Truth operation lists, evaluator rules, or hidden benchmark requirements.

Use the following common engineering setup:

- Language: C++17.
- Compiler: MSVC.
- FANUC SDK: use the official FOCAS SDK available in the benchmark environment.
- Include the official Fwlib32.h. This header is the SDK's authoritative compile-time
  declaration surface: it provides the declarations and type definitions needed to
  write code against the installed FOCAS SDK. It is important because the generated
  source must compile against the actual benchmark SDK rather than guessed interfaces.
- Link with the corresponding official Fwlib32.lib. This import library supplies the
  link-time SDK symbols for the declarations used by the source. It is important because
  successful compilation alone is not sufficient; the final program must also link
  against the same official x86 SDK used by the benchmark build environment.
- Produce a complete compilable C++ source file. This is the implementation artifact
  that contains the client logic, error handling, and resource cleanup and is the
  primary artifact compiled and evaluated by the benchmark.
- Produce an NC program file if required by the scenario. This is the bounded CNC
  program payload/context associated with the client; it is important for scenarios
  whose behavior includes creating, documenting, or supplying an NC program for
  evaluation.
- The target endpoint is 127.0.0.1:8193.
- Do not run NCGuide, a CNC controller, or the generated executable during this code-generation-only benchmark.
- You may use the framework's native planning, coding, review, debugging, and compilation workflow.
- If compilation fails, you may repair the generated code using your normal workflow and compiler diagnostics.
- Do not invent SDK paths, local headers, libraries, API declarations, structure definitions, or undocumented implementation details.
- Use only information available from the scenario description, the local official SDK/build environment, and the framework's own native reasoning/workflow.
- Do not receive or rely on Ground Truth answers during generation.

The final artifacts must be suitable for evaluation in the common benchmark build environment."""


def task_for_scenario(scenario: str) -> str:
    """Return the canonical external task text for one supported RQ1 scenario."""
    try:
        return RQ1_SCENARIO_TASKS[scenario]
    except KeyError as exc:
        raise ValueError(f"Unsupported RQ1 scenario: {scenario}") from exc


def baseline_task_for_scenario(scenario: str, supplied_prompt: Path | None = None) -> str:
    """Resolve the canonical scenario task without allowing a divergent override.

    This helper is retained for callers that need only the task text. Baseline
    runners should use :func:`baseline_prompt_for_scenario` for their final
    external input.
    """
    task = task_for_scenario(scenario)
    if supplied_prompt is not None:
        provided = supplied_prompt.read_text(encoding="utf-8")
        if provided != task:
            raise ValueError(
                "--prompt differs from the canonical SMPAgent RQ1 task text; "
                "use --scenario without --prompt."
            )
    return task


def baseline_prompt_for_scenario(scenario: str, supplied_prompt: Path | None = None) -> str:
    """Build the identical final external prompt for every non-SMPAgent baseline."""
    task = baseline_task_for_scenario(scenario, supplied_prompt)
    return f"[Scenario Task]\n\n{task}\n\n[Common Engineering Framework]\n\n{RQ1_BASELINE_ENGINEERING_FRAMEWORK}"
