from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

from .base import AgentTool


@dataclass(frozen=True, slots=True)
class RenderFocasCppInput:
    harness_path: Path
    fragment_path: Path
    output_cpp_path: Path


@dataclass(slots=True)
class RenderFocasCppOutput:
    output_cpp_path: Path
    harness_path: Path
    fragment_path: Path
    success: bool
    error: str | None = None
    harness_sha256: str | None = None


class RenderFocasCppTool(AgentTool[RenderFocasCppInput, RenderFocasCppOutput]):
    name = "render_focas_cpp"
    description = "Assemble the verified fixed FOCAS harness and generated scenario fragment into the final C++ source."

    def invoke(self, tool_input: RenderFocasCppInput) -> RenderFocasCppOutput:
        harness_path = tool_input.harness_path
        fragment_path = tool_input.fragment_path
        output_cpp_path = tool_input.output_cpp_path
        if not harness_path.exists():
            return RenderFocasCppOutput(output_cpp_path, harness_path, fragment_path, False, "Harness file does not exist.")
        if not fragment_path.exists():
            return RenderFocasCppOutput(output_cpp_path, harness_path, fragment_path, False, "Scenario fragment file does not exist.")
        fragment = fragment_path.read_text(encoding="utf-8")
        if not fragment.strip():
            return RenderFocasCppOutput(output_cpp_path, harness_path, fragment_path, False, "Scenario fragment is empty.")

        before = harness_path.read_bytes()
        harness_hash = sha256(before).hexdigest()
        output_cpp_path.parent.mkdir(parents=True, exist_ok=True)
        # The fixed harness already includes scenario_fragment.inc. Rendering is
        # deliberately a byte-preserving copy of the verified harness.
        output_cpp_path.write_bytes(before)
        after = harness_path.read_bytes()
        if sha256(after).hexdigest() != harness_hash:
            return RenderFocasCppOutput(
                output_cpp_path,
                harness_path,
                fragment_path,
                False,
                "Fixed FOCAS harness changed during rendering; refusing to repair or overwrite it.",
                harness_hash,
            )
        return RenderFocasCppOutput(
            output_cpp_path,
            harness_path,
            fragment_path,
            True,
            None,
            harness_hash,
        )

    def input_summary(self, tool_input: RenderFocasCppInput) -> dict[str, Any]:
        return {
            "harness_path": str(tool_input.harness_path),
            "fragment_path": str(tool_input.fragment_path),
            "output_cpp_path": str(tool_input.output_cpp_path),
        }

    def output_summary(self, tool_output: RenderFocasCppOutput) -> dict[str, Any]:
        return {
            "output_cpp_path": str(tool_output.output_cpp_path),
            "harness_path": str(tool_output.harness_path),
            "fragment_path": str(tool_output.fragment_path),
            "success": tool_output.success,
            "harness_sha256": tool_output.harness_sha256,
            "error": tool_output.error,
        }

    def output_succeeded(self, tool_output: RenderFocasCppOutput) -> bool:
        return tool_output.success

    def output_error(self, tool_output: RenderFocasCppOutput) -> str | None:
        return tool_output.error
