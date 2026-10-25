from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .base import AgentTool


@dataclass(frozen=True, slots=True)
class WriteGeneratedArtifactInput:
    path: Path
    content: str


@dataclass(frozen=True, slots=True)
class WriteGeneratedArtifactOutput:
    path: Path
    bytes_written: int


class WriteGeneratedArtifactTool(AgentTool[WriteGeneratedArtifactInput, WriteGeneratedArtifactOutput]):
    name = "write_generated_artifact"
    description = "Write one generated source or NC artifact to its assigned workflow directory."

    def invoke(self, tool_input: WriteGeneratedArtifactInput) -> WriteGeneratedArtifactOutput:
        tool_input.path.parent.mkdir(parents=True, exist_ok=True)
        tool_input.path.write_text(tool_input.content, encoding="utf-8")
        return WriteGeneratedArtifactOutput(tool_input.path, len(tool_input.content.encode("utf-8")))

    def input_summary(self, tool_input: WriteGeneratedArtifactInput) -> dict[str, Any]:
        return {"path": str(tool_input.path), "content_chars": len(tool_input.content)}

    def output_summary(self, tool_output: WriteGeneratedArtifactOutput) -> dict[str, Any]:
        return {"path": str(tool_output.path), "bytes_written": tool_output.bytes_written}
