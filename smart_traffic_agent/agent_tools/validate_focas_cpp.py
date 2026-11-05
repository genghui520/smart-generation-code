from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .base import AgentTool
from ..execution_contracts import ExecutionContract


@dataclass(frozen=True, slots=True)
class ValidateFocasCppInput:
    cpp_path: Path
    fragment_path: Path | None = None
    nc_program: str = ""
    executable_steps: tuple[Any, ...] = ()
    allow_delete_all_programs: bool = False
    require_official_focas_header: bool = False
    require_real_pcap_capture: bool = False
    code_only_evaluation: bool = False
    code_spec: Any = None
    production_contract: ExecutionContract | None = None


@dataclass(slots=True)
class ValidateFocasCppOutput:
    success: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    checked_rules: list[str] = field(default_factory=list)


class ValidateFocasCppTool(AgentTool[ValidateFocasCppInput, ValidateFocasCppOutput]):
    name = "validate_focas_cpp"
    description = (
        "Validate generated FANUC FOCAS C++ code against "
        "known API and ABI constraints before MSVC compilation."
    )

    def invoke(self, tool_input: ValidateFocasCppInput) -> ValidateFocasCppOutput:
        if not tool_input.cpp_path.exists():
            return ValidateFocasCppOutput(
                False,
                [f"C++ source does not exist: {tool_input.cpp_path}"],
                [],
                ["source existence"],
            )

        source = tool_input.cpp_path.read_text(encoding="utf-8")
        shell = source
        fragment = ""
        if tool_input.fragment_path is not None:
            if not tool_input.fragment_path.exists():
                return ValidateFocasCppOutput(
                    False,
                    [f"Scenario fragment does not exist: {tool_input.fragment_path}"],
                    [],
                    ["source existence", "fragment existence"],
                )
            fragment = tool_input.fragment_path.read_text(encoding="utf-8")
            source = source + "\n" + fragment

        # Keep the existing validator as the single source of truth.  The
        # import is intentionally lazy because the legacy validator currently
        # lives beside CodeGenerationAgent and imports AgentTool definitions.
        from ..agents.code_generator import (
            validate_code_only_source,
            validate_code_spec_coverage,
            validate_download_payload,
            validate_generated,
            validate_focas_argument_semantics,
            validate_official_output_usage,
            validate_programmatic_abi_binding,
            validate_scenario_fragment,
        )
        from ..integrations.ncguide import default_focas_header_dir
        from ..sdk_type_index import (
            load_or_build_sdk_type_index,
            validate_sdk_type_declarations,
            validate_struct_member_access,
        )
        sdk_header = default_focas_header_dir() / "Fwlib32.h"
        if sdk_header.is_file():
            sdk_index = load_or_build_sdk_type_index(
                sdk_header, Path("rag_indexes/focas/sdk_type_index_0iD.json")
            )
            type_diagnostics = validate_struct_member_access(
                fragment or source, sdk_index,
                source_name=tool_input.fragment_path.name if tool_input.fragment_path else tool_input.cpp_path.name,
            )
            type_diagnostics.extend(validate_sdk_type_declarations(
                fragment or source, sdk_index,
                source_name=tool_input.fragment_path.name if tool_input.fragment_path else tool_input.cpp_path.name,
            ))
        else:
            type_diagnostics = [f"BLOCKING: SDK_TYPE_MISMATCH: configured SDK header not found: {sdk_header}"]

        if tool_input.production_contract is not None:
            from ..focas_capture_runtime import validate_production_source
            from ..integrations.ncguide import default_focas_header_dir
            diagnostics = validate_production_source(
                shell, fragment, (default_focas_header_dir() / "Fwlib32.h").read_text(encoding="latin-1"),
                tool_input.production_contract, tool_input.nc_program,
                allow_delete_all=tool_input.allow_delete_all_programs,
            )
            # Output buffer semantics are checked on model code; the immutable
            # driver is validated by its own compilation/regression tests.
            from ..agents.code_generator import validate_poselem_scalar_access, validate_odbact_scalar_access
            diagnostics.extend(validate_poselem_scalar_access(fragment))
            diagnostics.extend(validate_odbact_scalar_access(fragment))
            # Production contracts use the fixed harness path, so the normal
            # generated-source validator is not run.  Keep the SDK semantic
            # argument-order check active here as well; C++ cannot detect a
            # swapped pair of same-typed arguments such as axis/length.
            diagnostics.extend(validate_focas_argument_semantics(fragment or shell))
        else:
            diagnostics = validate_generated(
                source,
                tool_input.nc_program,
                list(tool_input.executable_steps),
                allow_delete_all_programs=tool_input.allow_delete_all_programs,
                require_official_focas_header=tool_input.require_official_focas_header,
                require_real_pcap_capture=tool_input.require_real_pcap_capture,
                code_only_evaluation=tool_input.code_only_evaluation,
            )
        if tool_input.fragment_path is not None:
            diagnostics.extend(validate_scenario_fragment(fragment))
        diagnostics.extend(type_diagnostics)
        if tool_input.code_only_evaluation:
            diagnostics.extend(validate_code_only_source(source))
            diagnostics.extend(validate_code_spec_coverage(source, tool_input.code_spec))
            diagnostics.extend(validate_programmatic_abi_binding(source, tool_input.code_spec))
            diagnostics.extend(validate_official_output_usage(source, tool_input.code_spec))
            diagnostics.extend(validate_download_payload(source, tool_input.code_spec))

        errors = [item for item in diagnostics if item.startswith("BLOCKING:")]
        warnings = [item for item in diagnostics if not item.startswith("BLOCKING:")]
        return ValidateFocasCppOutput(
            success=not errors,
            errors=errors,
            warnings=warnings,
            checked_rules=[
                "fixed harness and Fwlib32.h usage",
                "official FOCAS symbol existence and decltype ABI binding",
                "POSELM scalar data access",
                "ODBAXIS array data access",
                "ODBACT scalar data access",
                "SDK-derived struct/union member access",
                "SDK type declaration existence against active Fwlib32.h",
                "planned API coverage and semantic bindings",
                "scenario fragment boundary",
                "NC payload framing",
            ],
        )

    def input_summary(self, tool_input: ValidateFocasCppInput) -> dict[str, Any]:
        return {
            "cpp_path": str(tool_input.cpp_path),
            "fragment_path": str(tool_input.fragment_path) if tool_input.fragment_path else None,
            "executable_step_count": len(tool_input.executable_steps),
            "code_only_evaluation": tool_input.code_only_evaluation,
            "production_contract": tool_input.production_contract.to_dict() if tool_input.production_contract else None,
        }

    def output_summary(self, tool_output: ValidateFocasCppOutput) -> dict[str, Any]:
        return {
            "success": tool_output.success,
            "error_count": len(tool_output.errors),
            "warning_count": len(tool_output.warnings),
            "checked_rule_count": len(tool_output.checked_rules),
        }

    def output_succeeded(self, tool_output: ValidateFocasCppOutput) -> bool:
        return tool_output.success

    def output_error(self, tool_output: ValidateFocasCppOutput) -> str | None:
        if tool_output.success:
            return None
        return "; ".join(tool_output.errors) or "FOCAS C++ validation failed."
