"""Deterministic assembly of the API-operation fragment.

The model may provide values in ``ExecutionPlan``/parameter contracts, but it
does not choose FOCAS symbols, ABI types, argument order, or buffer lengths.
Those parts are delegated to the existing SDK-backed deterministic recipes.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Iterable

from .integrations.ncguide import default_focas_header_dir
from .sdk_type_index import load_or_build_sdk_type_index


class DeterministicAssemblyError(ValueError):
    """Raised when the active SDK/recipe cannot assemble a legal fragment."""


_SDK_DECLARATION = re.compile(
    r"(?m)^\s*(?:const\s+)?(?P<type>[A-Z][A-Z0-9_]*)\s*(?:[*&]\s*)?"
    r"(?P<name>[A-Za-z_]\w*)\s*(?:\{|\[|=|;|\()"
)
_API_MEMBER = re.compile(r"\bapi\.(?P<name>(?:cnc|pmc)_\w+)\s*\(")


def _validate_recipe_against_sdk(core: str, sdk_index: Any) -> None:
    """Reject recipe output that names a type/symbol absent from the SDK.

    Recipes are source templates, so this check is intentionally limited to
    declaration and API-member syntax.  It does not scan return-code constants,
    strings, comments, or ordinary expressions; that avoids the historical
    ``EW_OK`` → ``EW_O``/``K`` false positive while preserving real unknown
    type failures.
    """
    sdk_types = set(getattr(sdk_index, "structs", {})) | set(getattr(sdk_index, "unions", {}))
    unknown_types = sorted({
        match.group("type") for match in _SDK_DECLARATION.finditer(core)
        if match.group("type") not in sdk_types
    })
    if unknown_types:
        raise DeterministicAssemblyError(
            "DETERMINISTIC_ASSEMBLY_UNKNOWN_TYPE: " + ",".join(unknown_types)
            + "; declaration is absent from active Fwlib32.h"
        )
    missing_functions = sorted({
        match.group("name") for match in _API_MEMBER.finditer(core)
        if match.group("name") not in getattr(sdk_index, "functions", {})
    })
    if missing_functions:
        raise DeterministicAssemblyError(
            "DETERMINISTIC_ASSEMBLY_MISSING_PROTOTYPE: " + ",".join(missing_functions)
            + "; api member is absent from active Fwlib32.h"
        )


def _step_values(steps: Iterable[Any]) -> dict[str, dict[str, object]]:
    values: dict[str, dict[str, object]] = {}
    for step in steps:
        names = [str(item).strip() for item in str(getattr(step, "protocol_function", "")).split("/") if str(item).strip()]
        params = getattr(step, "parameters", {})
        if not isinstance(params, dict):
            continue
        concrete = {
            str(key): value for key, value in params.items()
            if key not in {"parameter_generation", "parameter_cases", "parameter_constraints", "cases", "bound_parameters"}
            and isinstance(value, (int, float, str))
        }
        if not concrete:
            continue
        for name in names:
            values.setdefault(name, {}).update(concrete)
    return values


def _api_cases(parameter_contract: dict[str, Any], api_names: list[str], steps: Iterable[Any]) -> list[dict[str, dict[str, object]]]:
    """Materialize bounded value cases without inventing a Cartesian product."""
    step_list = list(steps)
    base = _step_values(step_list)
    step_cases: dict[str, list[dict[str, object]]] = {}
    for step in step_list:
        names = [str(item).strip() for item in str(getattr(step, "protocol_function", "")).split("/") if str(item).strip()]
        params = getattr(step, "parameters", {})
        if not isinstance(params, dict):
            continue
        raw_cases = params.get("parameter_cases", params.get("cases", []))
        if not isinstance(raw_cases, list):
            continue
        normalized = [dict(item) for item in raw_cases if isinstance(item, dict)]
        for name in names:
            if normalized:
                step_cases.setdefault(name, []).extend(normalized)
    rules = parameter_contract.get("functions", {}) if isinstance(parameter_contract, dict) else {}
    count = max((
        len(rules.get(name, {}).get("cases", [])) or len(step_cases.get(name, []))
        for name in api_names
    ), default=0)
    if count == 0:
        return [{name: dict(base.get(name, {})) for name in api_names}]
    cases: list[dict[str, dict[str, object]]] = []
    for index in range(count):
        row: dict[str, dict[str, object]] = {}
        for name in api_names:
            values = dict(base.get(name, {}))
            api_rule = rules.get(name, {}) if isinstance(rules, dict) else {}
            selected = api_rule.get("cases", []) if isinstance(api_rule, dict) else []
            if not selected:
                selected = step_cases.get(name, [])
            if selected:
                candidate = selected[index % len(selected)]
                if isinstance(candidate, dict):
                    values.update(candidate)
            row[name] = values
        cases.append(row)
    return cases


def assemble_core_cpp(
    *,
    targets: Iterable[str],
    planned_functions: Iterable[str],
    steps: Iterable[Any],
    parameter_contract: dict[str, Any],
    sdk_header: str | Path | None = None,
) -> str:
    """Assemble statements using only the active deterministic SDK recipes.

    A missing recipe or missing concrete value is reported explicitly.  The
    caller can route that structured error back through the existing Router →
    CodeGenerationAgent path; no LLM-generated C++ is used as a fallback.
    """
    # Preserve the order declared by the ExecutionPlan.  A set or a sorted
    # list would make otherwise valid sequence plans reorder their calls.
    step_list = list(steps)
    requested = [str(item).strip() for item in targets if str(item).strip()]
    requested_set = set(requested)
    target_names: list[str] = []
    seen_targets: set[str] = set()
    for step in step_list:
        for name in str(getattr(step, "protocol_function", "")).split("/"):
            name = name.strip()
            if name in requested_set and name not in seen_targets:
                target_names.append(name)
                seen_targets.add(name)
    for name in requested:
        if name not in seen_targets:
            target_names.append(name)
            seen_targets.add(name)
    if not target_names:
        raise DeterministicAssemblyError("DETERMINISTIC_ASSEMBLY_MISSING: no target API")
    planned = {str(item).strip() for item in planned_functions if str(item).strip()}
    header = Path(sdk_header) if sdk_header is not None else default_focas_header_dir() / "Fwlib32.h"
    if not header.is_file():
        raise DeterministicAssemblyError(f"DETERMINISTIC_ASSEMBLY_MISSING_SDK: active Fwlib32.h not found: {header}")
    sdk_index = load_or_build_sdk_type_index(header, None)
    missing_symbols = [name for name in target_names if name not in sdk_index.functions]
    if missing_symbols:
        raise DeterministicAssemblyError(
            "DETERMINISTIC_ASSEMBLY_MISSING_PROTOTYPE: " + ",".join(missing_symbols) +
            "; no active Fwlib32.h prototype"
        )
    cases = _api_cases(parameter_contract, target_names, step_list)
    fragments: list[str] = []
    # Importing the runtime lazily keeps this module usable by tooling that
    # only inspects the assembler, while production still uses the same
    # SDK-backed recipes and legacy runtime implementation.
    from .focas_capture_runtime import deterministic_read_core, motion_read_core

    def recipe_for(api_name: str, values: dict[str, dict[str, object]]) -> str | None:
        """Return one existing deterministic recipe for one planned API.

        Parameterized APIs are resolved by ``deterministic_read_core``.  The
        motion recipe is an existing ABI-safe template for output-only motion
        reads (for example ``cnc_actf``); it does not invent input values.
        No generic fallback is used for an API without a recipe.
        """
        recipe = deterministic_read_core(
            {api_name},
            parameter_values=values,
            planned_functions=planned,
        )
        if recipe:
            return recipe
        motion = motion_read_core({api_name})
        return motion if motion else None

    for case_index, values in enumerate(cases, 1):
        missing = [name for name in target_names if not values.get(name)]
        # APIs without input dimensions legitimately have an empty value map.
        rules = parameter_contract.get("functions", {}) if isinstance(parameter_contract, dict) else {}
        missing = [name for name in missing if (rules.get(name, {}).get("dimensions") if isinstance(rules, dict) else [])]
        if missing:
            raise DeterministicAssemblyError(
                "DETERMINISTIC_ASSEMBLY_MISSING_VALUES: " + ",".join(missing) +
                "; parameter values must be supplied by ExecutionPlan/runtime binding"
            )
        recipe_fragments: list[str] = []
        missing_recipes: list[str] = []
        for name in target_names:
            recipe = recipe_for(name, values)
            if recipe:
                recipe_fragments.append(recipe)
            else:
                missing_recipes.append(name)
        if missing_recipes:
            raise DeterministicAssemblyError(
                "DETERMINISTIC_ASSEMBLY_UNSUPPORTED: no SDK-backed deterministic recipe for "
                + ",".join(missing_recipes)
            )
        core = "\n".join(recipe_fragments)
        _validate_recipe_against_sdk(core, sdk_index)
        # Recipes contain local return statements. Execute each bounded case in
        # a lambda so a return only ends that case and does not corrupt the
        # fixed scenario wrapper or the next parameter combination.
        fragments.append(
            "    ret = [&]() -> short {\n"
            + core.rstrip()
            + "\n        return EW_OK;\n    }();\n"
            + f"    if (ret != EW_OK) {{ ok = false; }} // deterministic case {case_index}\n"
        )
    return "\n".join(fragments)
