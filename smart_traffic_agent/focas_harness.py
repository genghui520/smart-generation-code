"""Shared renderer for the one fixed FANUC FOCAS C++ harness."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable


FOCAS_HARNESS_TEMPLATE_PATH = Path(__file__).resolve().parent / "templates" / "focas_harness.cpp"
_FOCAS_SYMBOL = re.compile(r"^(?:cnc|pmc)_[A-Za-z0-9_]+$")
_FIXED_HARNESS_FUNCTIONS = {"cnc_allclibhndl3", "cnc_freelibhndl"}


def load_focas_harness_template() -> str:
    """Load the single immutable harness source used by every variant."""

    return FOCAS_HARNESS_TEMPLATE_PATH.read_text(encoding="utf-8")


def render_focas_harness_template(selected_functions: Iterable[str]) -> str:
    """Fill only the generic dynamic-binding placeholders in the shared template.

    Function names are supplied by the calling agent or Planner. This renderer
    performs no scenario inference, API selection, ordering, or parameter work.
    """

    requested = {str(name).strip() for name in selected_functions if str(name).strip()}
    invalid = sorted(name for name in requested if not _FOCAS_SYMBOL.fullmatch(name))
    if invalid:
        raise ValueError("Invalid FOCAS symbol names: " + ", ".join(invalid))
    functions = sorted(requested | _FIXED_HARNESS_FUNCTIONS)
    fields = "\n".join(
        f"    decltype(&::{name}) {name} = nullptr;" for name in functions
    )
    bindings = "\n".join(
        f'        {name} = reinterpret_cast<decltype({name})>(GetProcAddress(dll, "{name}"));\n'
        f'        if (!{name}) return false;'
        for name in functions
    )
    template = load_focas_harness_template()
    return template.replace("__FOCAS_FIELDS__", fields).replace("__FOCAS_BINDINGS__", bindings)
