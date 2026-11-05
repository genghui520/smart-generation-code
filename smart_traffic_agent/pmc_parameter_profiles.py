"""Deterministic PMC range profiles used for read/write coverage planning.

The profiles mirror the documented IODBPMC examples.  They are data, not
device writes: read calls use the address/type/length fields and record the
returned union bytes. Write calls use the same coupled profiles while
preserving each real snapshot and verifying restoration.
"""
from __future__ import annotations

from typing import Any


_WIDTH_BY_TYPE = {0: 1, 1: 2, 2: 4, 4: 4, 5: 8}
_UNION_BY_TYPE = {0: "cdata", 1: "idata", 2: "ldata", 4: "fdata", 5: "dfdata"}


def pmc_data_width(data_type: int) -> int:
    try:
        return _WIDTH_BY_TYPE[int(data_type)]
    except (KeyError, TypeError, ValueError):
        raise ValueError(f"unsupported PMC data_type: {data_type!r}") from None


def pmc_union_field(data_type: int) -> str:
    try:
        return _UNION_BY_TYPE[int(data_type)]
    except (KeyError, TypeError, ValueError):
        raise ValueError(f"unsupported PMC data_type: {data_type!r}") from None


def pmc_length(data_type: int, element_count: int) -> int:
    """Return the SDK IODBPMC length, including its 8-byte header."""
    count = int(element_count)
    if count <= 0:
        raise ValueError("PMC element_count must be positive")
    return 8 + pmc_data_width(data_type) * count


def _profile(name: str, type_a: int, type_d: int, start: int, end: int,
             values: list[Any]) -> dict[str, Any]:
    length = pmc_length(type_d, len(values))
    return {
        "id": name,
        "type_a": int(type_a),
        "type_d": int(type_d),
        "datano_s": int(start),
        "datano_e": int(end),
        "element_count": len(values),
        "length": length,
        "union": pmc_union_field(type_d),
        "values": list(values),
        "address_kind": "R" if int(type_a) == 5 else "D" if int(type_a) == 9 else "unknown",
        "length_formula": f"8 + {pmc_data_width(type_d)} * {len(values)} = {length}",
    }


# These are the four explicitly requested, SDK-shaped examples.  The first
# range has two address numbers but one word value, as in the supplied table;
# element_count therefore comes from the union payload, not end-start+1.
PMC_RANGE_PROFILES: tuple[dict[str, Any], ...] = (
    _profile("D0100_word", 9, 1, 100, 101, [250]),
    _profile("R0200_R0209_byte", 5, 0, 200, 209, [0] * 10),
    _profile("D0100_float32", 9, 4, 100, 103, [1.1]),
    _profile("R0104_R0119_float64", 5, 5, 104, 119, [-2.2, 33.3]),
)


def pmc_profile_inputs(profile: dict[str, Any]) -> dict[str, Any]:
    """Fields that belong in an API input log/audit signature."""
    # pmc_rdpmcrng names the four selectors adr_type/data_type/s_number/
    # e_number; the returned IODBPMC header uses type_a/type_d/datano_*.
    return {
        "adr_type": profile["type_a"],
        "data_type": profile["type_d"],
        "s_number": profile["datano_s"],
        "e_number": profile["datano_e"],
        "length": profile["length"],
    }


def pmc_buffer_fields(profile: dict[str, Any]) -> dict[str, Any]:
    """Expected output-header and union fields for a profile."""
    return {
        "buf.type_a": profile["type_a"],
        "buf.type_d": profile["type_d"],
        "buf.datano_s": profile["datano_s"],
        "buf.datano_e": profile["datano_e"],
        "buf.union": profile["union"],
        "buf.values": list(profile["values"]),
    }


def validate_pmc_profile(profile: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    try:
        data_type = int(profile["type_d"])
        values = profile["values"]
        expected = pmc_length(data_type, len(values))
        if int(profile["length"]) != expected:
            errors.append(f"length={profile.get('length')} expected={expected}")
        if int(profile["datano_e"]) < int(profile["datano_s"]):
            errors.append("datano_e is below datano_s")
        if profile.get("union") != pmc_union_field(data_type):
            errors.append("union field does not match type_d")
    except (KeyError, TypeError, ValueError) as exc:
        errors.append(str(exc))
    return errors


def validate_all_pmc_profiles() -> dict[str, list[str]]:
    return {str(profile["id"]): validate_pmc_profile(profile) for profile in PMC_RANGE_PROFILES}
