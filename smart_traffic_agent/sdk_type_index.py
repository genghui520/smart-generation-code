"""Build and validate a lightweight ABI index from the active Fwlib32.h.

The index is deliberately derived from the configured header at runtime.  It
is a validation aid, not a replacement C/C++ parser: declarations and field
types are retained as evidence while member access checks use the declaration
type of each generated variable.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import re
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class SdkIndex:
    header_path: str
    header_sha256: str
    structs: dict[str, dict[str, Any]]
    unions: dict[str, dict[str, Any]]
    functions: dict[str, str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 5,
            "header_path": self.header_path,
            "header_sha256": self.header_sha256,
            "structs": self.structs,
            "unions": self.unions,
            "functions": self.functions,
        }


def _strip_comments(text: str) -> str:
    return re.sub(r"/\*.*?\*/|//[^\r\n]*", "", text, flags=re.DOTALL)


def _active_header(text: str, initial_macros: set[str] | None = None) -> str:
    """Select the default SDK preprocessor branch before parsing typedefs."""
    macros = set(initial_macros or ())
    output: list[str] = []
    stack: list[tuple[bool, bool]] = []  # parent, branch_taken
    active = True

    def condition(expression: str) -> bool:
        expression = re.sub(
            r"defined\s*(?:\(\s*([A-Za-z_]\w*)\s*\)|\s+([A-Za-z_]\w*))",
            lambda match: "1" if (match.group(1) or match.group(2)) in macros else "0",
            expression,
        )
        expression = re.sub(r"\b[A-Za-z_]\w*\b", "0", expression)
        expression = expression.replace("&&", " and ").replace("||", " or ")
        expression = re.sub(r"!(?!=)", " not ", expression)
        if not re.fullmatch(r"[\s\d()!<>=&|+*/%.-]*(?:and|or|not|[\s\d()!<>=&|+*/%.-])*", expression):
            return False
        try:
            return bool(eval(expression, {"__builtins__": {}}, {}))
        except (SyntaxError, ValueError, TypeError, ZeroDivisionError):
            return False

    for line in _strip_comments(text).splitlines(keepends=True):
        directive = re.match(r"\s*#\s*(if|ifdef|ifndef|elif|else|endif|define|undef)\b(.*)", line)
        if not directive:
            output.append(line if active else "\n")
            continue
        kind, argument = directive.group(1), directive.group(2).strip()
        if kind in {"if", "ifdef", "ifndef"}:
            selected = (condition(argument) if kind == "if" else argument in macros if kind == "ifdef" else argument not in macros)
            stack.append((active, selected if active else False))
            active = active and selected
        elif kind == "elif" and stack:
            parent, taken = stack[-1]
            active = parent and not taken and condition(argument)
            stack[-1] = (parent, taken or active)
        elif kind == "else" and stack:
            parent, taken = stack[-1]
            active = parent and not taken
            stack[-1] = (parent, True)
        elif kind == "endif" and stack:
            parent, _ = stack.pop()
            active = parent
        elif kind == "define" and active:
            name = re.match(r"([A-Za-z_]\w*)", argument)
            if name:
                macros.add(name.group(1))
        elif kind == "undef" and active:
            macros.discard(argument.split()[0] if argument else "")
        output.append("\n")
    return "".join(output)


def _matching_brace(text: str, opening: int) -> int:
    depth = 0
    for index in range(opening, len(text)):
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
            if depth == 0:
                return index
    return -1


def _fields(body: str) -> dict[str, dict[str, Any]]:
    fields: dict[str, dict[str, Any]] = {}
    # Parse only top-level semicolon statements. This avoids treating fields
    # inside nested anonymous structs/unions as members of the outer type.
    statements: list[str] = []
    start = 0
    depth = 0
    for index, char in enumerate(body):
        if char == "{":
            depth += 1
        elif char == "}":
            depth = max(depth - 1, 0)
        elif char == ";" and depth == 0:
            statements.append(body[start:index].strip())
            start = index + 1
    declaration = re.compile(
        r"(?P<type>.+?)\s+(?:\*\s*)?(?P<name>[A-Za-z_]\w*)\s*"
        r"(?P<arrays>(?:\[\s*[^\]]+\s*\])*)$",
        re.DOTALL,
    )
    for statement in statements:
        match = declaration.match(statement)
        if not match or "(" in statement:
            continue
        name = match.group("name")
        prefix = " ".join(match.group("type").split())
        if not prefix or prefix in {"#if", "#ifdef", "#ifndef", "#else", "#endif"}:
            continue
        dimensions = [value.strip() for value in re.findall(r"\[\s*([^\]]+)\s*\]", match.group("arrays"))]
        fields[name] = {
            "type": prefix.replace("* ", "*").strip(),
            "array": "][".join(dimensions) if dimensions else None,
            "array_dimensions": dimensions,
        }
    return fields


def _extract_types(header: str) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    clean = _active_header(header)
    structs: dict[str, dict[str, Any]] = {}
    unions: dict[str, dict[str, Any]] = {}
    # Conditional alternatives have been filtered by _active_header.
    pattern = re.compile(r"typedef\s+(struct|union)\s+([A-Za-z_]\w*)?\s*\{", re.IGNORECASE)
    for match in pattern.finditer(clean):
        close = _matching_brace(clean, clean.find("{", match.start()))
        if close < 0:
            continue
        alias = re.match(r"\s*([A-Za-z_]\w*)\s*;", clean[close + 1:])
        if not alias:
            continue
        name = alias.group(1).upper()
        value = {
            "tag": (match.group(2) or "").strip(),
            "fields": _fields(clean[clean.find("{", match.start()) + 1:close]),
            "declaration": clean[match.start():close + 1] + " " + alias.group(0).strip(),
        }
        (structs if match.group(1).lower() == "struct" else unions)[name] = value
    return structs, unions


def build_sdk_type_index(header_path: Path) -> SdkIndex:
    header_path = Path(header_path)
    raw = header_path.read_bytes()
    text = raw.decode("latin-1")
    structs, unions = _extract_types(text)
    active_text = _active_header(text)
    functions: dict[str, str] = {}
    for match in re.finditer(
        r"((?:FWLIBAPI\s+[^;]*?\b)?(cnc_[A-Za-z0-9_]+|pmc_[A-Za-z0-9_]+)\s*\([^;]*?\)\s*;)",
        active_text,
        re.IGNORECASE | re.DOTALL,
    ):
        functions[match.group(2)] = " ".join(match.group(1).split())
    return SdkIndex(str(header_path), hashlib.sha256(raw).hexdigest(), structs, unions, functions)


def load_or_build_sdk_type_index(header_path: Path, index_path: Path | None = None) -> SdkIndex:
    header_path = Path(header_path)
    current_hash = hashlib.sha256(header_path.read_bytes()).hexdigest()
    if index_path is not None and Path(index_path).is_file():
        try:
            payload = json.loads(Path(index_path).read_text(encoding="utf-8"))
            if (
                payload.get("schema_version") == 5
                and payload.get("header_sha256") == current_hash
                and Path(str(payload.get("header_path", ""))).resolve() == header_path.resolve()
            ):
                return SdkIndex(
                    str(payload.get("header_path", header_path)), current_hash,
                    dict(payload.get("structs", {})), dict(payload.get("unions", {})),
                    dict(payload.get("functions", {})),
                )
        except (OSError, ValueError, TypeError):
            pass
    index = build_sdk_type_index(header_path)
    if index_path is not None:
        path = Path(index_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(index.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    return index


def _declarations(source: str) -> dict[str, str]:
    declarations: dict[str, str] = {}
    type_names = re.compile(r"\b(?:const\s+)?([A-Z][A-Z0-9_]*)\s*(?:[&*]\s*)*([A-Za-z_]\w*)\s*(?:\[\s*[^\]]*\s*\])?\s*(?:[;={,)])")
    for match in type_names.finditer(source):
        declarations[match.group(2)] = match.group(1).upper()
    return declarations


def _mask_cpp_literals_and_comments(source: str) -> str:
    pattern = re.compile(r'"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'|/\*.*?\*/|//[^\r\n]*', re.DOTALL)
    return pattern.sub(lambda match: re.sub(r"[^\r\n]", " ", match.group(0)), source)


def validate_struct_member_access(source: str, index: SdkIndex, *, source_name: str = "scenario_fragment.inc") -> list[str]:
    code = _mask_cpp_literals_and_comments(source)
    declarations = _declarations(code)
    diagnostics: list[str] = []
    access = re.compile(r"\b([A-Za-z_]\w*)\s*(?:\.|->)\s*([A-Za-z_]\w*)")
    seen: set[tuple[str, str, int]] = set()
    for match in access.finditer(code):
        variable, field = match.group(1), match.group(2)
        type_name = declarations.get(variable)
        if not type_name or type_name not in index.structs and type_name not in index.unions:
            continue
        definition = index.structs.get(type_name) or index.unions.get(type_name) or {}
        fields = definition.get("fields", {})
        if field in fields:
            continue
        line = source.count("\n", 0, match.start()) + 1
        key = (type_name, field, line)
        if key in seen:
            continue
        seen.add(key)
        valid = ", ".join(sorted(fields)) or "<none>"
        diagnostics.append(
            f"BLOCKING: SDK_TYPE_MISMATCH: struct={type_name} field={field} location={source_name}:{line} "
            f"valid_members={valid} header={index.header_path}"
        )
    return diagnostics


def validate_sdk_type_declarations(source: str, index: SdkIndex, *, source_name: str = "scenario_fragment.inc") -> list[str]:
    """Reject invented SDK struct/union names before MSVC compilation.

    The generated fragment may use framework types and ordinary C++ scalar
    types.  A declaration that looks like an all-caps SDK payload type must,
    however, exist in the active header index.  This catches names such as
    ``ODBPARAINFO`` deterministically instead of allowing the compiler/LLM
    repair loop to guess a similar type.
    """
    code = _mask_cpp_literals_and_comments(source)
    known = set(index.structs) | set(index.unions)
    scalar = {
        "BOOL", "BYTE", "CHAR", "DOUBLE", "DWORD", "FLOAT", "INT", "LONG",
        "SHORT", "SIZE_T", "UCHAR", "UINT", "ULONG", "USHORT", "VOID",
        "WCHAR", "WORD", "LONGLONG", "ULONGLONG", "HANDLE", "LPSTR",
        "LPCSTR", "LPCTSTR", "TCHAR", "CHAR_T",
    }
    diagnostics: list[str] = []
    # Restrict matching to declaration positions.  In particular, do not
    # scan arbitrary expressions: ``if (ret == EW_OK)`` must never be parsed
    # as a declaration of type ``EW_O`` and variable ``K``.  The delimiter
    # prefix also keeps the match anchored to a source statement/block.
    declaration = re.compile(
        r"(?:^|[;{}])(?P<indent>\s*)"
        r"(?P<qual>(?:(?:const|volatile|struct)\s+)*)"
        r"(?P<type>[A-Z][A-Z0-9_]*)"
        r"(?P<spacing>\s*\*+\s*|\s+)"
        r"(?P<variable>[A-Za-z_]\w*)"
        r"(?P<tail>\s*(?:\{\s*\}|\[[^\]]*\]|=[^;]+|;))",
        re.MULTILINE,
    )
    seen: set[tuple[str, str, int]] = set()
    for match in declaration.finditer(code):
        variable = match.group("variable")
        type_name = match.group("type").upper()
        normalized = str(type_name).upper()
        if normalized in known or normalized in scalar:
            continue
        if not re.fullmatch(r"[A-Z][A-Z0-9_]*", normalized):
            continue
        type_offset = match.start("type")
        line = source.count("\n", 0, type_offset) + 1
        line_start = source.rfind("\n", 0, type_offset) + 1
        line_end = source.find("\n", type_offset)
        if line_end < 0:
            line_end = len(source)
        snippet = source[line_start:line_end].strip()
        key = (normalized, variable, line)
        if key in seen:
            continue
        seen.add(key)
        diagnostics.append(
            f"BLOCKING: SDK_TYPE_UNKNOWN: type={normalized} variable={variable} "
            f"location={source_name}:{line}; code={snippet!r}; type is absent from active Fwlib32.h "
            f"({index.header_path}); use only prototype-associated SDK types"
        )
    return diagnostics


def sdk_index_prompt_context(index: SdkIndex, type_names: set[str] | None = None) -> str:
    selected = type_names or set(index.structs)
    rows = []
    for name in sorted(selected):
        item = index.structs.get(name) or index.unions.get(name)
        if not item:
            continue
        fields = ", ".join(
            f"{field}:{details.get('type')}" + (f"[{details['array']}]" if details.get("array") else "")
            for field, details in sorted(item.get("fields", {}).items())
        )
        rows.append(f"{name} {{{fields}}}")
    return f"SDK header={index.header_path} sha256={index.header_sha256}\n" + "\n".join(rows)


def sdk_environment_identity(header_path: Path, dll_path: Path) -> dict[str, Any]:
    """Record the exact ABI inputs; matching releases require external proof."""
    header_path = Path(header_path)
    dll_path = Path(dll_path)
    import_library = header_path.parent.parent / "Fwlib32.lib"

    def identity(path: Path) -> dict[str, Any]:
        if not path.is_file():
            return {"path": str(path), "exists": False}
        raw = path.read_bytes()
        return {
            "path": str(path), "exists": True,
            "size": len(raw), "sha256": hashlib.sha256(raw).hexdigest(),
        }

    return {
        "header": identity(header_path),
        "import_library": identity(import_library),
        "runtime_dll": identity(dll_path),
        "abi_release_match": "unverified",
    }
