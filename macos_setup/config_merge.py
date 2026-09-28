"""Pure standard library configuration mergers for JSON and TOML (Python 3.9+)."""

from __future__ import annotations

import json
import re
from typing import Any


class ConfigMergeError(Exception):
    """Raised when configuration content cannot be parsed or merged."""


# ---------------------------------------------------------------------------
# JSON Merger
# ---------------------------------------------------------------------------


def _merge_dict(
    repo: dict[str, Any], target: dict[str, Any], path: tuple[str, ...] = ()
) -> dict[str, Any]:
    merged: dict[str, Any] = dict(target)
    for key, repo_val in repo.items():
        if key not in merged:
            merged[key] = repo_val
            continue
        target_val = merged[key]
        if isinstance(repo_val, dict) and isinstance(target_val, dict):
            merged[key] = _merge_dict(repo_val, target_val, (*path, key))
        elif isinstance(repo_val, list) and isinstance(target_val, list):
            # Permission lists in Claude/Gemini settings: order-preserving union
            if path and path[-1] == "permissions" and key in ("allow", "deny"):
                combined = list(repo_val)
                for item in target_val:
                    if item not in combined:
                        combined.append(item)
                merged[key] = combined
            else:
                # Other lists (e.g. structured hooks or fallback filenames): repo authoritative
                merged[key] = repo_val
        else:
            merged[key] = repo_val

    # Resolve allow/deny conflict: deny rules always take precedence
    if (
        "allow" in merged
        and "deny" in merged
        and isinstance(merged["allow"], list)
        and isinstance(merged["deny"], list)
    ):
        if (path and path[-1] == "permissions") or (
            not path and "permissions" in merged
        ):
            deny_set = set(merged["deny"])
            merged["allow"] = [item for item in merged["allow"] if item not in deny_set]

    return merged


def merge_json(repo_content: str, target_content: str) -> str:
    """Merge JSON repo updates into target machine settings, preserving target keys.

    Args:
        repo_content: Source configuration template from the repository.
        target_content: Existing host configuration to update in-place.

    Returns:
        The merged JSON configuration string ending in a newline.

    Raises:
        ConfigMergeError: If either source or target contains invalid JSON syntax or non-object root.
    """
    try:
        repo_data = json.loads(repo_content)
    except Exception as exc:
        raise ConfigMergeError(f"Malformed source JSON: {exc}") from exc
    try:
        target_data = json.loads(target_content)
    except Exception as exc:
        raise ConfigMergeError(f"Malformed target JSON: {exc}") from exc

    if not isinstance(repo_data, dict) or not isinstance(target_data, dict):
        raise ConfigMergeError("JSON configuration root must be an object")

    merged = _merge_dict(repo_data, target_data)
    return json.dumps(merged, indent=2, ensure_ascii=False) + "\n"


# ---------------------------------------------------------------------------
# TOML Merger
# ---------------------------------------------------------------------------


def _extract_header(line: str) -> tuple[str | None, str | None, str | None]:
    """Parse section header, type ('table' | 'array'), and table name.

    Ignores trailing inline comments, e.g. '[agents] # settings' -> ('[agents]', 'table', 'agents').
    """
    stripped = line.strip()
    if not stripped.startswith("["):
        return None, None, None
    m_array = re.match(r"^\[\[([A-Za-z0-9_.-]+)\]\](?:\s*#.*)?$", stripped)
    if m_array:
        name = m_array.group(1).strip()
        return f"[[{name}]]", "array", name
    m_table = re.match(r"^\[([A-Za-z0-9_.-]+)\](?:\s*#.*)?$", stripped)
    if m_table:
        name = m_table.group(1).strip()
        return f"[{name}]", "table", name
    raise ConfigMergeError(f"Malformed TOML section header: {stripped}")


def _is_multiline_continuation(raw: str) -> bool:
    """Return True if raw line or block has an unclosed string or unclosed bracket."""
    in_single = False
    in_double = False
    in_triple_single = False
    in_triple_double = False
    open_brackets = 0
    i = 0
    n = len(raw)
    while i < n:
        if not in_single and not in_double:
            if not in_triple_single and raw.startswith('"""', i):
                in_triple_double = not in_triple_double
                i += 3
                continue
            if not in_triple_double and raw.startswith("'''", i):
                in_triple_single = not in_triple_single
                i += 3
                continue
        if in_triple_single or in_triple_double:
            i += 1
            continue

        char = raw[i]
        if char == "\\" and in_double:
            i += 2
            continue
        if char == '"' and not in_single:
            in_double = not in_double
        elif char == "'" and not in_double:
            in_single = not in_single
            if char == "#":
                nl = raw.find("\n", i)
                if nl == -1:
                    break
                i = nl
                continue
            if char == "[":
                open_brackets += 1
            elif char == "]":
                open_brackets -= 1
        i += 1

    return (
        in_triple_single
        or in_triple_double
        or in_single
        or in_double
        or open_brackets > 0
    )


def _parse_toml_sections(content: str) -> list[dict[str, Any]]:
    sections: list[dict[str, Any]] = []
    current_header: str | None = None
    current_type: str | None = None
    current_name: str | None = None
    current_lines: list[tuple[bool, str | None, str]] = []
    seen_tables: set[str] = set()

    def _flush_section() -> None:
        nonlocal current_header, current_type, current_name, current_lines
        if current_header is not None or current_lines:
            sections.append(
                {
                    "header": current_header,
                    "header_type": current_type,
                    "name": current_name,
                    "lines": current_lines,
                }
            )
            current_lines = []

    lines = content.splitlines(keepends=True)
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        header, header_type, name = _extract_header(line)
        if header is not None:
            _flush_section()
            if header_type == "table":
                if name in seen_tables:
                    raise ConfigMergeError(f"Duplicate table header in TOML: {name}")
                if name is not None:
                    seen_tables.add(name)
            current_header = header
            current_type = header_type
            current_name = name
            i += 1
            continue

        kv_match = re.match(r"^([A-Za-z0-9_-]+)\s*=", stripped)
        if kv_match:
            key = kv_match.group(1)
            raw = line
            while _is_multiline_continuation(raw) and i + 1 < len(lines):
                i += 1
                raw += lines[i]
            if _is_multiline_continuation(raw):
                raise ConfigMergeError(f"Unclosed multiline value for key: {key}")
            current_lines.append((True, key, raw))
        else:
            current_lines.append((False, None, line))
        i += 1

    _flush_section()
    return sections


def merge_toml(repo_content: str, target_content: str) -> str:
    """Merge TOML repo updates into target machine settings, preserving target keys and comments.

    Args:
        repo_content: Source configuration template from the repository.
        target_content: Existing host configuration to update in-place.

    Returns:
        The merged TOML configuration string ending in a newline.

    Raises:
        ConfigMergeError: If either source or target contains invalid TOML syntax.
    """
    repo_sections = _parse_toml_sections(repo_content)
    target_sections = _parse_toml_sections(target_content)

    repo_root = next((s for s in repo_sections if s["header_type"] is None), None)
    repo_tables = {s["name"]: s for s in repo_sections if s["header_type"] == "table"}
    repo_arrays = [s for s in repo_sections if s["header_type"] == "array"]
    repo_array_names = {a["name"] for a in repo_arrays}

    out_sections: list[dict[str, Any]] = []
    handled_tables: set[str] = set()

    target_has_root = any(s["header_type"] is None for s in target_sections)
    if not target_has_root and repo_root is not None:
        out_sections.append(repo_root)

    for sec in target_sections:
        if sec["header_type"] is None:
            new_lines: list[str] = []
            seen_keys: set[str] = set()
            repo_kv = {
                k: raw
                for is_kv, k, raw in (repo_root["lines"] if repo_root else [])
                if is_kv and k is not None
            }
            for is_kv, key, raw in sec["lines"]:
                if is_kv and key is not None:
                    seen_keys.add(key)
                    new_lines.append(repo_kv.get(key, raw))
                else:
                    new_lines.append(raw)
            if repo_root:
                for is_kv, key, raw in repo_root["lines"]:
                    if is_kv and key is not None and key not in seen_keys:
                        new_lines.append(raw)
            sec["lines"] = [(False, None, line) for line in new_lines]
            out_sections.append(sec)

        elif sec["header_type"] == "table":
            name = sec["name"]
            if name is not None:
                handled_tables.add(name)
            if name in repo_tables:
                repo_table = repo_tables[name]
                repo_kv = {
                    k: raw
                    for is_kv, k, raw in repo_table["lines"]
                    if is_kv and k is not None
                }
                new_lines = []
                seen_keys = set()
                for is_kv, key, raw in sec["lines"]:
                    if is_kv and key is not None:
                        seen_keys.add(key)
                        new_lines.append(repo_kv.get(key, raw))
                    else:
                        new_lines.append(raw)
                for is_kv, key, raw in repo_table["lines"]:
                    if is_kv and key is not None and key not in seen_keys:
                        new_lines.append(raw)
                sec["lines"] = [(False, None, line) for line in new_lines]
            out_sections.append(sec)

        elif sec["header_type"] == "array":
            if sec["name"] not in repo_array_names:
                out_sections.append(sec)

    for name, rsec in repo_tables.items():
        if name not in handled_tables:
            out_sections.append(rsec)

    for rsec in repo_arrays:
        out_sections.append(rsec)

    result: list[str] = []
    for sec in out_sections:
        if sec["header"]:
            result.append(sec["header"] + "\n")
        for _, _, line in sec["lines"]:
            result.append(line if line.endswith("\n") else line + "\n")
    return "".join(result)
