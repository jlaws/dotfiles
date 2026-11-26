"""Pure standard library configuration mergers for JSON and TOML (Python 3.9+)."""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
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
        if (path and path[-1] == "permissions") or (not path and "permissions" in merged):
            deny_set = set(merged["deny"])
            merged["allow"] = [item for item in merged["allow"] if item not in deny_set]

    return merged


_COLON_WILDCARD = re.compile(r"^(Bash\(.+):\*\)$")


def permission_key(entry: Any) -> Any:
    """Canonical form of a permission entry: the legacy ``Bash(cmd:*)`` equals ``Bash(cmd *)``."""
    if isinstance(entry, str):
        match = _COLON_WILDCARD.match(entry)
        if match:
            return f"{match.group(1)} *)"
    return entry


def _retire_json_prompt_hook(data: dict[str, Any]) -> None:
    """Drop only the retired shipped Claude handler, retaining local hook groups."""
    hooks = data.get("hooks")
    if not isinstance(hooks, dict):
        return
    groups = hooks.get("UserPromptSubmit")
    if not isinstance(groups, list):
        return
    kept = []
    for group in groups:
        if not isinstance(group, dict) or not isinstance(group.get("hooks"), list):
            kept.append(group)
            continue
        handlers = group["hooks"]
        remaining = [
            handler
            for handler in handlers
            if not (
                isinstance(handler, dict)
                and handler.get("type") == "command"
                and handler.get("command") == "bash ~/.claude/hooks/log-prompt.sh"
            )
        ]
        if len(remaining) == len(handlers):
            kept.append(group)
        elif remaining:
            kept.append({**group, "hooks": remaining})
    if kept:
        hooks["UserPromptSubmit"] = kept
    else:
        hooks.pop("UserPromptSubmit", None)


def merge_json(
    repo_content: str,
    target_content: str,
    retired: Sequence[str] | str = (),
    *,
    retire_claude_prompt_hook: bool = False,
) -> str:
    """Merge JSON repo updates into target machine settings, preserving target keys.

    Args:
        repo_content: Source configuration template from the repository.
        target_content: Existing host configuration to update in-place.
        retired: ``permissions.allow`` entries dropped from the merged result, a
            hand-added one included.
        retire_claude_prompt_hook: Remove the retired shipped prompt logger handler.

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

    if isinstance(retired, str):
        retired = (retired,)
    if retire_claude_prompt_hook:
        _retire_json_prompt_hook(target_data)
    merged = _merge_dict(repo_data, target_data)
    permissions = merged.get("permissions")
    if retired and isinstance(permissions, dict) and isinstance(permissions.get("allow"), list):
        gone = {permission_key(entry) for entry in retired}
        permissions["allow"] = [
            item for item in permissions["allow"] if permission_key(item) not in gone
        ]
    return json.dumps(merged, indent=2, ensure_ascii=False) + "\n"


# ---------------------------------------------------------------------------
# TOML Merger
# ---------------------------------------------------------------------------


def _validate_toml_table_name(name: str) -> None:
    """Validate that name is composed of valid dot-separated TOML keys (bare or quoted)."""
    idx = 0
    n = len(name)
    segment_count = 0

    while idx < n:
        while idx < n and name[idx].isspace():
            idx += 1
        if idx >= n:
            break

        char = name[idx]
        if char == '"':
            idx += 1
            closed = False
            while idx < n:
                if name[idx] == "\\":
                    idx += 2
                    continue
                if name[idx] == '"':
                    idx += 1
                    closed = True
                    break
                idx += 1
            if not closed:
                raise ConfigMergeError(f"Unclosed double quote in TOML section header: {name}")
            segment_count += 1
        elif char == "'":
            idx += 1
            closed = False
            while idx < n:
                if name[idx] == "'":
                    idx += 1
                    closed = True
                    break
                idx += 1
            if not closed:
                raise ConfigMergeError(f"Unclosed single quote in TOML section header: {name}")
            segment_count += 1
        elif char == ".":
            raise ConfigMergeError(f"Empty key segment in TOML section header: {name}")
        else:
            start = idx
            while idx < n and (name[idx].isalnum() or name[idx] in "_-"):
                idx += 1
            if idx == start:
                raise ConfigMergeError(f"Invalid character '{char}' in TOML section header: {name}")
            segment_count += 1

        while idx < n and name[idx].isspace():
            idx += 1

        if idx < n:
            if name[idx] != ".":
                raise ConfigMergeError(
                    f"Expected '.' separating keys in TOML section header: {name}"
                )
            idx += 1
            if idx >= n or not name[idx:].strip():
                raise ConfigMergeError(f"Trailing dot in TOML section header: {name}")

    if segment_count == 0:
        raise ConfigMergeError(f"Empty TOML section header: {name}")


def _extract_header(line: str) -> tuple[str | None, str | None, str | None]:
    """Parse section header, type ('table' | 'array'), and table name.

    Supports bare keys, quoted keys (such as file paths and IDs),
    and ignores trailing inline comments.
    """
    stripped = line.strip()
    if not stripped.startswith("["):
        return None, None, None

    is_array = stripped.startswith("[[")
    bracket_len = 2 if is_array else 1

    i = bracket_len
    in_double = False
    in_single = False
    escaped = False
    end_idx = -1

    while i < len(stripped):
        char = stripped[i]
        if escaped:
            escaped = False
            i += 1
            continue
        if char == "\\" and in_double:
            escaped = True
            i += 1
            continue
        if char == '"' and not in_single:
            in_double = not in_double
            i += 1
            continue
        if char == "'" and not in_double:
            in_single = not in_single
            i += 1
            continue
        if not in_double and not in_single:
            if is_array:
                if char == "]" and i + 1 < len(stripped) and stripped[i + 1] == "]":
                    end_idx = i
                    break
            else:
                if char == "]":
                    end_idx = i
                    break
        i += 1

    if end_idx == -1 or in_double or in_single:
        raise ConfigMergeError(f"Malformed TOML section header: {stripped}")

    after_header = stripped[end_idx + bracket_len :].strip()
    if after_header and not after_header.startswith("#"):
        raise ConfigMergeError(f"Malformed TOML section header: {stripped}")

    name = stripped[bracket_len:end_idx].strip()
    if not name:
        raise ConfigMergeError(f"Malformed TOML section header: {stripped}")

    _validate_toml_table_name(name)

    remainder = stripped[end_idx + bracket_len :]
    header_str = f"[[{name}]]{remainder}" if is_array else f"[{name}]{remainder}"
    header_type = "array" if is_array else "table"
    return header_str, header_type, name


def _is_multiline_continuation(raw: str) -> bool:
    """Return True if raw line or block has an unclosed string or unclosed bracket."""
    in_single = False
    in_double = False
    in_triple_single = False
    in_triple_double = False
    open_brackets = 0
    open_braces = 0
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
        elif not in_double and not in_single:
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
            elif char == "{":
                open_braces += 1
            elif char == "}":
                open_braces -= 1
        i += 1

    return (
        in_triple_single
        or in_triple_double
        or in_single
        or in_double
        or open_brackets > 0
        or open_braces > 0
    )


_TOML_KEY_PART = r"""(?:[A-Za-z0-9_-]+|"[^"\\]*(?:\\.[^"\\]*)*"|'[^']*')"""
_TOML_KEY_TOKEN = re.compile(_TOML_KEY_PART)
_TOML_ASSIGNMENT = re.compile(rf"^\s*({_TOML_KEY_PART}(?:\s*\.\s*{_TOML_KEY_PART})*)\s*=")


def _toml_key_path(key: str) -> tuple[str, ...]:
    """Decode dotted keys without splitting dots inside quoted literal names."""
    parts = []
    for match in _TOML_KEY_TOKEN.finditer(key):
        token = match.group()
        if token.startswith('"'):
            parts.append(json.loads(token))
        elif token.startswith("'"):
            parts.append(token[1:-1])
        else:
            parts.append(token)
    return tuple(parts)


def _toml_assignment(raw: str) -> tuple[str, str]:
    """Separate an assignment at its unquoted equals sign."""
    match = _TOML_ASSIGNMENT.match(raw)
    if match is None:
        raise ConfigMergeError(f"Malformed TOML assignment: {raw.strip()}")
    return match.group(1), raw[match.end() :].lstrip()


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

        kv_match = _TOML_ASSIGNMENT.match(line)
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


def _toml_string(raw: str) -> str | None:
    """Read a single-line TOML string, leaving unsupported value forms untouched."""
    _, value = _toml_assignment(raw)
    value = value.strip()
    if value.startswith("'") and not value.startswith("'''"):
        end = value.find("'", 1)
        if end >= 0 and (not value[end + 1 :].strip() or value[end + 1 :].lstrip().startswith("#")):
            return value[1:end]
    if value.startswith('"'):
        try:
            parsed, end = json.JSONDecoder().raw_decode(value)
        except ValueError:
            return None
        if isinstance(parsed, str) and (
            not value[end:].strip() or value[end:].lstrip().startswith("#")
        ):
            return parsed
    return None


def _inline_toml_entries(value: str) -> tuple[str, str] | None:
    """Expand an inline table into assignments while retaining nested TOML values."""
    if not value.startswith("{"):
        return None
    end = next(
        (
            index
            for index, char in enumerate(value)
            if char == "}" and not _is_multiline_continuation(value[: index + 1])
        ),
        None,
    )
    if end is None:
        return None
    trailing = value[end + 1 :].strip()
    if trailing and not trailing.startswith("#"):
        return None
    entries = []
    start = 1
    for index in range(1, end):
        if value[index] == "," and not _is_multiline_continuation(value[start:index]):
            entries.append(value[start:index].strip())
            start = index + 1
    entries.append(value[start:end].strip())
    return "\n".join(entry for entry in entries if entry) + "\n", trailing


def _expand_codex_inline_ancestors(
    target_sections: list[dict[str, Any]], path: tuple[str, ...]
) -> None:
    """Open only inline ancestors that block an explicitly managed Codex path."""
    for section in target_sections:
        if section["header_type"] == "array":
            continue
        section_path = _toml_key_path(section["name"]) if section["name"] else ()
        if len(section_path) >= len(path) or section_path != path[: len(section_path)]:
            continue
        lines = []
        for is_key, key, raw in section["lines"]:
            ancestor = section_path + _toml_key_path(key) if is_key and key is not None else ()
            if not ancestor or len(ancestor) >= len(path) or ancestor != path[: len(ancestor)]:
                lines.append((is_key, key, raw))
                continue
            _, value = _toml_assignment(raw)
            inline = _inline_toml_entries(value.strip())
            if inline is None:
                lines.append((is_key, key, raw))
                continue
            entries, comment = inline
            parsed = _parse_toml_sections(entries)
            canonical = ".".join(ancestor)
            target_sections.append(
                {
                    "header": f"[{canonical}]",
                    "header_type": "table",
                    "name": canonical,
                    "lines": parsed[0]["lines"] if parsed else [],
                }
            )
            if comment:
                lines.append((False, None, comment + "\n"))
        section["lines"] = lines


def _normalize_codex_table(
    repo_sections: list[dict[str, Any]],
    target_sections: list[dict[str, Any]],
    path: tuple[str, ...],
    retired: tuple[str, ...] = (),
) -> None:
    """Normalize managed table paths while preserving local values as raw TOML."""
    source = next(
        (
            section
            for section in repo_sections
            if section["header_type"] == "table" and _toml_key_path(section["name"]) == path
        ),
        None,
    )
    if source is None:
        return
    _expand_codex_inline_ancestors(target_sections, path)
    owned = {_toml_key_path(key) for is_key, key, _ in source["lines"] if is_key}
    canonical = ".".join(path)
    existing = None
    promoted: list[tuple[bool, str | None, str]] = []

    def normalized_line(key_path: tuple[str, ...], raw: str) -> tuple[bool, str | None, str]:
        key = ".".join(
            part if re.fullmatch(r"[A-Za-z0-9_-]+", part) else json.dumps(part) for part in key_path
        )
        _, value = _toml_assignment(raw)
        return True, key, f"{key} = {value}"

    for section in target_sections:
        if section["header_type"] == "array":
            continue
        section_path = _toml_key_path(section["name"]) if section["name"] else ()
        if len(section_path) > len(path):
            continue
        if section_path == path:
            old_name = section["name"]
            section["name"] = canonical
            section["header"] = f"[{canonical}]" + section["header"][len(old_name) + 2 :]
            existing = section
        lines = []
        for is_key, key, raw in section["lines"]:
            if not is_key or key is None:
                lines.append((is_key, key, raw))
                continue
            key_path = _toml_key_path(key)
            full_path = section_path + key_path
            if full_path[: len(path)] != path:
                lines.append((is_key, key, raw))
                continue
            relative = full_path[len(path) :]
            if len(relative) == 1 and relative[0] in retired:
                continue
            if not relative:
                _, value = _toml_assignment(raw)
                value = value.strip()
                boolean = re.fullmatch(r"(?:true|false)\s*(#.*)?", value)
                inline = _inline_toml_entries(value)
                if boolean:
                    comment = boolean.group(1)
                elif inline is not None:
                    entries, comment = inline
                    for parsed in _parse_toml_sections(entries):
                        for entry_key, entry_name, entry_raw in parsed["lines"]:
                            if entry_key and entry_name is not None:
                                entry_path = _toml_key_path(entry_name)
                                if entry_path not in {(name,) for name in retired}:
                                    promoted.append(normalized_line(entry_path, entry_raw))
                else:
                    lines.append((is_key, key, raw))
                    continue
                if comment:
                    lines.append((False, None, comment + "\n"))
            elif section_path == path:
                lines.append(
                    normalized_line(key_path, raw) if key_path in owned else (is_key, key, raw)
                )
            else:
                promoted.append(normalized_line(relative, raw))
        section["lines"] = lines
    if not promoted:
        return
    if existing is not None:
        existing_keys = {_toml_key_path(key) for is_key, key, _ in existing["lines"] if is_key}
        existing["lines"].extend(
            line
            for line in promoted
            if line[1] is None or _toml_key_path(line[1]) not in existing_keys
        )
    else:
        target_sections.append(
            {
                "header": f"[{canonical}]",
                "header_type": "table",
                "name": canonical,
                "lines": promoted,
            }
        )


def _migrate_codex_sections(
    repo_sections: list[dict[str, Any]], target_sections: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Retire shipped Codex settings without replacing user hook arrays."""
    _normalize_codex_table(repo_sections, target_sections, ("features", "multi_agent_v2"))
    canonical_agents = any(
        section["name"] == "agents"
        and any(
            is_key and key == "max_concurrent_threads_per_session"
            for is_key, key, _ in section["lines"]
        )
        for section in repo_sections
    )
    _normalize_codex_table(
        repo_sections,
        target_sections,
        ("agents",),
        ("max_threads", "max_depth") if canonical_agents else (),
    )
    kept: list[dict[str, Any]] = []

    def preserve_comments(section: dict[str, Any]) -> None:
        comments = [line for line in section["lines"] if not line[0]]
        if not comments:
            return
        if not kept:
            kept.append({"header": None, "header_type": None, "name": None, "lines": []})
        kept[-1]["lines"].extend(comments)

    parents: dict[int, list[int]] = {}
    parent: int | None = None
    removed: set[int] = set()
    for index, section in enumerate(target_sections):
        if section["header_type"] == "array" and section["name"] == "hooks.UserPromptSubmit":
            parent = index
            parents[parent] = []
            continue
        name = section["name"]
        if (
            parent is not None
            and isinstance(name, str)
            and name.startswith("hooks.UserPromptSubmit.")
        ):
            parents[parent].append(index)
            values = {key: _toml_string(raw) for is_key, key, raw in section["lines"] if is_key}
            if (
                section["header_type"] == "array"
                and name == "hooks.UserPromptSubmit.hooks"
                and values.get("type") == "command"
                and values.get("command") == "bash ~/.codex/hooks/log-prompt.sh"
            ):
                removed.add(index)
    for parent, children in parents.items():
        if children and all(child in removed for child in children):
            removed.add(parent)
    for index, section in enumerate(target_sections):
        if index in removed:
            preserve_comments(section)
        else:
            kept.append(section)
    return kept


def merge_toml(repo_content: str, target_content: str, *, migrate_codex: bool = False) -> str:
    """Merge TOML repo updates into target machine settings, preserving target keys and comments.

    Args:
        repo_content: Source configuration template from the repository.
        target_content: Existing host configuration to update in-place.
        migrate_codex: Retire the shipped prompt logger and superseded agent limits.

    Returns:
        The merged TOML configuration string ending in a newline.

    Raises:
        ConfigMergeError: If either source or target contains invalid TOML syntax.
    """
    repo_sections = _parse_toml_sections(repo_content)
    target_sections = _parse_toml_sections(target_content)
    if migrate_codex:
        target_sections = _migrate_codex_sections(repo_sections, target_sections)

    repo_root = next((s for s in repo_sections if s["header_type"] is None), None)
    repo_tables = {s["name"]: s for s in repo_sections if s["header_type"] == "table"}
    repo_arrays = [s for s in repo_sections if s["header_type"] == "array"]
    repo_array_names = {a["name"] for a in repo_arrays}

    out_sections: list[dict[str, Any]] = []
    handled_tables: set[str] = set()

    target_has_root = any(s["header_type"] is None for s in target_sections)
    if not target_has_root and repo_root is not None:
        out_sections.append(repo_root)

    replaced_repo_arrays_inserted = False

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
                    k: raw for is_kv, k, raw in repo_table["lines"] if is_kv and k is not None
                }
                trailing_comments: list[tuple[bool, str | None, str]] = []
                while sec["lines"] and not sec["lines"][-1][0]:
                    trailing_comments.insert(0, sec["lines"].pop())

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
                for _, _, raw in trailing_comments:
                    new_lines.append(raw)
                sec["lines"] = [(False, None, line) for line in new_lines]
            out_sections.append(sec)

        elif sec["header_type"] == "array":
            if sec["name"] in repo_array_names:
                if not replaced_repo_arrays_inserted:
                    out_sections.extend(repo_arrays)
                    replaced_repo_arrays_inserted = True
            else:
                out_sections.append(sec)

    if not replaced_repo_arrays_inserted:
        for rsec in repo_arrays:
            out_sections.append(rsec)

    for name, rsec in repo_tables.items():
        if name not in handled_tables:
            out_sections.append(rsec)

    result: list[str] = []
    for sec in out_sections:
        if sec["header"]:
            result.append(sec["header"] + "\n")
        for _, _, line in sec["lines"]:
            result.append(line if line.endswith("\n") else line + "\n")
    return "".join(result)
