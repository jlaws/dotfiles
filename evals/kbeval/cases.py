"""Discover, lint, and check coverage of eval cases.

Cases live under `evals/cases/<suite>/<name>/` as `prompt.md`, `graders/*.md`, and optionally
`case.yaml` and `fixture.sh`. The rules enforced here are the case-authoring rules in
`evals/README.md`; that file owns their wording.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

SUITES = ("triggers", "effects", "workflows")
KINDS = ("trigger-should", "trigger-near", "trigger-noauto", "effect")
TRIGGER_KINDS = KINDS[:3]

# Assets whose cases are not written yet. Shrinks with each case PR and is deleted, with its
# exemption branch in `coverage`, once every asset has cases.
PENDING_EVAL_COVERAGE = frozenset(
    {
        # agents
        "architecture-specialist", "business-analyst", "cloud-architect", "code-reviewer",
        "create-pr", "data-engineer", "devops-engineer", "documentation-writer",
        "frontend-engineer", "language-specialist", "ml-engineer", "product-manager",
        "research-analyst", "scope-reviewer", "security-reviewer", "test-writer",
        "workflow-architect",
    }
)  # fmt: skip

PROMPT_KEYS = frozenset(
    {"description", "tags", "runs", "expected_outcome", "max_turns", "timeout_seconds",
     "allowed_tools", "env"}
)  # fmt: skip
GRADER_KEYS = {
    "regex": {"pattern", "flags", "match", "target"},
    "tool_used": {"tool", "input_match", "min", "max"},
    "tool_order": {"before", "after"},
    "file_exists": {"path", "exists"},
    "llm": {"criteria", "focus"},
}
COMMON_GRADER_KEYS = {"type", "weight", "arm"}
RESULT_TYPES = frozenset({"regex", "file_exists", "llm"})
PROCESS_TYPES = frozenset({"tool_used", "tool_order"})

TRIGGER_TOOLS = ["Skill", "Read", "Glob", "Grep"]
EFFECT_TOOLS = ["Skill", "Read", "Glob", "Grep", "Agent", "TodoWrite"]
SUITE_LIMITS = {
    "triggers": (6, 120, TRIGGER_TOOLS),
    "effects": (40, 900, EFFECT_TOOLS),
    "workflows": (150, 3600, EFFECT_TOOLS),
}

# A guard rail against mistakes in reviewed, in-repo cases, not a security boundary.
# Whole command words only (`ncurses` and `evals` are fine), plus their common variants; the
# multi-word forms carry their own boundaries.
FORBIDDEN_FIXTURE = re.compile(
    r"\b(?:curl|wget2?|ssh(?:-[a-z]+)?|scp|nc|ncat|sudo|brew|eval)(?![\w-])"
    r"|\bgit\s+(?:-\S+\s+)*clone\b|\bnpm\s+(?:install|i|ci)\b|\bpip3?\s+install\b"
    r"|\brm\s+(?:-\w+\s+)*(?:-[a-zA-Z]*[rR][a-zA-Z]*|--recursive)\b"
    r"|\b(?:ba|z|da)?sh\s+-[a-z]*c\b|\b(?:python3?|node|perl|ruby)\s+(?:-\w+\s+)*-[A-Za-z]*[ce]\b"
    r"|\bgit\s+config\s+--(?:global|system)\b"
    r"|\bgit\s+(?:-\S+\s+)*push\b[^\n]*(?:\s-f\b|--force|\s\+\S)"
    r"|\bgit\s+remote\s+add\s+\S+\s+(?:https?:|ssh:|git@)"
)
HEREDOC_START = re.compile(r"<<-?\s*['\"]?(\w+)['\"]?")
PARENT_PATH = re.compile(r"(?:^|[/\s'\"=])\.\.(?:/|\s|$|['\"])")
# JavaScript regex has no inline flag groups like (?i), no Python named groups (?P<x>), no POSIX
# classes, and no \A, \Z, or \z anchors. An escaped backslash (\\A) is a literal and passes.
JS_UNSUPPORTED = re.compile(r"\(\?[a-zA-Z]+\)|\(\?[a-zA-Z]+:|\(\?P<|\[\[:|(?<!\\)(?:\\\\)*\\[AZz]")
JS_FLAGS = frozenset("dgimsuvy")
YAML_ESCAPES = frozenset('0abtnvfre "/\\N_LPxuU\t')
REQUIRED_KEYS = {
    "regex": ("pattern", "target"),
    "tool_used": ("tool",),
    "tool_order": ("before", "after"),
    "file_exists": ("path",),
    "llm": (),
}
CLAIM = re.compile(r"(?m)^(?:PASS|FAIL) if ")
CONTEXT_LINE = re.compile(r"^  (?:add_dirs: \[[^\]]*\]|history_file: \S+)$")
ABSOLUTE_PATH = re.compile(r"""(?:^|[\s'"=])(~|/[A-Za-z])""")
# `$1` and any path under it, quoted or not, which is the one place a fixture may read from.
FIXTURE_ARG = re.compile(r'"?\$\{?1\}?[^\s"]*"?')
FRONTMATTER = re.compile(r"\A---\n(.*?)\n---\n?", re.DOTALL)
CASE_NAME = re.compile(r"^(?P<asset>[a-z0-9-]+?)--(?P<kind>" + "|".join(KINDS) + r")-[a-z0-9-]+$")
WORKFLOW_NAME = re.compile(r"^wf--[a-z0-9-]+$")


@dataclass
class Grader:
    """One `graders/*.md` file: its frontmatter fields and its body."""

    name: str
    fields: Dict[str, str]
    body: str
    text: str


@dataclass
class Case:
    """One case directory and what lint needs from it."""

    suite: str
    name: str
    path: Path
    asset: Optional[str]
    kind: Optional[str]
    fields: Dict[str, str]
    body: str
    graders: List[Grader] = field(default_factory=list)
    fixture: Optional[str] = None
    case_yaml: Optional[str] = None

    @property
    def tags(self) -> List[str]:
        """Return the case's tags."""
        return parse_list(self.fields.get("tags", "[]"))


def split_frontmatter(text: str) -> Tuple[Dict[str, str], str]:
    """Return a flat `key: raw value` map of the frontmatter, and the body.

    A key with an empty value takes its indented continuation lines as its raw value, which is
    enough for the one-level block maps graders use (`target:` then `  source: file`).
    """
    match = FRONTMATTER.match(text)
    if match is None:
        return {}, text.strip()
    fields: Dict[str, str] = {}
    key: Optional[str] = None
    for line in match.group(1).splitlines():
        if line.startswith((" ", "\t")) and key is not None:
            fields[key] = (fields[key] + "\n" + line.strip()).strip()
            continue
        if ":" in line:
            key, _, value = line.partition(":")
            key = key.strip()
            fields[key] = value.strip()
    return fields, text[match.end() :].strip()


def unquote(value: str) -> str:
    """Strip one layer of YAML quotes from a scalar."""
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] == "'":
        return value[1:-1].replace("''", "'")
    if len(value) >= 2 and value[0] == value[-1] == '"':
        return value[1:-1].replace('\\"', '"').replace("\\\\", "\\")
    return value


def _flow_items(value: str) -> List[Tuple[str, str]]:
    """Split a one-line YAML flow map `{ a: x, b: 'y' }` into (key, raw value) pairs."""
    inner = value.strip()[1:-1]
    items, current, quote = [], "", ""
    for char in inner:
        if quote:
            current += char
            if char == quote:
                quote = ""
            continue
        if char in "'\"":
            quote = char
        if char == "," and not quote:
            items.append(current)
            current = ""
            continue
        current += char
    if current.strip():
        items.append(current)
    pairs = []
    for item in items:
        key, _, raw = item.partition(":")
        pairs.append((key.strip(), raw.strip()))
    return pairs


def scalar_values(value: str) -> List[str]:
    """Return the raw scalars in a value: itself, or each value of a flow map."""
    value = value.strip()
    if value.startswith("{") and value.endswith("}"):
        return [raw for _, raw in _flow_items(value)]
    return [value]


def malformed_scalar(value: str) -> bool:
    """Return True for a quoted YAML scalar the CLI's parser will reject."""
    for raw in scalar_values(value):
        raw = raw.strip()
        if raw in ("'", '"'):
            return True
        if len(raw) >= 2 and raw[0] == "'":
            if raw[-1] != "'" or "'" in raw[1:-1].replace("''", ""):
                return True
        elif len(raw) >= 2 and raw[0] == '"':
            inner = raw[1:-1]
            if raw[-1] != '"':
                return True
            index = 0
            while index < len(inner):
                if inner[index] == "\\":
                    if index + 1 >= len(inner) or inner[index + 1] not in YAML_ESCAPES:
                        return True
                    index += 2
                    continue
                if inner[index] == '"':
                    return True
                index += 1
    return False


def graded_path(value: str) -> Optional[str]:
    """Return the `path` of a `{ source: file, path: x }` target or focus, if any."""
    if value.strip().startswith("{"):
        for key, raw in _flow_items(value):
            if key == "path":
                return unquote(raw)
    return None


def parse_list(value: str) -> List[str]:
    """Parse a YAML flow list of scalars, such as `[a, b]`."""
    value = value.strip()
    if not (value.startswith("[") and value.endswith("]")):
        return [unquote(value)] if value else []
    return [unquote(item) for item in value[1:-1].split(",") if item.strip()]


def parse_int(value: Optional[str]) -> Optional[int]:
    """Return an integer scalar, or None when absent or not an integer."""
    if value is None:
        return None
    try:
        return int(unquote(value))
    except ValueError:
        return None


def assets(repo: Path) -> Dict[str, str]:
    """Map every Claude skill, command, and agent name to its kind."""
    claude = repo / ".claude"
    found: Dict[str, str] = {}
    for skill in sorted((claude / "skills").glob("*/SKILL.md")):
        found[skill.parent.name] = "skill"
    for command in sorted((claude / "commands").glob("*.md")):
        fields, _ = split_frontmatter(command.read_text(encoding="utf-8"))
        user_only = unquote(fields.get("disable-model-invocation", "")) == "true"
        found[command.stem] = "user-command" if user_only else "command"
    for agent in sorted((claude / "agents").glob("*.md")):
        found[agent.stem] = "agent"
    return found


def discover(cases_root: Path) -> List[Case]:
    """Return every case under `cases_root`, sorted by suite then name."""
    found: List[Case] = []
    for suite in SUITES:
        root = cases_root / suite
        if not root.is_dir():
            continue
        for folder in sorted(p for p in root.iterdir() if p.is_dir()):
            prompt = folder / "prompt.md"
            fields, body = split_frontmatter(prompt.read_text(encoding="utf-8")) if prompt.is_file() else ({}, "")
            match = CASE_NAME.match(folder.name)
            graders = []
            for grader in sorted((folder / "graders").glob("*.md")):
                text = grader.read_text(encoding="utf-8")
                gfields, gbody = split_frontmatter(text)
                graders.append(Grader(grader.stem, gfields, gbody, text))
            fixture = folder / "fixture.sh"
            case_yaml = folder / "case.yaml"
            found.append(
                Case(
                    suite=suite,
                    name=folder.name,
                    path=folder,
                    asset=match.group("asset") if match else None,
                    kind=match.group("kind") if match else None,
                    fields=fields,
                    body=body,
                    graders=graders,
                    fixture=fixture.read_text(encoding="utf-8") if fixture.is_file() else None,
                    case_yaml=case_yaml.read_text(encoding="utf-8") if case_yaml.is_file() else None,
                )
            )
    return found


def lint(cases: List[Case], repo: Path) -> List[str]:
    """Return one `<case>: <problem>` string per authoring-rule violation."""
    known = assets(repo)
    problems: List[str] = []
    for case in cases:
        problems.extend(f"{case.name}: {p}" for p in _lint_case(case, known, repo))
    extended: Dict[str, int] = {}
    for case in cases:
        if case.asset and "extended" in case.tags:
            extended[case.asset] = extended.get(case.asset, 0) + 1
    for asset, count in sorted(extended.items()):
        if count > 1:
            problems.append(f"{asset}: {count} cases tagged extended; at most one")
    return problems


def _lint_case(case: Case, known: Dict[str, str], repo: Path) -> List[str]:
    problems: List[str] = []
    if not (case.path / "prompt.md").is_file():
        return ["no prompt.md"]
    if case.suite == "workflows":
        if not WORKFLOW_NAME.match(case.name):
            problems.append("workflow case names start with wf--")
    elif case.asset is None or case.kind is None:
        return ["name is not <asset>--<kind>-<slug>"]
    elif case.asset not in known:
        return [f"name prefix {case.asset!r} is not a known skill, command, or agent asset"]
    elif (case.suite == "triggers") != (case.kind in TRIGGER_KINDS):
        problems.append(f"kind {case.kind} does not belong in {case.suite}")

    problems.extend(_lint_prompt(case, known))
    problems.extend(_lint_graders(case, known))
    problems.extend(_lint_files(case))
    problems.extend(_lint_grounding(case, repo))
    for stray in sorted(p.name for p in (case.path / "graders").glob("*") if p.is_file() and p.suffix != ".md"):
        problems.append(f"graders/{stray} is not a .md grader and would be ignored")
    return problems


def _lint_prompt(case: Case, known: Dict[str, str]) -> List[str]:
    problems: List[str] = []
    for key in sorted(set(case.fields) - PROMPT_KEYS):
        problems.append(f"prompt.md frontmatter key {key!r} is not allowed")
    turns_cap, timeout_cap, tools = SUITE_LIMITS[case.suite]
    if parse_int(case.fields.get("max_turns")) != turns_cap:
        problems.append(f"max_turns must be {turns_cap} in {case.suite}")
    if parse_int(case.fields.get("timeout_seconds")) != timeout_cap:
        problems.append(f"timeout_seconds must be {timeout_cap} in {case.suite}")
    allowed = parse_list(case.fields.get("allowed_tools", "[]"))
    for web in ("WebFetch", "WebSearch"):
        if web in allowed:
            problems.append(f"allowed_tools must not list {web}")
    if allowed != tools:
        problems.append(f"allowed_tools must be [{', '.join(tools)}] in {case.suite}")
    runs = parse_int(case.fields.get("runs"))
    if "runs" in case.fields and (runs is None or not 1 <= runs <= 50):
        problems.append("runs must be 1-50")
    env = case.fields.get("env", "{}").strip("{} ")
    for item in filter(None, (part.strip() for part in env.split(","))):
        if not re.match(r"^EVAL_[A-Z0-9_]*\s*:", item):
            problems.append(f"env key in {item!r} must match EVAL_[A-Z0-9_]*")
    expected = [case.suite] + ([case.kind] if case.kind and case.suite != "workflows" else [])
    if any(tag not in case.tags for tag in expected):
        problems.append(f"tags must include {', '.join(expected)}")

    kind = known.get(case.asset or "")
    body = case.body
    effect = case.suite == "workflows" or case.kind == "effect"
    if case.suite == "workflows":
        if not body.startswith("/kb:"):
            problems.append("workflow prompts start with /kb:<command> ")
    elif effect and kind in ("command", "user-command"):
        if not body.startswith(f"/kb:{case.asset} "):
            problems.append(f"command effect prompts start with '/kb:{case.asset} '")
    elif effect and kind == "agent":
        if not body.startswith(f"Use the kb:{case.asset} agent to "):
            problems.append(f"agent effect prompts start with 'Use the kb:{case.asset} agent to '")
    elif case.asset and re.search(rf"(?<![\w-]){re.escape(case.asset)}(?![\w-])", body):
        problems.append("prompt names the skill or command; write it as a user request")
    elif "kb:" in body:
        problems.append("trigger and skill prompts never mention kb:")
    return problems


def _lint_graders(case: Case, known: Dict[str, str]) -> List[str]:
    problems: List[str] = []
    if not case.graders:
        return ["no graders"]
    for grader in case.graders:
        for key, value in grader.fields.items():
            if malformed_scalar(value):
                problems.append(f"grader {grader.name}: `{key}` is a malformed quoted YAML scalar")
        regexes = [unquote(grader.fields.get("pattern", "")), unquote(grader.fields.get("input_match", ""))]
        for key in ("before", "after"):
            regexes += [unquote(raw) for k, raw in _flow_items(grader.fields.get(key, "{}") or "{}") if k == "input_match"]
        if any(JS_UNSUPPORTED.search(rx) for rx in regexes if rx):
            problems.append(
                f"grader {grader.name}: regex uses inline flags, (?P<, POSIX classes, or \\A/\\Z/\\z; "
                "JS regex needs `flags:` and ^/$"
            )
        bad_flags = set(unquote(grader.fields.get("flags", ""))) - JS_FLAGS
        if bad_flags:
            problems.append(f"grader {grader.name}: flags {''.join(sorted(bad_flags))!r} are not JS regex flags")
        for path in _grader_paths(grader):
            clean = path[2:] if path.startswith("./") else path
            if clean == ".claude" or clean.startswith(".claude/"):
                problems.append(f"grader {grader.name}: grades a file under .claude/, which runs cannot write")
        if grader.name.endswith(".md"):
            problems.append(f"grader {grader.name}.md: file name has a double .md extension")
        gtype = unquote(grader.fields.get("type", ""))
        if gtype not in GRADER_KEYS:
            problems.append(f"grader {grader.name}: type {gtype!r} is not allowed")
            continue
        extra = set(grader.fields) - GRADER_KEYS[gtype] - COMMON_GRADER_KEYS
        for key in sorted(extra):
            problems.append(f"grader {grader.name}: key {key!r} is not allowed for {gtype}")
        for key in REQUIRED_KEYS[gtype]:
            if not unquote(grader.fields.get(key, "")):
                problems.append(f"grader {grader.name}: {gtype} needs `{key}`")
        if gtype == "regex" and unquote(grader.fields.get("pattern", "x")) in ("", ".*", ".+", "^", "$"):
            problems.append(f"grader {grader.name}: pattern matches everything")
        if gtype in RESULT_TYPES and "kb:" in grader.text:
            problems.append(f"grader {grader.name}: result graders never mention kb:")
        if gtype == "llm":
            criteria = grader.body or unquote(grader.fields.get("criteria", ""))
            if "PASS if" not in criteria or "FAIL if" not in criteria:
                problems.append(f"grader {grader.name}: llm criteria need PASS if and FAIL if claims")
            claims = len(CLAIM.findall(criteria))
            if not 2 <= claims <= 5:
                problems.append(f"grader {grader.name}: llm criteria hold {claims} claims; use 2-5")

    if case.suite == "triggers":
        problems.extend(_lint_trigger_grader(case))
        return problems
    types = [unquote(g.fields.get("type", "")) for g in case.graders]
    if not any(t in RESULT_TYPES for t in types):
        problems.append("needs at least one result grader (regex, file_exists, or llm)")
    if not any(t in PROCESS_TYPES for t in types):
        problems.append("needs at least one process grader (tool_used or tool_order)")
    return problems


def _grader_paths(grader: Grader) -> List[str]:
    paths = [graded_path(grader.fields.get(k, "")) for k in ("target", "focus")]
    if unquote(grader.fields.get("type", "")) == "file_exists":
        paths.append(unquote(grader.fields.get("path", "")))
    return [p for p in paths if p]


def _fixture_names(case: Case) -> List[str]:
    return re.findall(r'"\$1/([\w.-]+)/\."', case.fixture or "")


def _initial_file(case: Case, repo: Path, path: str) -> Optional[str]:
    """Return a file's content in the workspace before the agent acts, or None if absent."""
    for name in _fixture_names(case):
        source = repo / "evals" / "fixtures" / name / path
        if source.is_file():
            return source.read_text(encoding="utf-8", errors="ignore")
    return None


def _python_flags(flags: str) -> int:
    value = 0
    for letter, flag in (("i", re.IGNORECASE), ("m", re.MULTILINE), ("s", re.DOTALL)):
        if letter in flags:
            value |= flag
    return value


def _lint_grounding(case: Case, repo: Path) -> List[str]:
    """Graded paths must be named in the prompt or seeded; result graders must not pass on the seed."""
    problems: List[str] = []
    if case.suite == "triggers":
        return problems
    for grader in case.graders:
        gtype = unquote(grader.fields.get("type", ""))
        for path in _grader_paths(grader):
            if path not in case.body and _initial_file(case, repo, path) is None:
                problems.append(f"grader {grader.name}: graded path {path!r} is neither named in the prompt nor seeded")
        if gtype == "file_exists" and unquote(grader.fields.get("exists", "true")) == "false":
            problems.append(f"grader {grader.name}: `exists: false` passes on a run that does nothing")
        positive_needed = gtype == "file_exists" or unquote(grader.fields.get("match", "")) == "not_contains"
        if positive_needed:
            path = _grader_paths(grader)[0] if _grader_paths(grader) else None
            others = [
                g for g in case.graders
                if g is not grader and path in _grader_paths(g)
                and unquote(g.fields.get("type", "")) in ("regex", "llm")
                and unquote(g.fields.get("match", "")) != "not_contains"
            ]
            if path and not others:
                problems.append(
                    f"grader {grader.name}: needs a positive regex or llm grader on {path}; alone it "
                    "cannot tell a correct file from an empty or missing one"
                )
        if gtype == "regex":
            path = graded_path(grader.fields.get("target", ""))
            if not path:
                continue
            initial = _initial_file(case, repo, path)
            if initial is None:
                continue
            try:
                found = re.search(unquote(grader.fields.get("pattern", "")), initial,
                                  _python_flags(unquote(grader.fields.get("flags", ""))))
            except re.error:
                continue
            inverted = unquote(grader.fields.get("match", "")) == "not_contains"
            if bool(found) != inverted:
                problems.append(f"grader {grader.name}: already passes on the seeded {path}; it scores a run that does nothing")
    return problems


def _lint_trigger_grader(case: Case) -> List[str]:
    if len(case.graders) != 1:
        return ["trigger cases need exactly one grader: tool_used on Skill"]
    grader = case.graders[0]
    fields = grader.fields
    problems: List[str] = []
    if unquote(fields.get("type", "")) != "tool_used" or unquote(fields.get("tool", "")) != "Skill":
        problems.append("trigger grader must be tool_used on Skill")
    expected = '"skill"\\s*:\\s*"(?:kb:)?' + (case.asset or "") + '"'
    if unquote(fields.get("input_match", "")) != expected:
        problems.append(f"trigger grader input_match must be '{expected}'")
    low, high = parse_int(fields.get("min")), parse_int(fields.get("max"))
    if case.kind == "trigger-should" and (low != 1 or "max" in fields):
        problems.append("trigger-should grader needs min: 1 and no max")
    if case.kind in ("trigger-near", "trigger-noauto") and (low != 0 or high != 0):
        problems.append(f"{case.kind} grader needs min: 0 and max: 0")
    return problems


def _lint_files(case: Case) -> List[str]:
    problems: List[str] = []
    for path in sorted(p for p in case.path.rglob("*") if p.is_file()):
        try:
            if "sk-ant-" in path.read_text(encoding="utf-8"):
                problems.append(f"{path.name} contains an sk-ant- string")
        except UnicodeDecodeError:
            continue
    if case.case_yaml is not None:
        if "scaffold_script" in case.case_yaml:
            problems.append("case.yaml must not set scaffold_script; the builder owns it")
        lines = [ln for ln in case.case_yaml.splitlines() if ln.strip()]
        if not lines or lines[0] != "context:" or not all(CONTEXT_LINE.match(ln) for ln in lines[1:]):
            problems.append("case.yaml holds only `context:` with `  add_dirs: [..]` or `  history_file: x`")
    if case.fixture is not None:
        code_lines = [ln.strip() for ln in case.fixture.splitlines() if ln.strip() and not ln.strip().startswith("#")]
        if not code_lines or code_lines[0] != "set -euo pipefail":
            problems.append("fixture.sh must start with `set -euo pipefail`")
        heredoc_end: Optional[str] = None
        for number, line in enumerate(case.fixture.splitlines(), 1):
            if heredoc_end is not None:  # seeded file content, not a command
                if line.strip() == heredoc_end:
                    heredoc_end = None
                continue
            if line.strip().startswith("#"):
                continue
            code = line
            start = HEREDOC_START.search(code)
            if start:
                heredoc_end = start.group(1)
            if PARENT_PATH.search(code):
                problems.append(f"fixture.sh:{number} uses a .. path")
            if FORBIDDEN_FIXTURE.search(code):
                problems.append(f"fixture.sh:{number} runs a forbidden command")
            elif ABSOLUTE_PATH.search(FIXTURE_ARG.sub("", code)):
                problems.append(f"fixture.sh:{number} touches a path outside $1 and the workspace")
    return problems


def coverage(cases: List[Case], repo: Path, exempt: frozenset) -> List[str]:
    """Return one `<asset>: <gap>` string per asset missing required cases."""
    counts: Dict[str, Dict[str, int]] = {}
    for case in cases:
        if case.asset and case.kind:
            per = counts.setdefault(case.asset, {})
            per[case.kind] = per.get(case.kind, 0) + 1
    gaps: List[str] = []
    for asset, kind in sorted(assets(repo).items()):
        if asset in exempt:
            continue
        have = counts.get(asset, {})
        if kind in ("skill", "command"):
            need = {"trigger-should": 1, "trigger-near": 1, "effect": 2}
        elif kind == "user-command":
            need = {"trigger-noauto": 1, "effect": 2}
        else:
            need = {"effect": 2}
        for case_kind, minimum in need.items():
            if have.get(case_kind, 0) < minimum:
                gaps.append(f"{asset}: needs {minimum} {case_kind} case(s), has {have.get(case_kind, 0)}")
    return gaps
