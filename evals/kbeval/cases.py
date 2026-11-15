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
        # commands
        "j-arch", "j-audit", "j-brainstorm", "j-cloud", "j-config-audit", "j-create-pr", "j-data",
        "j-debug", "j-defrag", "j-devops", "j-diff-review", "j-doc-sync", "j-docs",
        "j-email-analysis", "j-email-consolidate", "j-execute-plan", "j-explain", "j-frontend",
        "j-init", "j-lang", "j-new", "j-next", "j-observe", "j-paper-analysis", "j-paper",
        "j-plan", "j-pr-fix", "j-rebase", "j-retro", "j-review-claudemd", "j-skill-audit",
        "j-system-design", "j-tdd",
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
FORBIDDEN_FIXTURE = re.compile(
    r"\b(curl|wget|ssh|scp|nc|sudo|brew|eval|git\s+clone|npm\s+(?:install|i|ci)\b|pip3?\s+install"
    r"|rm\s+-[a-z]*r|(?:ba)?sh\s+-c|python3?\s+-c|git\s+remote\s+add\s+\S+\s+(?:https?:|ssh:|git@))"
)
PARENT_PATH = re.compile(r"(?:^|[/\s'\"=])\.\.(?:/|\s|$|['\"])")
# JavaScript regex has no inline flag groups like (?i) and no \A or \Z anchors.
JS_UNSUPPORTED = re.compile(r"\(\?[a-zA-Z]+\)|\(\?[a-zA-Z]+:|\\[AZ]")
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


def malformed_scalar(value: str) -> bool:
    """Return True for a quoted YAML scalar the CLI's parser will reject."""
    value = value.strip()
    if len(value) >= 2 and value[0] == "'":
        return value[-1] != "'" or "'" in value[1:-1].replace("''", "")
    if len(value) >= 2 and value[0] == '"':
        inner = re.sub(r'\\.', "", value[1:-1])
        return value[-1] != '"' or '"' in inner
    return value in ("'", '"')


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
        problems.extend(f"{case.name}: {p}" for p in _lint_case(case, known))
    extended: Dict[str, int] = {}
    for case in cases:
        if case.asset and "extended" in case.tags:
            extended[case.asset] = extended.get(case.asset, 0) + 1
    for asset, count in sorted(extended.items()):
        if count > 1:
            problems.append(f"{asset}: {count} cases tagged extended; at most one")
    return problems


def _lint_case(case: Case, known: Dict[str, str]) -> List[str]:
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
        pattern = unquote(grader.fields.get("pattern", ""))
        if JS_UNSUPPORTED.search(pattern):
            problems.append(f"grader {grader.name}: pattern uses inline flags or \\A/\\Z; JS regex needs `flags:` and ^/$")
        graded = " ".join(grader.fields.get(k, "") for k in ("target", "focus", "path"))
        if re.search(r"(?:path:\s*|^)['\"]?\.claude/", graded) or unquote(grader.fields.get("path", "")).startswith(".claude/"):
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
        if gtype in RESULT_TYPES and "kb:" in grader.text:
            problems.append(f"grader {grader.name}: result graders never mention kb:")
        if gtype == "llm":
            criteria = grader.body or unquote(grader.fields.get("criteria", ""))
            if "PASS if" not in criteria or "FAIL if" not in criteria:
                problems.append(f"grader {grader.name}: llm criteria need PASS if and FAIL if claims")

    if case.suite == "triggers":
        problems.extend(_lint_trigger_grader(case))
        return problems
    types = [unquote(g.fields.get("type", "")) for g in case.graders]
    if not any(t in RESULT_TYPES for t in types):
        problems.append("needs at least one result grader (regex, file_exists, or llm)")
    if not any(t in PROCESS_TYPES for t in types):
        problems.append("needs at least one process grader (tool_used or tool_order)")
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
        for number, line in enumerate(case.fixture.splitlines(), 1):
            if line.strip().startswith("#"):
                continue
            code = line
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
