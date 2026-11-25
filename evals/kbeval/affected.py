"""Map a diff to the eval assets it touches, and project what a one-run smoke would cost.

`affected` is pure: it reads files under `repo` and never runs git. The command line wrapper in
`__main__` supplies the changes and prints the result.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Tuple

from evals.kbeval import cases as kcases
from evals.kbeval.orchestrate import VANILLA_SAMPLE

SMOKE_USD = {"triggers": 0.07, "effects": 0.29, "workflows": 1.00}
VANILLA_EFFECT_USD = 0.19
EVERYTHING = (".claude/CLAUDE.md", "evals/harness.json", "evals/kbeval/stage.py")
UNMEASURED = (".agents/", ".codex/", ".gemini/", ".claude/settings.json", ".claude/hooks/")
KB_TREES = (".claude/skills/", ".claude/commands/", ".claude/agents/", ".claude/references/")


class Plan(NamedTuple):
    """Asset and case names to smoke, notes for the owner, and whether the diff touches everything."""

    names: List[str]
    notes: List[str]
    everything: bool


def _word(text: str, word: str) -> Optional[int]:
    """Return the 1-based line of the first whole-word `word` in `text`, or None."""
    pattern = re.compile(rf"(?<![\w-]){re.escape(word)}(?![\w-])")
    for number, line in enumerate(text.splitlines(), 1):
        if pattern.search(line):
            return number
    return None


def _asset_files(repo: Path) -> Dict[str, Path]:
    claude = repo / ".claude"
    files = {p.parent.name: p for p in sorted((claude / "skills").glob("*/SKILL.md"))}
    for kind in ("commands", "agents"):
        files.update({p.stem: p for p in sorted((claude / kind).glob("*.md"))})
    return files


def _body(text: str) -> str:
    return kcases.split_frontmatter(text)[1]


def _preloaded_skills(text: str) -> List[str]:
    value = kcases.split_frontmatter(text)[0].get("skills", "")
    if value.startswith("["):
        return kcases.parse_list(value)
    return [line.strip()[1:].strip() for line in value.splitlines() if line.strip().startswith("-")]


def _referencing(
    repo: Path,
    changed: str,
    files: Dict[str, Path],
    texts: Dict[str, str],
    references: Dict[str, str],
) -> Tuple[List[str], List[str]]:
    """Return assets citing the `changed` reference directly or through references that cite it, with notes.

    `texts` maps asset names to their text and `references` maps repo-relative reference paths to
    theirs. A reference is cited by its stem; the traversal is keyed by path.
    """
    seen, queue = {changed}, [changed]
    names: List[str] = []
    notes: List[str] = []
    while queue:
        current = Path(queue.pop(0)).stem
        for rel, text in references.items():
            if rel not in seen and _word(text, current):
                seen.add(rel)
                queue.append(rel)
        for name, path in files.items():
            line = _word(texts[name], current)
            if line and name not in names:
                names.append(name)
                notes.append(f"AFFECTED {name} <- {changed} via {path.relative_to(repo)}:{line}")
    return names, notes


def affected(repo: Path, changes: List[Tuple[str, str]]) -> Plan:
    """Map `(status, path)` changes to the asset and case names whose evals they touch."""
    files = _asset_files(repo)
    texts = {name: path.read_text(encoding="utf-8") for name, path in files.items()}
    references = {
        str(p.relative_to(repo)): p.read_text(encoding="utf-8")
        for p in sorted((repo / ".claude" / "references").rglob("*.md"))
    }
    found = kcases.discover(repo / "evals" / "cases")
    commands = {n: p for n, p in files.items() if p.parent.name == "commands"}
    names: set = set()
    notes: List[str] = []
    everything = False

    def note(text: str) -> None:
        if text not in notes:
            notes.append(text)

    for status, path in changes:
        parts = path.split("/")
        got: set = set()
        deleted = False
        if path in EVERYTHING:
            everything = True
        elif path.startswith(UNMEASURED):
            note(f"UNMEASURED {path}")
        elif path.startswith("evals/kbeval/") and len(parts) == 3 and path.endswith(".py"):
            note(f"HARNESS {path}: covered by make test")
        elif parts[:2] == [".claude", "skills"] and len(parts) > 3:
            skill = parts[2]
            if status == "D" and parts[3:] == ["SKILL.md"]:
                deleted = True
                note(f"DELETED {skill}: add an AGENT_REMOVALS entry")
            elif skill in files:
                got.add(skill)
                got.update(n for n in files if skill in _preloaded_skills(texts[n]))
                got.update(n for n in commands if f"`{skill}`" in _body(texts[n]))
        elif parts[:2] == [".claude", "commands"] and len(parts) == 3 and path.endswith(".md"):
            command = Path(path).stem
            if status == "D":
                deleted = True
                note(f"DELETED {command}: add an AGENT_REMOVALS entry")
            elif command in files:
                got.add(command)
                got.update(
                    c.name for c in found
                    if c.suite == "workflows" and c.name.startswith("wf--") and c.body.startswith(f"/kb:{command} ")
                )  # fmt: skip
        elif parts[:2] == [".claude", "agents"] and len(parts) == 3 and path.endswith(".md"):
            agent = Path(path).stem
            if status == "D":
                deleted = True
                note(f"DELETED {agent}: add an AGENT_REMOVALS entry")
            elif agent in files:
                got.add(agent)
                got.update(n for n in commands if f"`{agent}`" in _body(texts[n]))
        elif parts[:2] == [".claude", "references"] and path.endswith(".md"):
            cited, reasons = _referencing(repo, path, files, texts, references)
            got.update(cited)
            for reason in reasons:
                note(reason)
        elif parts[:2] == ["evals", "cases"] and len(parts) > 4:
            got.update(c.name for c in found if c.name == parts[3] and c.suite == parts[2])
        elif parts[:2] == ["evals", "fixtures"] and len(parts) > 3:
            got.update(c.name for c in found if c.fixture and _word(c.fixture, parts[2]))
        if not got and not deleted and path.startswith(KB_TREES):
            note(f"UNMAPPED {path}: no eval case names it")
        names |= got
    return Plan(sorted(names), notes, everything)


def estimate(repo: Path, names: List[str]) -> Tuple[int, int, float, List[str]]:
    """Return (kb runs, vanilla runs, dollars, `UNPRICED` notes) for a one-run smoke of `names`."""
    found = [
        c for c in kcases.discover(repo / "evals" / "cases") if c.name in names or c.asset in names
    ]
    plain = [n for n in names if not n.startswith("wf--")]
    vanilla = sorted(
        c.name
        for c in found
        if c.suite == "effects" and any(c.name.startswith(f"{a}--") for a in plain)
    )[:VANILLA_SAMPLE]
    cost = sum(SMOKE_USD[c.suite] for c in found) + VANILLA_EFFECT_USD * len(vanilla)
    priced = {c.name for c in found} | {c.asset for c in found}
    unpriced = [f"UNPRICED {n}: no case" for n in names if n not in priced]
    return len(found), len(vanilla), cost, unpriced
