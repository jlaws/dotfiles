"""Mechanical conformance checks for the Claude knowledge base.

Everything here is deterministic: naming, frontmatter, link resolution, reachability. Judgment
calls (is this grounded? is it worth its tokens?) stay in SKILL.md, where a model can weigh them.

Usage:
    python3 .claude/skills/skill-audit/scripts/audit.py [REPO_ROOT] [--type TYPE ...]

Exits non-zero when any FAIL-severity check trips.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set

FAIL = "FAIL"
WARN = "WARN"

KEBAB = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
FRONTMATTER = re.compile(r"\A---\n(.*?)\n---\n", re.DOTALL)
FIELD = re.compile(r"^([A-Za-z][A-Za-z0-9_-]*):[ \t]*(.*)$")
TRIGGER = re.compile(r"\b(use when|use this when|use for|use to|when )", re.IGNORECASE)
REF_PATH = re.compile(r"references/[A-Za-z0-9._/-]+\.md")
REF_DIR = re.compile(r"\.claude/references/([a-z0-9-]+)/")
MD_ANCHOR = re.compile(r"\]\(([^)\s]+#[^)\s]+)\)")
BARE_BULLET = re.compile(r"^\s*[-*]\s+(.*)$")
SKILL_INVOCATION = re.compile(
    r"(?:invoke|load|follow|use)(?: the)? ([a-z0-9-]+) skill|skill `([a-z0-9-]+)`",
    re.IGNORECASE,
)
SUBAGENT = re.compile(r"subagent_type[\"']?\s*[:=]\s*[\"']([a-z0-9-]+)")
ARG_TOKEN = re.compile(r"\$ARGUMENTS|\$\d")

VALID_TOOLS = {
    "Read",
    "Grep",
    "Glob",
    "Bash",
    "Write",
    "Edit",
    "NotebookEdit",
    "WebFetch",
    "WebSearch",
    "Task",
}

# Codex reads shared skill metadata from a budgeted listing; only that tree is constrained.
SHARED_DESCRIPTION_BUDGET = 64
# Claude truncates a listed description well above this; flag only genuine runaways.
DESCRIPTION_LIMIT = 1024


@dataclass
class Finding:
    """One conformance violation."""

    severity: str
    check: str
    path: Path
    message: str


class Audit:
    """Collects findings across the knowledge base."""

    def __init__(self, repo: Path) -> None:
        """Bind the audit to a repository root."""
        self.repo = repo
        self.claude = repo / ".claude"
        self.shared = repo / ".agents"
        self.findings: List[Finding] = []
        self.clean: Dict[str, List[str]] = {}
        self.reference_stems: Dict[str, Path] = {
            path.stem: path for path in (self.claude / "references").rglob("*.md")
        }

    # -- helpers ---------------------------------------------------------

    def add(self, severity: str, check: str, path: Path, message: str) -> None:
        """Record a finding."""
        self.findings.append(Finding(severity, check, self.rel(path), message))

    def rel(self, path: Path) -> Path:
        """Return the path relative to the repo root when possible."""
        try:
            return path.relative_to(self.repo)
        except ValueError:
            return path

    @staticmethod
    def frontmatter(text: str) -> Optional[Dict[str, str]]:
        """Parse a flat YAML frontmatter block into a dict, or return None if absent."""
        match = FRONTMATTER.match(text)
        if match is None:
            return None
        fields: Dict[str, str] = {}
        for line in match.group(1).splitlines():
            field = FIELD.match(line)
            if field:
                fields[field.group(1)] = field.group(2).strip().strip("\"'")
        return fields

    @staticmethod
    def body(text: str) -> str:
        """Return the document body with any frontmatter removed."""
        match = FRONTMATTER.match(text)
        return text[match.end() :] if match else text

    def note_clean(self, kind: str, name: str) -> None:
        """Record an asset with no findings."""
        self.clean.setdefault(kind, []).append(name)

    def had_findings(self, path: Path) -> bool:
        """Report whether this path already produced a finding."""
        rel = self.rel(path)
        return any(f.path == rel for f in self.findings)

    # -- skills ----------------------------------------------------------

    def check_skills(self) -> None:
        """Validate skill folders, frontmatter, and supporting-file usage."""
        root = self.claude / "skills"
        for folder in sorted(p for p in root.iterdir() if p.is_dir()):
            skill = folder / "SKILL.md"
            if not skill.is_file():
                lower = [p.name for p in folder.iterdir() if p.name.lower() == "skill.md"]
                detail = f"found {lower[0]}" if lower else "no SKILL.md"
                self.add(FAIL, "SK-S1", folder, f"missing SKILL.md ({detail})")
                continue
            if (folder / "README.md").is_file():
                self.add(FAIL, "SK-S2", folder / "README.md", "use SKILL.md, not README.md")
            if not KEBAB.match(folder.name):
                self.add(FAIL, "SK-S3", folder, f"folder name is not kebab-case: {folder.name}")

            text = skill.read_text()
            fields = self.frontmatter(text)
            if fields is None:
                self.add(FAIL, "SK-F1", skill, "no YAML frontmatter")
                continue
            self._check_skill_frontmatter(skill, folder, fields)
            self._check_supporting_files(skill, folder, text)

            if not self.had_findings(skill) and not self.had_findings(folder):
                self.note_clean("Skills", folder.name)

    def _check_skill_frontmatter(self, skill: Path, folder: Path, fields: Dict[str, str]) -> None:
        name = fields.get("name")
        if name is None:
            self.add(FAIL, "SK-F2", skill, "frontmatter has no `name`")
        else:
            if not KEBAB.match(name):
                self.add(FAIL, "SK-F3", skill, f"`name` is not kebab-case: {name}")
            if name != folder.name:
                self.add(FAIL, "SK-F4", skill, f"`name` {name!r} != folder {folder.name!r}")
            if re.search(r"claude|anthropic", name, re.IGNORECASE):
                self.add(FAIL, "SK-F8", skill, f"`name` uses a reserved term: {name}")

        description = fields.get("description")
        if description is None:
            self.add(FAIL, "SK-F5", skill, "frontmatter has no `description`")
        elif not TRIGGER.search(description):
            self.add(WARN, "SK-F9", skill, "`description` states no triggering condition")

        for key, value in fields.items():
            if "<" in value or ">" in value:
                self.add(FAIL, "SK-F7", skill, f"`{key}` contains XML angle brackets")

        tools = fields.get("allowed-tools")
        if tools:
            unknown = sorted({t.strip() for t in tools.split(",")} - VALID_TOOLS)
            if unknown:
                self.add(WARN, "SK-F10", skill, f"unknown allowed-tools: {', '.join(unknown)}")

    def _check_supporting_files(self, skill: Path, folder: Path, text: str) -> None:
        """Flag bundled files unreachable from SKILL.md, following references transitively.

        A supporting file named only by another supporting file is still reachable, so long as the
        chain starts at SKILL.md.
        """
        extras = [p for p in sorted(folder.rglob("*")) if p.is_file() and p != skill]
        reachable = {skill}
        frontier = [text]
        while frontier:
            body = frontier.pop()
            for extra in extras:
                if extra in reachable:
                    continue
                if extra.name in body or str(extra.relative_to(folder)) in body:
                    reachable.add(extra)
                    try:
                        frontier.append(extra.read_text())
                    except UnicodeDecodeError:
                        pass
        for extra in extras:
            if extra not in reachable:
                self.add(
                    WARN,
                    "SK-C3",
                    extra,
                    "supporting file is unreachable from SKILL.md",
                )

    def check_shared_skill_budget(self) -> None:
        """Enforce the Codex metadata budget on the shared tree only."""
        root = self.shared / "skills"
        if not root.is_dir():
            return
        for skill in sorted(root.glob("*/SKILL.md")):
            fields = self.frontmatter(skill.read_text()) or {}
            description = fields.get("description", "")
            if len(description) > SHARED_DESCRIPTION_BUDGET:
                self.add(
                    FAIL,
                    "SK-F6",
                    skill,
                    f"shared description is {len(description)} chars "
                    f"(budget {SHARED_DESCRIPTION_BUDGET})",
                )

    # -- agents ----------------------------------------------------------

    def check_agents(self) -> None:
        """Validate agent files, tool declarations, and outbound references."""
        root = self.claude / "agents"
        skills = {p.parent.name for p in (self.claude / "skills").glob("*/SKILL.md")}
        for sub in sorted(p for p in root.iterdir() if p.is_dir()):
            self.add(WARN, "AG-S2", sub, "agents should be flat files, not subdirectories")
        for agent in sorted(root.glob("*.md")):
            if not KEBAB.match(agent.stem):
                self.add(FAIL, "AG-S1", agent, f"filename is not kebab-case: {agent.name}")
            text = agent.read_text()
            fields = self.frontmatter(text)
            if fields is None:
                self.add(FAIL, "AG-F1", agent, "no YAML frontmatter")
                continue
            self._check_named_asset(agent, fields, "AG-F2", "AG-F3", "AG-F4", "AG-F5")

            if "tools" not in fields:
                self.add(WARN, "AG-F6", agent, "declares no `tools`")
            else:
                unknown = sorted({t.strip() for t in fields["tools"].split(",")} - VALID_TOOLS)
                if unknown:
                    self.add(WARN, "AG-F7", agent, f"unknown tools: {', '.join(unknown)}")

            if len(self.body(text).split()) < 20:
                self.add(WARN, "AG-C1", agent, "body gives almost no role instruction")

            self._check_declared_skills(agent, text, skills, "AG-C2")
            self._check_reference_mentions(agent, text)
            self._check_agent_index(agent, text)

            if not self.had_findings(agent):
                self.note_clean("Agents", agent.stem)

    def _check_agent_index(self, agent: Path, text: str) -> None:
        """Every bare stem in a reference bullet index must resolve to a real file."""
        for lineno, line in enumerate(text.splitlines(), 1):
            bullet = BARE_BULLET.match(line)
            if not bullet:
                continue
            body = re.sub(r"\s*\(.*?\)\s*$", "", bullet.group(1)).strip().rstrip(".")
            parts = [p.strip() for p in body.split(",")]
            if len(parts) < 2 or not all(KEBAB.match(p) for p in parts):
                continue
            for part in parts:
                if part not in self.reference_stems:
                    self.add(
                        FAIL,
                        "AG-C4",
                        agent,
                        f"line {lineno}: indexed reference {part!r} does not exist",
                    )

    # -- commands --------------------------------------------------------

    def check_commands(self) -> None:
        """Validate command files and the skills and agents they route to."""
        root = self.claude / "commands"
        skills = {p.parent.name for p in (self.claude / "skills").glob("*/SKILL.md")}
        agents = {p.stem for p in (self.claude / "agents").glob("*.md")}
        for sub in sorted(p for p in root.iterdir() if p.is_dir()):
            self.add(WARN, "CM-S2", sub, "commands should be flat files, not subdirectories")
        for command in sorted(root.glob("*.md")):
            if not KEBAB.match(command.stem):
                self.add(FAIL, "CM-S1", command, f"filename is not kebab-case: {command.name}")
            if not command.stem.startswith("j-"):
                self.add(WARN, "CM-S3", command, "custom commands use the `j-` prefix")
            text = command.read_text()
            fields = self.frontmatter(text)
            if fields is None:
                self.add(FAIL, "CM-F1", command, "no YAML frontmatter")
                continue
            self._check_named_asset(command, fields, "CM-F2", "CM-F3", "CM-F4", "CM-F5")

            description = fields.get("description", "")
            if description and not TRIGGER.search(description):
                self.add(WARN, "CM-F6", command, "`description` states no triggering condition")

            body = self.body(text)
            if len(body.split()) < 10:
                self.add(WARN, "CM-C1", command, "body gives almost no instruction")
            if ARG_TOKEN.search(body) and "argument-hint" not in fields:
                self.add(WARN, "CM-C2", command, "uses arguments but declares no `argument-hint`")

            self._check_declared_skills(command, text, skills, "CM-C3")
            for match in SUBAGENT.finditer(body):
                if match.group(1) not in agents:
                    self.add(
                        WARN,
                        "CM-C4",
                        command,
                        f"routes to unknown agent {match.group(1)!r}",
                    )
            self._check_reference_mentions(command, text)

            if not self.had_findings(command):
                self.note_clean("Commands", command.stem)

    # -- references ------------------------------------------------------

    def check_references(self) -> None:
        """Validate reference naming, structure, link targets, and reachability."""
        root = self.claude / "references"
        indexed = self._indexed_stems()
        for path in sorted(root.rglob("*.md")):
            if not KEBAB.match(path.stem):
                self.add(FAIL, "RF-S1", path, f"filename is not kebab-case: {path.name}")
            if path.parent == root:
                self.add(WARN, "RF-S2", path, "reference is not inside a category directory")
            text = path.read_text()
            if not any(line.startswith("#") for line in text.splitlines()):
                self.add(WARN, "RF-C1", path, "no markdown heading")
            if path.stem not in indexed:
                self.add(
                    WARN,
                    "RF-C3",
                    path,
                    "not indexed by any agent, command, or skill",
                )
            self._check_relative_reference_paths(path, text)

            if not self.had_findings(path):
                self.note_clean("References", str(path.relative_to(root)))

    def _indexed_stems(self) -> Set[str]:
        """Collect every reference stem reachable from an agent, command, skill, or CLAUDE.md.

        Reachability is transitive: agent indexes name top-level references, and those link on to
        nested children. A child cited only by a reachable parent is reachable.
        """
        seen: Set[str] = set()
        roots = [
            self.claude / "agents",
            self.claude / "commands",
            self.claude / "skills",
            self.claude / "CLAUDE.md",
        ]
        for base in roots:
            paths = [base] if base.is_file() else sorted(base.rglob("*.md"))
            for path in paths:
                text = path.read_text()
                for stem in self.reference_stems:
                    if re.search(rf"\b{re.escape(stem)}\b", text):
                        seen.add(stem)
        frontier = list(seen)
        while frontier:
            text = self.reference_stems[frontier.pop()].read_text()
            for stem in self.reference_stems:
                if stem in seen:
                    continue
                if re.search(rf"\b{re.escape(stem)}\b", text):
                    seen.add(stem)
                    frontier.append(stem)
        return seen

    def _check_relative_reference_paths(self, path: Path, text: str) -> None:
        root = self.claude / "references"
        for lineno, line in enumerate(text.splitlines(), 1):
            for target in set(REF_PATH.findall(line)):
                rel = target[len("references/") :]
                if (root / rel).exists() or (path.parent / rel).exists():
                    continue
                actual = self.reference_stems.get(Path(rel).stem)
                hint = f"; actual location references/{actual.relative_to(root)}" if actual else ""
                self.add(FAIL, "XR-7", path, f"line {lineno}: {target} does not resolve{hint}")

    # -- shared frontmatter and link helpers -----------------------------

    def _check_named_asset(
        self,
        path: Path,
        fields: Dict[str, str],
        name_check: str,
        match_check: str,
        desc_check: str,
        limit_check: str,
    ) -> None:
        name = fields.get("name")
        if name is None:
            self.add(FAIL, name_check, path, "frontmatter has no `name`")
        elif name != path.stem:
            self.add(FAIL, match_check, path, f"`name` {name!r} != filename {path.stem!r}")
        description = fields.get("description")
        if description is None:
            self.add(FAIL, desc_check, path, "frontmatter has no `description`")
        elif len(description) > DESCRIPTION_LIMIT:
            self.add(
                WARN,
                limit_check,
                path,
                f"`description` is {len(description)} chars (limit {DESCRIPTION_LIMIT})",
            )

    def _check_declared_skills(
        self, path: Path, text: str, skills: Set[str], check: str
    ) -> None:
        """Resolve both `skills:` frontmatter entries and prose skill invocations."""
        fields = self.frontmatter(text) or {}
        declared: List[str] = []
        inline = fields.get("skills")
        if inline:
            declared.extend(s.strip() for s in inline.strip("[]").split(","))
        match = FRONTMATTER.match(text)
        if match:
            block = match.group(1)
            capture = False
            for line in block.splitlines():
                if line.startswith("skills:"):
                    capture = True
                    continue
                if capture:
                    item = re.match(r"^\s+-\s+(\S+)", line)
                    if item:
                        declared.append(item.group(1))
                        continue
                    capture = False
        for name in declared:
            if name and name not in skills:
                self.add(FAIL, check, path, f"declares unknown skill {name!r}")
        for prose in SKILL_INVOCATION.finditer(self.body(text)):
            name = prose.group(1) or prose.group(2)
            if name in skills or name is None:
                continue
            if any(name == d for d in declared):
                continue
            if (self.claude / "skills" / name).exists():
                continue
            # Only flag names that look like a real skill reference, not ordinary prose.
            if "-" in name:
                self.add(WARN, check, path, f"names a skill that does not exist: {name!r}")

    def _check_reference_mentions(self, path: Path, text: str) -> None:
        for lineno, line in enumerate(text.splitlines(), 1):
            for target in set(REF_PATH.findall(line)):
                if (self.claude / target).exists():
                    continue
                # Skill-local references/ is a separate namespace from .claude/references/.
                if (path.parent / target).exists():
                    continue
                self.add(WARN, "XR-2", path, f"line {lineno}: {target} does not resolve")
            for category in set(REF_DIR.findall(line)):
                if not (self.claude / "references" / category).is_dir():
                    self.add(
                        WARN,
                        "XR-2",
                        path,
                        f"line {lineno}: reference category {category!r} does not exist",
                    )

    def check_anchors(self) -> None:
        """Every in-repo markdown anchor must resolve against a real heading."""
        headings: Dict[Path, Set[str]] = {}
        for path in self.claude.rglob("*.md"):
            headings[path.resolve()] = {
                _slug(line.lstrip("#").strip())
                for line in path.read_text().splitlines()
                if line.startswith("#")
            }
        for path in sorted(self.claude.rglob("*.md")):
            own = headings[path.resolve()]
            for lineno, line in enumerate(path.read_text().splitlines(), 1):
                for target in MD_ANCHOR.findall(line):
                    filepart, _, anchor = target.partition("#")
                    if not anchor:
                        continue
                    if filepart:
                        resolved = (path.parent / filepart).resolve()
                        if resolved not in headings:
                            continue
                        pool = headings[resolved]
                    else:
                        pool = own
                    if anchor not in pool:
                        self.add(FAIL, "XR-8", path, f"line {lineno}: anchor {target} not found")

    # -- config ----------------------------------------------------------

    def check_config(self) -> None:
        """Validate CLAUDE.md presence and settings-file syntax."""
        claude_md = self.claude / "CLAUDE.md"
        if not claude_md.is_file():
            self.add(FAIL, "CF-C1", claude_md, "missing")
        elif not any(line.startswith("#") for line in claude_md.read_text().splitlines()):
            self.add(WARN, "CF-C2", claude_md, "no headings")

        settings = self.claude / "settings.json"
        if not settings.is_file():
            self.add(WARN, "CF-S1", settings, "missing")
        for name, severity in (("settings.json", FAIL), ("settings.local.json", FAIL)):
            path = self.claude / name
            if not path.is_file():
                continue
            try:
                json.loads(path.read_text())
            except json.JSONDecodeError as exc:
                self.add(severity, "CF-S2", path, f"invalid JSON: {exc}")

    # -- reporting -------------------------------------------------------

    def report(self) -> int:
        """Print grouped findings and return a process exit code."""
        fails = [f for f in self.findings if f.severity == FAIL]
        warns = [f for f in self.findings if f.severity == WARN]
        print("Knowledge Base Audit (mechanical)")
        print("=" * 34)
        for kind in ("Skills", "Agents", "Commands", "References"):
            names = self.clean.get(kind, [])
            print(f"{kind + ':':12}{len(names)} clean")
        print(f"{'Findings:':12}{len(fails)} fail  |  {len(warns)} warn")

        for label, group in ((FAIL, fails), (WARN, warns)):
            if not group:
                continue
            print(f"\n{label}")
            print("-" * len(label))
            for finding in sorted(group, key=lambda f: (str(f.path), f.check)):
                print(f"[{finding.check}] {finding.path}: {finding.message}")
        return 1 if fails else 0


def _slug(text: str) -> str:
    """Convert a heading to a GitHub-style anchor slug."""
    text = re.sub(r"[`*_]", "", text.strip().lower())
    text = re.sub(r"[^a-z0-9\s-]", "", text)
    return re.sub(r"\s+", "-", text).strip("-")


ALL_TYPES = ("skills", "agents", "commands", "references", "config")


def run(repo: Path, types: Sequence[str]) -> int:
    """Run the selected check groups and print the report."""
    audit = Audit(repo)
    if "skills" in types:
        audit.check_skills()
        audit.check_shared_skill_budget()
    if "agents" in types:
        audit.check_agents()
    if "commands" in types:
        audit.check_commands()
    if "references" in types:
        audit.check_references()
    if "config" in types:
        audit.check_config()
    audit.check_anchors()
    return audit.report()


def main(argv: Optional[Iterable[str]] = None) -> int:
    """Parse arguments and run the audit."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repo", nargs="?", default=".", help="repository root")
    parser.add_argument(
        "--type",
        action="append",
        choices=ALL_TYPES,
        dest="types",
        help="limit to one asset type (repeatable)",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)
    return run(Path(args.repo).resolve(), args.types or ALL_TYPES)


if __name__ == "__main__":
    sys.exit(main())
