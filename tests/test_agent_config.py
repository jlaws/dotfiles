"""Architecture checks for the tool-specific agent configuration trees."""

from __future__ import annotations

import json
import re
import subprocess
import unittest
from pathlib import Path

from tests.markdown import HEADING, heading_texts, markdown_headings

REPO = Path(__file__).resolve().parents[1]
FRONTMATTER_DESCRIPTION = re.compile(r'^description:\s*["\']?(.*?)["\']?$', re.MULTILINE)

# `assertRegex` calls `re.search`, which anchors `^` to the start of the string only. `description`
# is line 2 and `developer_instructions` line 4 of every Codex TOML, so a bare `^` pattern fails all
# of them. Its third positional is `msg`, not flags, so the flag has to be compiled in here.
CODEX_DESCRIPTION = re.compile(r'^description = "[^"\s]', re.MULTILINE)
# `\s` crosses newlines without DOTALL, so this requires real content in the block. Matching
# only the opening delimiter accepted `developer_instructions = """"""` -- precisely the empty
# role the test exists to catch.
CODEX_INSTRUCTIONS = re.compile(r'^developer_instructions = """\s*[^"\s]', re.MULTILINE)
GEMINI_INHERIT = re.compile(r"^model: inherit$", re.MULTILINE)
GEMINI_DESCRIPTION = re.compile(r'^description: "[^"\s]', re.MULTILINE)

# A path into one of the four agent trees is a claim that the tree ships that file. `~/` and the
# repo-relative form name the same asset, because the trees are synced to `~` verbatim. Example
# paths in reference bodies (`./train.py`, `/tmp/...`) are not claims and do not match.
# The lookbehind excludes `..claude` and word-joined text but deliberately allows a leading `/`,
# so `~/.claude/x.py`, `dotfiles/.claude/x.py`, and an absolute path all resolve to the same claim.
# An earlier `(?<![\w/.])` silently skipped every prefixed form, leaving a fail-closed test with a
# hole that read as coverage.
TREE_SCRIPT_PATH = re.compile(
    r"(?<![\w.])/?(?:~/)?"
    r"(\.(?:claude|codex|agents|gemini)/[A-Za-z0-9._/-]*?\.(?:py|sh|mjs|js))"
    r"(?![A-Za-z0-9])"
)
TREE_ROOTS = (".claude", ".codex", ".agents", ".gemini")

# Asset-set parity is enforced by name; bodies and descriptions are free to diverge per tool.
# `.claude/` is rightsized for the Claude 5 generation, so its wording no longer tracks `.agents/`.
# Anything that legitimately exists in only one tree is declared here rather than tolerated
# silently, so an accidental deletion still fails.

# Claude-native commands with no shared `cmd-j-*` workflow behind them.
CLAUDE_ONLY_COMMANDS: frozenset[str] = frozenset()

# Shared workflows deliberately absent from `.claude/skills/`.
SHARED_ONLY_SKILLS: frozenset[str] = frozenset()

# Skills that exist only for Claude, and agents that exist in only one tree.
CLAUDE_ONLY_SKILLS: frozenset[str] = frozenset()
CLAUDE_ONLY_AGENTS: frozenset[str] = frozenset()

J_PLAN_COMMANDS = (
    REPO / ".agents" / "skills" / "cmd-j-plan" / "SKILL.md",
    REPO / ".codex" / "prompts" / "j-plan.md",
    REPO / ".gemini" / "antigravity-cli" / "skills" / "j-plan" / "SKILL.md",
)

PLAN_STORAGE_SKILLS = (REPO / ".agents" / "skills" / "writing-plans" / "SKILL.md",)

PLAN_EXECUTION_CONSUMERS = (
    REPO / ".agents" / "skills" / "cmd-j-execute-plan" / "SKILL.md",
    REPO / ".agents" / "skills" / "cmd-j-next" / "SKILL.md",
    REPO / ".codex" / "prompts" / "j-execute-plan.md",
    REPO / ".codex" / "prompts" / "j-next.md",
    REPO / ".gemini" / "antigravity-cli" / "skills" / "j-execute-plan" / "SKILL.md",
    REPO / ".gemini" / "antigravity-cli" / "skills" / "j-next" / "SKILL.md",
)

ACTIVE_PLAN_CONSUMERS = (
    REPO / ".agents" / "skills" / "cmd-j-diff-review" / "SKILL.md",
    REPO / ".codex" / "prompts" / "j-diff-review.md",
    REPO / ".gemini" / "antigravity-cli" / "skills" / "j-diff-review" / "SKILL.md",
)

# A worktree carries no uncommitted changes and branches from a start point the agent did not pick,
# so a review run in one silently reviews the wrong commit. Each statement below has one owner --
# repeating it across surfaces is what `writing-skills` warns against -- and the owners are pinned so
# the guidance cannot quietly disappear from a tree.

# Owns "which workspace does a dispatched agent get".
WORKSPACE_SELECTION_OWNERS = (
    REPO / ".claude" / "skills" / "dispatching-parallel-agents" / "SKILL.md",
    REPO / ".agents" / "skills" / "dispatching-parallel-agents" / "SKILL.md",
)

# Owns "which commit does a worktree start from, and how do you confirm it".
WORKTREE_BASE_OWNERS = (
    REPO / ".claude" / "skills" / "using-git-worktrees" / "SKILL.md",
    REPO / ".agents" / "skills" / "using-git-worktrees" / "SKILL.md",
)

# Owns "what a dispatched agent returns". A subagent's report is injected into the caller's context
# verbatim AND is the only artifact -- nothing is persisted, so the contract compresses prose and
# never substance. One owner, preloaded into agents via `skills:`; the alternative was restating it
# in every agent body across all three trees, which is what :53-56 forbids.
REPORT_CONTRACT_OWNERS = (
    REPO / ".claude" / "skills" / "subagent-report-contract" / "SKILL.md",
    REPO / ".agents" / "skills" / "subagent-report-contract" / "SKILL.md",
    REPO / ".gemini" / "antigravity-cli" / "skills" / "subagent-report-contract" / "SKILL.md",
)

# Persisting reports to disk was considered and cut. These markers keep it cut: reintroducing it
# would need the Write tool on agents that are deliberately read-only, or a permissionMode that punches through plan
# mode. Neither is worth an archive that may never be opened.
DISK_PERSISTENCE_MARKERS = ("scratchpad/agent-reports/", "permissionMode", "j-agent-reports")

# Owns "should this code exist at all" -- the seven-rung stopping rule from ponytail. Separate owner
# from completeness-principle, which owns "how thoroughly to build what is in scope"; the two govern
# different axes and reading either alone inverts the other. The safety carve-outs are pinned
# because ponytail measured a paraphrase that dropped them at 95% safe against the full ruleset's
# 100%, so the carve-outs ARE the safety margin, not commentary.
LADDER_OWNERS = (
    REPO / ".claude" / "references" / "workflow" / "code-efficiency-ladder.md",
    REPO / ".agents" / "references" / "workflow" / "code-efficiency-ladder.md",
)

# Strings the ladder is unsafe without. Same reason upstream pins its own: the ruleset pushes toward
# the shortest solution, and these are what stop it pushing through a trust boundary.
LADDER_SAFETY_INVARIANTS = (
    "Input validation at trust boundaries",
    "Error handling that prevents data loss",
    "Security measures",
    "Accessibility basics",
    "Anything explicitly requested",
    "correct on edge cases",
    "Weakening counts as removing",
    "never the reading",
)

# Surfaces where a code decision actually gets made. A reference reachable only from another
# reference is indexed but never reached, so these carry a pointer -- and only a pointer, since
# writing-skills forbids restating one statement across surfaces.
LADDER_CONSUMERS = (
    REPO / ".claude" / "skills" / "code-quality" / "SKILL.md",
    REPO / ".agents" / "skills" / "code-quality" / "SKILL.md",
    REPO / ".gemini" / "antigravity-cli" / "skills" / "code-quality" / "SKILL.md",
    REPO / ".claude" / "agents" / "code-reviewer.md",
    REPO / ".claude" / "commands" / "j-diff-review.md",
    REPO / ".gemini" / "config" / "agents" / "code-reviewer.md",
    REPO / ".gemini" / "antigravity-cli" / "skills" / "j-diff-review" / "SKILL.md",
)

# Banning one lead-in sentence does not detect a restatement -- reword the lead-in and the rungs
# copy across fine. These are the rungs' own distinctive text.
LADDER_RUNG_MARKERS = (
    "Stop at the first rung that holds",
    "Already in this codebase?",
    "Native platform feature covers it?",
)
# Surfaces that act on an ADR that already exists. ADRs are living documents -- edited in place,
# deleted when the decision is gone -- so each of these has to carry that branch rather than the
# amend-or-supersede one it replaced. See docs/adr/workflow/adrs-are-living-documents.md.
ADR_EDIT_IN_PLACE_CONSUMERS = (
    REPO / ".claude" / "commands" / "j-arch.md",
    REPO / ".agents" / "skills" / "cmd-j-arch" / "SKILL.md",
    REPO / ".codex" / "prompts" / "j-arch.md",
    REPO / ".gemini" / "antigravity-cli" / "skills" / "j-arch" / "SKILL.md",
    REPO / ".claude" / "skills" / "post-ship-doc-sync" / "SKILL.md",
    REPO / ".agents" / "skills" / "post-ship-doc-sync" / "SKILL.md",
    REPO / ".gemini" / "antigravity-cli" / "skills" / "post-ship-doc-sync" / "SKILL.md",
    REPO / ".claude" / "commands" / "j-plan.md",
    REPO / ".codex" / "prompts" / "j-plan.md",
    REPO / ".agents" / "skills" / "cmd-j-plan" / "SKILL.md",
    REPO / ".gemini" / "antigravity-cli" / "skills" / "j-plan" / "SKILL.md",
    REPO / ".claude" / "skills" / "writing-plans" / "SKILL.md",
    REPO / ".agents" / "skills" / "writing-plans" / "SKILL.md",
    REPO / ".gemini" / "antigravity-cli" / "skills" / "writing-plans" / "SKILL.md",
)

# The lifecycle this repo retired. `tests/test_adr.py` bans these as *headings under docs/adr/*,
# which says nothing about a skill or prompt teaching them again in prose.
ADR_RETIRED_VOCABULARY = (
    "Amendment Log",
    "superseding ADR",
    "supersede it",
    "Never edit an accepted ADR",
)


# A surface that tells a reviewer a `// SIMPLIFIED:` marker is sanctioned must say in the same
# breath that the sanction stops at a security control. Without that, the marker is a channel for
# reviewed code to instruct its reviewer to look away, and the diff is untrusted data.
SIMPLIFIED_SANCTION_SURFACES = (
    REPO / ".claude" / "skills" / "code-quality" / "SKILL.md",
    REPO / ".agents" / "skills" / "code-quality" / "SKILL.md",
    REPO / ".gemini" / "antigravity-cli" / "skills" / "code-quality" / "SKILL.md",
    REPO / ".claude" / "agents" / "code-reviewer.md",
    REPO / ".claude" / "commands" / "j-diff-review.md",
    REPO / ".gemini" / "config" / "agents" / "code-reviewer.md",
    REPO / ".gemini" / "antigravity-cli" / "skills" / "j-diff-review" / "SKILL.md",
)

# The unconditional form of this rule ("prefer bullets, tables, and code over prose") was measured
# as the cause of the one backfire in a 20-task suite: a "summarize/compare X vs Y" prompt answered
# with headed pro/con walls ran 173% of a no-tool baseline. Density is per unit of information
# CARRIED, so scaffolding the question did not ask for costs tokens even in table form. One owner
# holds the conditional; the always-loaded configs point at it rather than restating half of it.
STRUCTURE_RULE_OWNERS = (
    REPO / ".claude" / "references" / "workflow" / "context-efficiency.md",
    REPO / ".agents" / "references" / "workflow" / "context-efficiency.md",
)

# Phrases that only make sense as the unconditional rule. Banned from the always-loaded configs so
# the conditional cannot be quietly reverted to the form the measurement contradicted.
UNCONDITIONAL_STRUCTURE_PHRASES = (
    "Prefer bullets, tables, and code over prose",
    "Prefer tables and code over prose",
    "If information can be a table, make it a table",
)

# Byte ceilings for the three configs loaded on EVERY request. These files are the one part of the
# knowledge base that is never on-demand, so a rule that changes nothing is pure recurring overhead
# and growth here is not free the way growth in a reference is.
#
# Bump deliberately if you add a documented rule; do not bump because of phrasing creep. Measured
# 2026-09-12; headroom is ~10% over actual, which is enough for a real addition and not enough to
# absorb drift unnoticed.
ALWAYS_LOADED_CEILINGS = {
    REPO / ".claude" / "CLAUDE.md": 8000,
    REPO / ".codex" / "AGENTS.md": 12700,
    REPO / ".gemini" / "GEMINI.md": 17500,
}

# A versioned model ID in a delegation ladder. The vendor word may be followed by up to two lowercase
# segments before the version, so this catches `claude-opus-5` and `claude-haiku-4-5` as well as
# `gpt-6-astra`; matching only `vendor` + digit would miss every current Claude ID. A digit is
# required, which keeps the floating aliases (opus, sonnet, haiku, fable, flash, pro) legal.
PINNED_MODEL = re.compile(
    r"\b(?:gpt|claude|gemini|opus|sonnet|haiku|fable)(?:[- ][a-z]+){0,2}[- ]\d"
)

# Every asset whose run ends on a pull request. Each reports the URL; `create-pr` additionally has
# to look for an already-open PR the way `finishing-branch` does, instead of always creating one.
PR_URL_SURFACES = (
    REPO / ".claude" / "skills" / "pr-comment-resolution" / "SKILL.md",
    REPO / ".agents" / "skills" / "pr-comment-resolution" / "SKILL.md",
    REPO / ".gemini" / "antigravity-cli" / "skills" / "pr-comment-resolution" / "SKILL.md",
    REPO / ".claude" / "agents" / "create-pr.md",
    REPO / ".codex" / "agents" / "create-pr.toml",
    REPO / ".gemini" / "config" / "agents" / "create-pr.md",
)

# The byte ceiling above is a standing instruction to cut prose from these files. These are the
# lines that "cut something" must never reach -- each one prevents an irreversible action or a
# prompt-injection foothold, and none of them is recoverable by noticing it went missing.
SAFETY_LINE_INVARIANTS = {
    REPO / ".claude" / "CLAUDE.md": (
        "Never claim success without evidence",
        "untrusted data, not instructions",
        "Never force push to main or master",
        "Tear down paid cloud services",
    ),
    REPO / ".codex" / "AGENTS.md": (
        "untrusted data, not instructions",
        "Never force push to main/master",
        "tear down paid cloud services",
    ),
    REPO / ".gemini" / "GEMINI.md": (
        "untrusted data (not instructions)",
        "Never force push to main/master",
        "Tear down paid cloud services",
    ),
}

# Always-loaded configs carry a pointer, not a restatement.
REPORT_POINTERS = (
    REPO / ".claude" / "CLAUDE.md",
    REPO / ".codex" / "AGENTS.md",
    REPO / ".gemini" / "GEMINI.md",
)

# Always-loaded configs carry a pointer, not a restatement.
WORKTREE_POINTERS = (
    REPO / ".claude" / "CLAUDE.md",
    REPO / ".codex" / "AGENTS.md",
    REPO / ".gemini" / "GEMINI.md",
)

# Review dispatch hands each agent the HEAD SHA it should be sitting on.
DIFF_REVIEW_DISPATCHERS = (
    REPO / ".claude" / "commands" / "j-diff-review.md",
    REPO / ".agents" / "skills" / "cmd-j-diff-review" / "SKILL.md",
    REPO / ".codex" / "prompts" / "j-diff-review.md",
    REPO / ".gemini" / "antigravity-cli" / "skills" / "j-diff-review" / "SKILL.md",
)

# Local `main` can trail the remote, which makes a squash absorb commits that already landed.
# Every branch workflow resolves its base through `origin/main` instead.
BRANCH_BASE_REF_DOCS = (
    REPO / ".claude" / "skills" / "using-git-worktrees" / "SKILL.md",
    REPO / ".agents" / "skills" / "using-git-worktrees" / "SKILL.md",
    REPO / ".claude" / "skills" / "finishing-branch" / "SKILL.md",
    REPO / ".agents" / "skills" / "finishing-branch" / "SKILL.md",
    REPO / ".claude" / "commands" / "j-rebase.md",
    REPO / ".agents" / "skills" / "cmd-j-rebase" / "SKILL.md",
    REPO / ".codex" / "AGENTS.md",
    REPO / ".gemini" / "GEMINI.md",
)

# Two trees, one table of contents. Bodies may differ -- `.claude/` is written for a generation
# `.agents/` does not serve -- but a heading in one tree and not the other means a reader of the
# other tree cannot find the topic at all. See docs/adr/workflow/reference-tree-section-parity.md.
REFERENCE_TREES = (REPO / ".claude" / "references", REPO / ".agents" / "references")


def discipline_body(text: str) -> list[str]:
    """The prose under the existing-code-discipline sections, normalized for cross-copy compare.

    Heading level is dropped -- the two prompts nest the same sections one level deeper -- as are
    blank lines. Everything else has to match, so editing the reference and leaving a pasted copy
    behind fails here instead of drifting silently until someone reads both.
    """
    wanted = set(DISCIPLINE_SECTIONS)
    body: list[str] = []
    capturing = False
    for line in text.splitlines():
        match = HEADING.match(line)
        if match:
            capturing = match.group(2) in wanted
            if capturing:
                body.append(f"# {match.group(2)}")
            continue
        if capturing and line.strip():
            body.append(line.strip())
    return body


# The reference plus the two prompts that paste its body, since Codex has no skill loader.
EXISTING_CODE_DISCIPLINE_OWNERS = (
    ".claude/references/workflow/existing-code-discipline.md",
    ".agents/references/workflow/existing-code-discipline.md",
    ".codex/prompts/j-diff-review.md",
    ".agents/skills/cmd-j-diff-review/SKILL.md",
)

DISCIPLINE_SECTIONS = (
    "Match Existing Patterns",
    "Understand Before Deleting",
    "Separate Refactoring from Features",
    "Surface Hidden Assumptions",
    "State Your Assumptions",
)

# Trimmed from the Claude tree by #76; the other trees follow rather than diverge. Checked
# against heading names, not raw text: "Scope guard:" is live prose in both diff-review
# prompts, and a whole-file substring ban would fail the moment someone title-cased it.
DISCIPLINE_TRIMMED = ("Read Before Modifying", "Scope Guard")

ASSUMPTION_ROWS = (
    "**Data:**",
    "**Failure:**",
    "**Boundaries:**",
    "**State:**",
    "**Environment:**",
    "**Scope:**",
    "**Testing:**",
)

DESIGN_FIRST_OWNERS = (
    ".claude/skills/design-first/SKILL.md",
    ".agents/skills/design-first/SKILL.md",
    ".gemini/antigravity-cli/skills/design-first/SKILL.md",
)

# The same pointer, in the skill that writes the plan rather than the one that designs it.
WRITING_PLANS_OWNERS = (
    ".claude/skills/writing-plans/SKILL.md",
    ".agents/skills/writing-plans/SKILL.md",
    ".gemini/antigravity-cli/skills/writing-plans/SKILL.md",
)

# Bullet labels, not sentences: a reworded rationale is fine, a deleted bullet is not.
DESIGN_FIRST_BULLET = "**Filter for blocking**"
WRITING_PLANS_BULLET = "**Assumptions stated**"


def skill_directories(root: Path) -> set[str]:
    return {path.parent.name for path in root.glob("*/SKILL.md")}


def description(path: Path) -> str:
    match = FRONTMATTER_DESCRIPTION.search(path.read_text())
    if match is None:
        raise AssertionError(f"missing description: {path}")
    return match.group(1)


class AgentConfigArchitectureTests(unittest.TestCase):
    # Heading lists run past the 640-char default for 58 of 191 references, and a truncated
    # diff names no section -- which is the one thing the failure exists to tell you.
    maxDiff = None

    def test_shared_skills_do_not_contain_agent_wrappers(self):
        wrappers = sorted((REPO / ".agents" / "skills").glob("agent-*/SKILL.md"))
        self.assertEqual(wrappers, [])

    def test_command_skills_match_all_native_command_sets(self):
        command_skills = {
            name.removeprefix("cmd-")
            for name in skill_directories(REPO / ".agents" / "skills")
            if name.startswith("cmd-j-")
        }
        codex = {path.stem for path in (REPO / ".codex" / "prompts").glob("j-*.md")}
        claude = {path.stem for path in (REPO / ".claude" / "commands").glob("j-*.md")}
        antigravity = {
            name
            for name in skill_directories(REPO / ".gemini" / "antigravity-cli" / "skills")
            if name.startswith("j-")
        }

        self.assertLessEqual(command_skills, codex)
        self.assertEqual(antigravity, claude - CLAUDE_ONLY_COMMANDS)
        # Claude is compared exactly so a deleted command fails instead of passing as a subset.
        self.assertEqual(command_skills, claude - CLAUDE_ONLY_COMMANDS)

    def test_native_agent_sets_match(self):
        codex = {path.stem for path in (REPO / ".codex" / "agents").glob("*.toml")}
        claude = {path.stem for path in (REPO / ".claude" / "agents").glob("*.md")}
        antigravity = {path.stem for path in (REPO / ".gemini" / "config" / "agents").glob("*.md")}

        self.assertEqual(codex, claude - CLAUDE_ONLY_AGENTS)
        self.assertEqual(codex, antigravity)

    def test_shared_skill_descriptions_fit_codex_budget(self):
        for skill in (REPO / ".agents" / "skills").glob("*/SKILL.md"):
            value = description(skill)
            with self.subTest(skill=skill.parent.name):
                self.assertLessEqual(len(value), 64)
                self.assertTrue(value.startswith("Use when"))

    def test_workflow_skill_sets_match_claude_mirror(self):
        """Shared workflows and their Claude counterparts must cover the same set of names.

        Bodies and descriptions are intentionally allowed to diverge: `.claude/` is written for
        the Claude 5 generation while `.agents/` serves Codex and Gemini. Only membership is
        pinned, so a skill cannot silently vanish from one tree.
        """
        shared = {
            name
            for name in skill_directories(REPO / ".agents" / "skills")
            if not name.startswith("cmd-j-")
        }
        claude = skill_directories(REPO / ".claude" / "skills")

        self.assertEqual(shared - SHARED_ONLY_SKILLS, claude - CLAUDE_ONLY_SKILLS)

    def test_parity_exceptions_are_live(self):
        """Every declared exception must still exist, so stale entries cannot hide a real gap."""
        for name in CLAUDE_ONLY_COMMANDS:
            with self.subTest(command=name):
                self.assertTrue((REPO / ".claude" / "commands" / f"{name}.md").is_file())
        for name in SHARED_ONLY_SKILLS:
            with self.subTest(skill=name):
                self.assertTrue((REPO / ".agents" / "skills" / name / "SKILL.md").is_file())
                self.assertFalse((REPO / ".claude" / "skills" / name).exists())
        for name in CLAUDE_ONLY_SKILLS:
            with self.subTest(skill=name):
                self.assertTrue((REPO / ".claude" / "skills" / name / "SKILL.md").is_file())
        for name in CLAUDE_ONLY_AGENTS:
            with self.subTest(agent=name):
                self.assertTrue((REPO / ".claude" / "agents" / f"{name}.md").is_file())

    def test_workspace_selection_is_documented_where_dispatch_is_decided(self):
        for path in WORKSPACE_SELECTION_OWNERS:
            content = path.read_text().lower()
            with self.subTest(path=path.relative_to(REPO)):
                self.assertIn("## workspace selection", content)
                self.assertIn("uncommitted", content)
                self.assertIn("git worktree list", content)

    def test_worktree_skills_pin_a_start_point_and_verify_on_entry(self):
        for path in WORKTREE_BASE_OWNERS:
            content = path.read_text()
            with self.subTest(path=path.relative_to(REPO)):
                self.assertIn("start point", content.lower())
                self.assertIn("git rev-parse HEAD", content)
                # `git worktree add <path> -b <branch>` with no trailing ref uses the current HEAD.
                bare = [
                    line
                    for line in content.splitlines()
                    if line.strip().startswith("git worktree add")
                    and len(line.split("-b", 1)[1].split()) < 2
                ]
                self.assertEqual(bare, [], f"{path.name}: `git worktree add` with no start point")

    def test_always_loaded_configs_point_at_the_workspace_rule(self):
        for path in WORKTREE_POINTERS:
            content = path.read_text().lower()
            with self.subTest(path=path.relative_to(REPO)):
                self.assertIn("worktree", content)
                self.assertIn("uncommitted", content)
                self.assertIn("dispatching-parallel-agents", content)

    def test_report_contract_forbids_disk_persistence(self):
        """A dispatched agent's report is its only artifact, so the contract trades prose for
        brevity but never findings. Persisting reports to disk was considered and cut; the
        negative assertions keep it cut, because reintroducing it would need the Write tool on
        agents that are deliberately read-only, or a permissionMode that punches through plan
        mode."""
        for path in REPORT_CONTRACT_OWNERS:
            content = path.read_text()
            with self.subTest(path=path.relative_to(REPO)):
                # The completeness half -- the report is all there is, so it must not drop findings.
                self.assertIn("no length budget", content)
                self.assertIn("prose, never substance", content)
                self.assertIn("byte-for-byte", content)
                self.assertIn("`file:line`", content)
                # The cut half.
                for marker in DISK_PERSISTENCE_MARKERS:
                    self.assertNotIn(marker, content)

    def test_always_loaded_configs_point_at_the_report_contract(self):
        """Always-loaded configs carry a pointer, not a restatement -- same convention as
        WORKTREE_POINTERS, so a refactor cannot silently drop the guidance from a tree."""
        for path in REPORT_POINTERS:
            content = path.read_text()
            with self.subTest(path=path.relative_to(REPO)):
                self.assertIn("subagent-report-contract", content)

    def test_always_loaded_configs_stay_under_their_byte_ceiling(self):
        """These three are re-sent on every request, so their size is a recurring cost rather than a
        one-time one. The ceiling makes growth a decision instead of an accident."""
        for path, ceiling in ALWAYS_LOADED_CEILINGS.items():
            actual = len(path.read_bytes())
            with self.subTest(path=path.relative_to(REPO)):
                self.assertLessEqual(
                    actual,
                    ceiling,
                    f"{path.relative_to(REPO)} is {actual} bytes, over its {ceiling}-byte ceiling. "
                    "Cut something, or raise the ceiling deliberately and say why in the commit.",
                )

    def test_byte_ceiling_never_reaches_the_safety_lines(self):
        """The ceiling tells a future editor to cut something. These are not cuttable: each prevents
        an irreversible action or an injection foothold, and a missing one is silent."""
        for path, invariants in SAFETY_LINE_INVARIANTS.items():
            content = path.read_text()
            for line in invariants:
                with self.subTest(path=path.relative_to(REPO), line=line):
                    self.assertIn(
                        line,
                        content,
                        f"{path.relative_to(REPO)} lost a safety line. Cut prose elsewhere or raise "
                        "the ceiling; this one is not a trim candidate.",
                    )

    def test_ladder_exists_in_both_reference_trees_with_its_safety_carve_outs(self):
        """The rungs without the carve-outs is the unsafe half. Upstream measured a paraphrase that
        dropped them scoring 95% safe against the full ruleset's 100%, so they are pinned."""
        for path in LADDER_OWNERS:
            with self.subTest(path=path.relative_to(REPO)):
                self.assertTrue(path.is_file(), f"{path} is missing")
                content = path.read_text()
                for invariant in LADDER_SAFETY_INVARIANTS:
                    self.assertIn(invariant, content)

    def test_ladder_consumers_point_at_it_rather_than_restating_it(self):
        """Each surface where a code decision is made names the ladder. Pointer, not copy: the rungs
        must appear in exactly one place per tree or they drift."""
        for path in LADDER_CONSUMERS:
            with self.subTest(path=path.relative_to(REPO)):
                self.assertTrue(path.is_file(), f"{path} is missing")
                content = path.read_text()
                self.assertIn("code-efficiency-ladder", content)
                for marker in LADDER_RUNG_MARKERS:
                    self.assertNotIn(
                        marker,
                        content,
                        f"{path.relative_to(REPO)} restates the ladder instead of pointing at it. "
                        "Cut the copy; the reference is the one owner.",
                    )

    def test_adr_consumers_carry_the_edit_in_place_rule(self):
        """Every surface that acts on an existing ADR says to update it in place. Nothing else in
        the suite reads these files for ADR wording, so without this the whole model reverts to
        amend-or-supersede with the suite green."""
        for path in ADR_EDIT_IN_PLACE_CONSUMERS:
            with self.subTest(path=path.relative_to(REPO)):
                rel = path.relative_to(REPO)
                self.assertTrue(path.is_file(), f"{path} is missing")
                content = path.read_text(encoding="utf-8")
                self.assertIn(
                    "in place",
                    content,
                    f"{rel} acts on an existing ADR but no longer says to update it in place",
                )
                for term in ADR_RETIRED_VOCABULARY:
                    self.assertNotIn(
                        term,
                        content,
                        f"{rel} teaches the retired ADR lifecycle ({term!r}). A changed decision is "
                        "edited in place and a dead one is deleted.",
                    )

    def test_simplified_sanction_never_ships_without_its_security_carve_out(self):
        """The marker tells a reviewer to stand down. Every surface that says so must also say the
        sanction stops at a trust boundary, or reviewed code can silence its own review."""
        for path in SIMPLIFIED_SANCTION_SURFACES:
            with self.subTest(path=path.relative_to(REPO)):
                self.assertTrue(path.is_file(), f"{path} is missing")
                content = path.read_text()
                self.assertIn("// SIMPLIFIED:", content)
                self.assertIn(
                    "trust boundary",
                    content,
                    f"{path.relative_to(REPO)} sanctions `// SIMPLIFIED:` without naming the "
                    "security carve-out. State it here or drop the sanction.",
                )

    def test_structure_rule_has_one_owner_carrying_the_conditional(self):
        """The owner states BOTH halves: the density ordering, and that structure the question did
        not ask for is a net cost anyway. Half the rule reads as a licence for the other half."""
        for path in STRUCTURE_RULE_OWNERS:
            with self.subTest(path=path.relative_to(REPO)):
                self.assertTrue(path.is_file(), f"{path} is missing")
                # Collapse wrapping first: a reflow must not read as the rule going missing.
                content = " ".join(path.read_text().split())
                self.assertIn("per unit of information carried", content)
                self.assertIn("Answer at the Question's Altitude", content)

    def test_always_loaded_configs_do_not_restate_the_unconditional_structure_rule(self):
        """A measured-wrong rule must not survive in a file re-sent on every request. The configs
        point at context-efficiency instead, same convention as REPORT_POINTERS."""
        for path in REPORT_POINTERS:
            with self.subTest(path=path.relative_to(REPO)):
                self.assertTrue(path.is_file(), f"{path} is missing")
                content = path.read_text()
                for phrase in UNCONDITIONAL_STRUCTURE_PHRASES:
                    self.assertNotIn(
                        phrase,
                        content,
                        f"{path.relative_to(REPO)} still carries the unconditional structure rule. "
                        "It points at context-efficiency instead.",
                    )
                self.assertIn("context-efficiency", content)
                # These files are synced to ~. Claude resolves a reference by bare name, but where
                # one of them writes a PATH it has to be home-anchored -- `references/...` on its
                # own resolves to nothing from the home directory.
                self.assertNotIn(
                    "`references/",
                    content,
                    f"{path.relative_to(REPO)} writes an unanchored reference path. "
                    "Use `~/.agents/references/...`, which is where the tree actually lands.",
                )

    def test_diff_review_hands_reviewers_the_head_sha(self):
        for path in DIFF_REVIEW_DISPATCHERS:
            content = path.read_text()
            with self.subTest(path=path.relative_to(REPO)):
                self.assertIn("git rev-parse HEAD", content)
                self.assertIn("HEAD SHA", content)
                self.assertIn("do not create or request a worktree", content)

    def test_delegation_ladders_name_no_pinned_model(self):
        """A ladder rung that names a model version goes stale every generation, so rungs describe
        tiers by role. Scoped to bullets on purpose: GEMINI.md's prose names a slug as a worked
        `--model` example and ships `agy models` beside it as the freshness pointer, which is the
        staleness problem already solved rather than an instance of it. The regex needs a digit, so
        floating aliases (opus, sonnet, haiku, fable, flash, pro) stay legal."""
        for path in ALWAYS_LOADED_CEILINGS:
            pinned = [
                line
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.lstrip().startswith(("-", "*")) and PINNED_MODEL.search(line)
            ]
            with self.subTest(path=path.relative_to(REPO)):
                self.assertEqual(
                    pinned, [], f"{path.name} pins a model version; use a role or floating alias"
                )

    def test_pr_workflows_report_the_url_and_reuse_the_open_pr(self):
        """Every asset that ends on a PR reports its URL, and never opens a second PR for a branch
        that already has one -- the guard `finishing-branch` has and these did not."""
        for path in PR_URL_SURFACES:
            body = path.read_text(encoding="utf-8")
            rel = path.relative_to(REPO)
            with self.subTest(path=rel):
                self.assertTrue("PR URL" in body, f"{rel} never reports the PR URL")
                if path.stem == "create-pr":
                    # Codex agents are prose and name no shell commands, so the guard is stated
                    # rather than scripted there. Both forms have to say a PR may already exist.
                    self.assertTrue(
                        "already open" in body, f"{rel}: no open-PR guard before creating one"
                    )
                    if path.suffix == ".md":
                        self.assertTrue(
                            "gh pr view" in body, f"{rel}: guard names no lookup command"
                        )

    def test_always_loaded_configs_require_reporting_the_pr_url(self):
        """ "Provide the PR link when done" is a standing rule, not a diff-review-only one. Each
        harness config states it where its Git rules live."""
        for path in ALWAYS_LOADED_CEILINGS:
            rel = path.relative_to(REPO)
            with self.subTest(path=rel):
                body = path.read_text(encoding="utf-8")
                self.assertTrue("PR URL" in body, f"{rel} never asks for the PR URL")

    def test_diff_review_lands_fixes_on_the_pr(self):
        """A rung-1 fix that stays a local commit is a finding the reviewer never sees. Every tree's
        diff-review ends by pushing to the branch's open PR and reporting its URL."""
        for path in DIFF_REVIEW_DISPATCHERS:
            body = path.read_text(encoding="utf-8")
            rel = path.relative_to(REPO)
            with self.subTest(path=rel):
                self.assertTrue("gh pr view" in body, f"{rel}: no open-PR lookup after the ladder")
                self.assertTrue("PR URL" in body, f"{rel}: the run never reports the PR URL")
                self.assertTrue("Steps 1-5" in body, f"{rel}: the gh ban is unscoped")
                # The push is conditional on a rung-1 commit; reporting the link never is. Gating
                # both on the same condition is the regression -- a clean review then ends with no
                # link, which is the behavior this step was added to remove.
                self.assertTrue(
                    "unconditional" in body, f"{rel}: the PR link report is gated on a fix existing"
                )

    def test_diff_review_maps_extensions_to_language_references(self):
        """Deleting the duplicate workflow from `code-review-patterns` also removed the only
        extension-to-reference mapping the Claude and Gemini commands had; their Step 3 named a
        directory of 20+ files instead. Every tree carries the mapping in its own Step 3 now."""
        for path in DIFF_REVIEW_DISPATCHERS:
            body = path.read_text(encoding="utf-8")
            rel = path.relative_to(REPO)
            for ext, ref in (
                (".py", "python-patterns.md"),
                (".js", "js-ts-patterns.md"),
                (".ts", "js-ts-patterns.md"),
                (".tsx", "js-ts-patterns.md"),
                (".go", "go-concurrency-patterns.md"),
                (".sh", "bash-defensive-patterns.md"),
                (".swift", "swift-patterns.md"),
                (".rs", "rust-project-patterns.md"),
            ):
                with self.subTest(path=rel, ext=ext):
                    self.assertTrue(
                        f"`{ext}`" in body, f"{rel}: Step 3 maps no reference for {ext}"
                    )
                    # Naming the file, not a prose label like "Python patterns": the Codex and
                    # `.agents` trees have no skill loader, so a label resolves to nothing there.
                    self.assertTrue(
                        f"`{ref}`" in body, f"{rel}: {ext} maps to no resolvable reference file"
                    )
                    self.assertTrue(
                        (REPO / ".agents" / "references" / "languages" / ref).is_file(),
                        f"{rel}: Step 3 names {ref}, which the repo does not ship",
                    )

    def test_one_diff_review_workflow_per_tree(self):
        """`code-review-patterns` used to carry a second diff-review workflow whose Step 6 said
        report-only while the command's said fix-and-commit. The command owns the workflow; the
        skill owns mindset, severity labels, and feedback. The two inlined command bodies keep a
        disposition ladder because there the section *is* the command."""
        inlined = (
            REPO / ".codex" / "prompts" / "j-diff-review.md",
            REPO / ".agents" / "skills" / "cmd-j-diff-review" / "SKILL.md",
        )
        checked = 0
        for tree in TREE_ROOTS:
            for path in sorted((REPO / tree).rglob("code-review-patterns/SKILL.md")):
                body = path.read_text(encoding="utf-8")
                with self.subTest(path=path.relative_to(REPO)):
                    rel = path.relative_to(REPO)
                    self.assertFalse(
                        "Pre-Submission Diff Review" in body,
                        f"{rel} still carries the duplicate diff-review workflow",
                    )
                    self.assertFalse(
                        "Decision Gate" in body,
                        f"{rel} still carries the report-only Step 6",
                    )
                checked += 1
        self.assertGreater(checked, 0, "no code-review-patterns copies compared; check is vacuous")
        for path in inlined:
            with self.subTest(path=path.relative_to(REPO)):
                self.assertIn("Disposition Ladder", path.read_text(encoding="utf-8"))

    def test_branch_workflows_resolve_their_base_through_the_remote(self):
        for path in BRANCH_BASE_REF_DOCS:
            lines = path.read_text().splitlines()
            with self.subTest(path=path.relative_to(REPO)):
                self.assertTrue(
                    any("git merge-base HEAD origin/main" in line for line in lines),
                    f"{path.name}: no origin/main merge-base found",
                )
                stale = [line.strip() for line in lines if "git merge-base HEAD main" in line]
                self.assertEqual(stale, [], f"{path.name}: merge-base against local main")

    def test_shared_skills_carry_no_upstream_superpowers_paths(self):
        """`.agents/` was seeded from obra/superpowers; its paths and cross-reference
        syntax are ours now. REFERENCES.md keeps the upstream attribution."""
        for path in sorted((REPO / ".agents" / "skills").glob("*/SKILL.md")):
            leftovers = [
                line.strip() for line in path.read_text().splitlines() if "superpowers" in line
            ]
            with self.subTest(skill=path.parent.name):
                self.assertEqual(leftovers, [], f"{path.parent.name}: upstream leftover")

    def test_bash_guard_is_registered_on_every_harness_that_has_hooks(self):
        """The guard only helps if it is wired. Claude reads `hooks.PreToolUse`; Codex reads
        `[[hooks.PreToolUse]]`. Gemini has no hook surface, so it gets none -- see
        `references/workflow/hook-patterns.md`, Per-Harness Support.

        Both registrations invoke the INSTALLED copy under `~`, matching how log-prompt.sh is
        invoked, because this config is synced to `~` and must work in every repo.
        """
        settings = json.loads((REPO / ".claude" / "settings.json").read_text())
        pre = settings.get("hooks", {}).get("PreToolUse", [])
        self.assertTrue(pre, ".claude/settings.json declares no PreToolUse hook")
        commands = [h["command"] for entry in pre for h in entry.get("hooks", [])]
        self.assertTrue(
            any("guard-bash-output.sh" in c and "--format claude" in c for c in commands),
            f"no Claude-format guard registration in {commands}",
        )
        self.assertEqual([e.get("matcher") for e in pre], ["Bash"])

        codex = (REPO / ".codex" / "config.toml").read_text()
        self.assertIn("[[hooks.PreToolUse]]", codex)
        self.assertIn("guard-bash-output.sh --format codex", codex)
        self.assertIn('matcher = "^Bash$"', codex)

    def test_prompt_logging_is_registered_on_every_harness_that_has_hooks(self):
        """`.codex/hooks/log-prompt.sh` shipped and synced since the tree was created, but
        `.codex/config.toml` had no `[hooks]` table, so it never fired. Its Claude twin was
        registered the whole time. Pin both so the asymmetry cannot come back."""
        settings = json.loads((REPO / ".claude" / "settings.json").read_text())
        claude = [
            h["command"]
            for entry in settings.get("hooks", {}).get("UserPromptSubmit", [])
            for h in entry.get("hooks", [])
        ]
        self.assertTrue(any("log-prompt.sh" in c for c in claude), claude)

        codex = (REPO / ".codex" / "config.toml").read_text()
        self.assertIn("[[hooks.UserPromptSubmit]]", codex)
        self.assertIn("log-prompt.sh", codex)

    def test_guard_hook_ships_identically_in_both_hook_trees(self):
        """One script, two registrations -- the only difference is the --format flag at call time.
        Divergent copies would mean one harness silently gets different advice."""
        claude = REPO / ".claude" / "hooks" / "guard-bash-output.sh"
        codex = REPO / ".codex" / "hooks" / "guard-bash-output.sh"
        for path in (claude, codex):
            with self.subTest(path=path.relative_to(REPO)):
                self.assertTrue(path.is_file())
        self.assertEqual(claude.read_bytes(), codex.read_bytes())

    def test_worktree_base_ref_is_pinned_to_local_head(self):
        """`fresh`, the harness default, branches agent-isolation worktrees off origin/<default>."""
        settings = json.loads((REPO / ".claude" / "settings.json").read_text())
        self.assertEqual(settings.get("worktree", {}).get("baseRef"), "head")

    def test_cursor_configuration_is_not_tracked(self):
        cursor = REPO / ".cursor"
        self.assertFalse(cursor.exists() and any(path.is_file() for path in cursor.rglob("*")))

    def test_j_plan_commands_persist_the_plan_outside_git(self):
        ignore_patterns = {
            line.strip()
            for line in (REPO / ".gitignore").read_text().splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        }
        self.assertIn("scratchpad/", ignore_patterns)

        for path in J_PLAN_COMMANDS:
            content = path.read_text()
            with self.subTest(path=path.relative_to(REPO)):
                self.assertIn("scratchpad/plans/", content)
                self.assertIn("git check-ignore -q scratchpad/", content)
                self.assertIn("${TMPDIR:-/tmp}/j-plan/<repo-id>/", content)
                self.assertIn("stable SHA-256 digest", content)
                self.assertIn("Create the parent directory and file immediately", content)
                self.assertIn("0700", content)
                self.assertIn("0600", content)
                self.assertRegex(content, r"exclusive,\s+no-clobber")
                self.assertIn("numeric suffix", content)
                self.assertRegex(content, r"owned by (?:the )?current user")
                self.assertIn("not symlinks", content)
                self.assertIn("Status: Researching", content)
                self.assertIn("## Planning Notes", content)
                self.assertRegex(content, r"source\s+of truth")
                self.assertIn("MUST NOT keep the only copy in context", content)

    def test_plan_storage_skills_forbid_context_only_plans(self):
        for path in PLAN_STORAGE_SKILLS:
            content = path.read_text()
            with self.subTest(path=path.relative_to(REPO)):
                self.assertIn("scratchpad/plans/", content)
                self.assertIn("git check-ignore -q scratchpad/", content)
                self.assertIn("${TMPDIR:-/tmp}/j-plan/<repo-id>/", content)
                self.assertIn("0700", content)
                self.assertIn("0600", content)
                self.assertRegex(content, r"exclusive,\s+no-clobber")
                self.assertRegex(content, r"owned by (?:the )?current user")
                self.assertIn("not symlinks", content)
                self.assertIn("MUST NOT keep the only copy in context", content)

    def test_plan_execution_commands_confirm_discovered_files(self):
        for path in PLAN_EXECUTION_CONSUMERS:
            content = path.read_text()
            with self.subTest(path=path.relative_to(REPO)):
                self.assertIn("scratchpad/plans/", content)
                self.assertIn("${TMPDIR:-/tmp}/j-plan/<repo-id>/", content)
                self.assertIn("regular, non-symlink plan files", content)
                self.assertIn("full paths", content)
                self.assertIn("modification times", content)
                self.assertIn("even when there is only one", content)
                self.assertIn("MUST NOT execute a discovered plan without confirmation", content)

    def test_active_plan_commands_use_the_persisted_locations(self):
        for path in ACTIVE_PLAN_CONSUMERS:
            content = path.read_text()
            with self.subTest(path=path.relative_to(REPO)):
                self.assertIn("scratchpad/plans/", content)
                self.assertIn("${TMPDIR:-/tmp}/j-plan/<repo-id>/", content)

    def test_workflow_skill_sets_match_gemini_antigravity(self):
        """Shared workflows and their Gemini/Antigravity counterparts must cover the same set of names."""
        shared = {
            name
            for name in skill_directories(REPO / ".agents" / "skills")
            if not name.startswith("cmd-j-")
        }
        gemini = {
            name
            for name in skill_directories(REPO / ".gemini" / "antigravity-cli" / "skills")
            if not name.startswith("j-")
        }

        self.assertEqual(shared, gemini)

    def test_gemini_skills_do_not_contain_agent_wrappers_or_commands(self):
        gemini_root = REPO / ".gemini" / "antigravity-cli" / "skills"
        wrappers = sorted(gemini_root.glob("agent-*/SKILL.md"))
        commands = sorted(gemini_root.glob("cmd-*/SKILL.md"))
        self.assertEqual(wrappers, [])
        self.assertEqual(commands, [])

    def test_no_tree_names_a_script_it_does_not_ship(self):
        """`.gemini/.../j-new` told the reader to run an `audit.py` under
        `~/.gemini/antigravity-cli/skills/skill-audit/scripts/`. That script exists only in the
        Claude tree, so the instruction was dead the day it was written and nothing caught it.

        Tracked files only. `rglob` also picks up gitignored working state such as
        `.claude/settings.local.json`, which makes the input set differ between this machine and a
        clean checkout -- and that file currently names a `.gemini/hooks/` script that does not
        exist. The `assertGreater` is the positive control: without it, an over-narrow regex, an
        empty root list, or a wrong suffix filter all pass as silently as a clean tree.
        """
        tracked = subprocess.run(
            ["git", "ls-files", "-z", *TREE_ROOTS],
            cwd=REPO,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.split("\0")
        missing = []
        scanned = []
        claims = 0
        for name in tracked:
            path = REPO / name
            if path.suffix not in {".md", ".toml", ".json"} or not path.is_file():
                continue
            scanned.append(name)
            for claimed in TREE_SCRIPT_PATH.findall(path.read_text(encoding="utf-8")):
                claims += 1
                if not (REPO / claimed).is_file():
                    missing.append(f"{name}: {claimed}")
        self.assertGreater(claims, 0, "regex matched no script path at all; the check is vacuous")
        # `git ls-files` with no pathspec lists the whole repo, so an empty TREE_ROOTS would widen
        # the scan rather than empty it, and a bare `for root in TREE_ROOTS` would then assert
        # nothing at all.
        self.assertTrue(TREE_ROOTS, "TREE_ROOTS is empty; the per-root check below cannot fire")
        for root in TREE_ROOTS:
            self.assertTrue(
                any(name.startswith(root + "/") for name in scanned),
                f"{root} contributed no scanned file; TREE_ROOTS is not constraining the scan",
            )
        self.assertEqual(
            missing, [], "assets name scripts that do not exist:\n" + "\n".join(missing)
        )

    def test_codex_agents_declare_name_description_and_instructions(self):
        """A Codex agent with no `developer_instructions` loads as an empty role and says nothing
        about it. These three fields are the whole contract, and TOML has no frontmatter validator
        here the way `audit.py` validates the Claude tree."""
        agents = sorted((REPO / ".codex" / "agents").glob("*.toml"))
        self.assertGreater(len(agents), 0, "must have Codex agents in .codex/agents")
        for path in agents:
            content = path.read_text()
            name = re.compile(rf'^name = "{re.escape(path.stem)}"$', re.MULTILINE)
            with self.subTest(path=path.relative_to(REPO)):
                self.assertRegex(content, name)
                self.assertRegex(content, CODEX_DESCRIPTION)
                self.assertRegex(content, CODEX_INSTRUCTIONS)

    def test_gemini_agents_inherit_the_parent_model(self):
        """Settled in #93. A per-agent tier in the Gemini tree diverges silently from the Claude
        tree, which is the one that chooses tiers."""
        agents = sorted((REPO / ".gemini" / "config" / "agents").glob("*.md"))
        self.assertGreater(len(agents), 0, "must have Gemini agents in .gemini/config/agents")
        for path in agents:
            with self.subTest(path=path.relative_to(REPO)):
                self.assertRegex(path.read_text(), GEMINI_INHERIT)

    def test_antigravity_agents_have_valid_frontmatter(self):
        """All Antigravity agents in .gemini/config/agents/ must declare subagent: true and mainAgent: true."""
        agents_dir = REPO / ".gemini" / "config" / "agents"
        agents = sorted(agents_dir.glob("*.md"))
        self.assertGreater(len(agents), 0, "must have Antigravity agents in .gemini/config/agents")
        for agent_file in agents:
            content = agent_file.read_text()
            name = re.compile(rf"^name: {re.escape(agent_file.stem)}$", re.MULTILINE)
            self.assertIn("subagent: true", content, f"{agent_file.name} missing subagent: true")
            self.assertIn("mainAgent: true", content, f"{agent_file.name} missing mainAgent: true")
            self.assertRegex(content, name, f"{agent_file.name} name does not match its filename")
            self.assertRegex(content, GEMINI_DESCRIPTION, f"{agent_file.name} description is empty")

    def test_antigravity_permissions_cover_claude_baseline(self):
        """Antigravity settings.json permissions must cover all Claude allowed commands and denied paths."""
        claude_settings = json.loads((REPO / ".claude" / "settings.json").read_text())
        agy_settings_path = REPO / ".gemini" / "antigravity-cli" / "settings.json"
        self.assertTrue(agy_settings_path.is_file(), "antigravity-cli/settings.json must exist")
        agy_settings = json.loads(agy_settings_path.read_text())

        claude_allow = set(claude_settings.get("permissions", {}).get("allow", []))
        agy_allow = set(agy_settings.get("permissions", {}).get("allow", []))
        claude_deny = set(claude_settings.get("permissions", {}).get("deny", []))
        agy_deny = set(agy_settings.get("permissions", {}).get("deny", []))

        # Antigravity uses literal word-by-word prefix matching without trailing ' *'
        for rule in agy_allow | agy_deny:
            self.assertFalse(
                rule.endswith(" *)"),
                f"Antigravity rule {rule} has trailing ' *' which is invalid prefix syntax",
            )
            self.assertFalse(
                "/**)" in rule,
                f"Antigravity rule {rule} has trailing '/**' which is invalid path syntax",
            )

        # Check command translations: Bash(cmd) -> command(prefix)
        for rule in claude_allow:
            if rule.startswith("Bash(") and rule.endswith(")"):
                cmd = rule[len("Bash(") : -1].removesuffix(" *")
                if cmd == "grep:*":
                    cmd = "grep"
                expected = f"command({cmd})"
                self.assertIn(expected, agy_allow, f"Missing allowed command: {expected}")

        for rule in claude_deny:
            if rule.startswith("Bash(") and rule.endswith(")"):
                cmd = rule[len("Bash(") : -1].removesuffix(" *")
                expected = f"command({cmd})"
                self.assertIn(expected, agy_deny, f"Missing denied command: {expected}")
            elif rule.startswith("Read(") and rule.endswith(")"):
                path = rule[len("Read(") : -1].removesuffix("/**")
                if path.startswith("./"):
                    continue  # non-absolute paths are rejected by the Antigravity sandbox
                expected = f"read_file({path})"
                self.assertIn(expected, agy_deny, f"Missing denied read_file: {expected}")

    def test_legacy_gemini_artifacts_are_not_tracked(self):
        """Legacy Gemini CLI policy engine, hooks, commands, and agents must not be present in .gemini/."""
        self.assertFalse(
            (REPO / ".gemini" / "policies").exists(), "legacy policies/ must not exist"
        )
        self.assertFalse((REPO / ".gemini" / "hooks").exists(), "legacy hooks/ must not exist")
        self.assertFalse(
            (REPO / ".gemini" / "settings.json").exists(),
            "root .gemini/settings.json must not exist",
        )
        self.assertFalse(
            (REPO / ".gemini" / "commands").exists(), "legacy commands/ must not exist"
        )
        self.assertFalse((REPO / ".gemini" / "agents").exists(), "legacy agents/ must not exist")

    def test_reference_trees_hold_the_same_files(self):
        """The section check below walks the Claude tree, so a `.claude`-only file fails loudly
        there. The reverse does not: an `.agents`-only reference is loaded by Codex and Gemini and
        checked by nobody, because audit.py never reads that tree. This closes that direction."""
        claude, agents = REFERENCE_TREES
        self.assertTrue(
            claude.is_dir(), f"{claude} is missing; every parity check would pass vacuously"
        )
        self.assertTrue(
            agents.is_dir(), f"{agents} is missing; every parity check would pass vacuously"
        )
        claude_files = {path.relative_to(claude).as_posix() for path in claude.rglob("*.md")}
        agents_files = {path.relative_to(agents).as_posix() for path in agents.rglob("*.md")}
        self.assertGreater(len(claude_files), 0, "no references found at all; the check is vacuous")
        self.assertEqual(claude_files, agents_files, "the two reference trees hold different files")

    def test_reference_trees_expose_the_same_sections(self):
        """Bodies may differ -- `Explore` in the Claude tree is "a dispatched search subagent" in
        the shared one, because Codex and Gemini have no tool by that name. Section sets may not:
        a heading in one tree and not the other means a reader of the other tree cannot find the
        topic. See docs/adr/workflow/reference-tree-section-parity.md."""
        claude, agents = REFERENCE_TREES
        compared = 0
        for claude_file in sorted(claude.rglob("*.md")):
            rel = claude_file.relative_to(claude)
            agents_file = agents / rel
            with self.subTest(reference=str(rel)):
                self.assertTrue(agents_file.is_file(), f"{rel} missing from .agents/references/")
                self.assertEqual(
                    markdown_headings(claude_file.read_text(encoding="utf-8")),
                    markdown_headings(agents_file.read_text(encoding="utf-8")),
                    f"{rel}: heading sets diverge between trees",
                )
            compared += 1
        self.assertGreater(compared, 0, "no reference pairs compared; the check is vacuous")

    def test_shared_references_do_not_cite_the_claude_reference_tree(self):
        """Codex and Gemini read .agents/ and have no .claude/ checkout, so a citation naming
        that tree resolves nowhere. audit.py is Claude-only and never sees these.

        Scoped to `.claude/references/` on purpose: permission-management.md, hook-patterns.md,
        and context-efficiency.md document Claude Code's own settings and config layout, so
        `.claude/settings.json` and `.claude/CLAUDE.md` are correct content there rather than
        dangling pointers. Scoped to references/ rather than all of .agents/ for the same reason
        -- skill-audit names the Claude tree because auditing it is the skill's whole job."""
        _, agents = REFERENCE_TREES
        offenders = []
        scanned = 0
        for agents_file in sorted(agents.rglob("*.md")):
            scanned += 1
            rel = agents_file.relative_to(REPO)
            lines = agents_file.read_text(encoding="utf-8").splitlines()
            offenders += [
                f"{rel}:{n}: {line.strip()}"
                for n, line in enumerate(lines, 1)
                if ".claude/references/" in line
            ]
        self.assertGreater(scanned, 0, "no shared references scanned; the check is vacuous")
        self.assertEqual(
            offenders, [], "shared references cite the Claude tree:\n" + "\n".join(offenders)
        )

    def test_existing_code_discipline_is_one_document_in_every_tree(self):
        """Four copies, one section set and one body. Two paste the text rather than linking it,
        so an edit to the reference alone leaves the diff-review prompts stale -- and comparing
        section names alone would not notice, because the prose is where the guidance lives."""
        bodies = {}
        for rel in EXISTING_CODE_DISCIPLINE_OWNERS:
            path = REPO / rel
            with self.subTest(owner=rel):
                self.assertTrue(path.is_file(), f"{rel} is missing")
                text = path.read_text(encoding="utf-8")
                names = heading_texts(text)
                for section in DISCIPLINE_SECTIONS:
                    self.assertIn(section, names, f"{rel} has no section named {section!r}")
                for section in DISCIPLINE_TRIMMED:
                    self.assertNotIn(
                        section, names, f"{rel} still has section {section!r}, trimmed in #76"
                    )
                missing = [row for row in ASSUMPTION_ROWS if row not in text]
                self.assertEqual(missing, [], f"{rel} is missing taxonomy rows: {missing}")
                bodies[rel] = discipline_body(text)
        source = EXISTING_CODE_DISCIPLINE_OWNERS[0]
        self.assertGreater(
            len(bodies[source]), 0, f"{source} yielded no discipline prose; the check is vacuous"
        )
        for rel in EXISTING_CODE_DISCIPLINE_OWNERS[1:]:
            with self.subTest(owner=rel):
                self.assertEqual(bodies[source], bodies[rel], f"{rel} has drifted from {source}")

    def test_design_first_and_writing_plans_reach_the_taxonomy(self):
        """A recommended answer makes a question cheap to answer; the filter is what keeps the
        count down. The pointer is what puts the taxonomy in front of the work at all -- neither
        skill indexes a reference otherwise. Pinned by bullet label rather than by sentence, so
        rewording the rationale is free and deleting the bullet is not."""
        checked = 0
        for owners, bullet in (
            (DESIGN_FIRST_OWNERS, DESIGN_FIRST_BULLET),
            (WRITING_PLANS_OWNERS, WRITING_PLANS_BULLET),
        ):
            self.assertTrue(bullet, "bullet label is empty; the check would be vacuous")
            for rel in owners:
                checked += 1
                with self.subTest(skill=rel):
                    path = REPO / rel
                    self.assertTrue(path.is_file(), f"{rel} is missing")
                    text = path.read_text(encoding="utf-8")
                    tree = "claude" if rel.split("/", 1)[0] == ".claude" else "agents"
                    pointer = f".{tree}/references/workflow/existing-code-discipline.md"
                    self.assertTrue(bullet in text, f"{rel} has no {bullet} bullet")
                    self.assertTrue(pointer in text, f"{rel} does not point at {pointer}")
        self.assertEqual(
            checked,
            len(DESIGN_FIRST_OWNERS) + len(WRITING_PLANS_OWNERS),
            "owner lists shrank; the check is vacuous",
        )

    def test_shared_skills_are_byte_identical_in_the_gemini_tree(self):
        """Gemini loads its own copy of every shared skill and nothing rewrites them on sync, so
        the pair is supposed to be the same bytes. `cmd-j-*` is excluded: those are commands, and
        each harness adapts their body. A convention 30 skills keep and no test held."""
        agents_skills = REPO / ".agents" / "skills"
        gemini_skills = REPO / ".gemini" / "antigravity-cli" / "skills"
        compared = 0
        for source in sorted(agents_skills.glob("*/SKILL.md")):
            name = source.parent.name
            if name.startswith("cmd-j-"):
                continue
            mirror = gemini_skills / name / "SKILL.md"
            with self.subTest(skill=name):
                self.assertTrue(mirror.is_file(), f"{name} has no Gemini copy")
                self.assertEqual(
                    source.read_text(encoding="utf-8"),
                    mirror.read_text(encoding="utf-8"),
                    f"{name}: the .agents and .gemini copies have diverged",
                )
            compared += 1
        self.assertGreater(compared, 0, "no shared skills compared; the check is vacuous")


if __name__ == "__main__":
    unittest.main()
