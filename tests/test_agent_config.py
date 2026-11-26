"""Architecture checks for the tool-specific agent configuration trees."""

from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
import unittest
from pathlib import Path

from macos_setup.dotfiles import EXPANDED_SUFFIXES, HOME_ANCHORS
from tests.markdown import (
    HEADING,
    has_unterminated_fence,
    heading_texts,
    lines_outside_fences,
    markdown_headings,
)

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
SKILL_DIR_SCRIPT_PATH = re.compile(r"\$\{CLAUDE_SKILL_DIR\}/([\w./-]+\.(?:py|sh|mjs|js))")
TREE_ROOTS = (".claude", ".codex", ".agents", ".gemini")
CODE_SPAN = re.compile(r"(`+).+?\1")
TABLE_PLACEHOLDER = re.compile(r"\|\s*--\s*(?=\|)")
PROSE_DOUBLE_HYPHEN = re.compile(r"(?<=\S)\s+--(?=\s|$)")
EM_DASH = "\u2014"

# Side-effect commands nothing invokes; user-only in Claude and Codex
# (docs/adr/workflow/side-effect-commands-are-user-invoked.md).
USER_ONLY_COMMANDS = frozenset({"j-defrag", "j-pr-fix", "j-review-claudemd", "j-tdd"})

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
    REPO / ".gemini" / "antigravity-cli" / "skills" / "j-plan" / "SKILL.md",
)

CLAUDE_J_PLAN_COMMAND = REPO / ".claude" / "commands" / "j-plan.md"

PLAN_STORAGE_SKILLS = (REPO / ".agents" / "skills" / "writing-plans" / "SKILL.md",)

PLAN_EXECUTION_CONSUMERS = (
    REPO / ".agents" / "skills" / "cmd-j-execute-plan" / "SKILL.md",
    REPO / ".agents" / "skills" / "cmd-j-next" / "SKILL.md",
    REPO / ".gemini" / "antigravity-cli" / "skills" / "j-execute-plan" / "SKILL.md",
    REPO / ".gemini" / "antigravity-cli" / "skills" / "j-next" / "SKILL.md",
)

# Owns "inline or subagents" for plan execution; every consumer above points here.
EXECUTION_MODE_OWNERS = tuple(
    root / "subagent-driven-development" / "SKILL.md"
    for root in (
        REPO / ".claude" / "skills",
        REPO / ".agents" / "skills",
        REPO / ".gemini" / "antigravity-cli" / "skills",
    )
)

# The pre-change handoff menu; none of it may come back once the executor picks the mode.
MODE_MENU_PHRASES = ("Which approach?", "Execute in new session", "a new session, or")

CLAUDE_PLAN_EXECUTION_COMMANDS = (
    REPO / ".claude" / "commands" / "j-execute-plan.md",
    REPO / ".claude" / "commands" / "j-next.md",
)

PLAN_WORKFLOW_SKILLS = tuple(
    root / skill / "SKILL.md"
    for root in (
        REPO / ".claude" / "skills",
        REPO / ".agents" / "skills",
        REPO / ".gemini" / "antigravity-cli" / "skills",
    )
    for skill in ("executing-plans", "writing-plans")
)

MID_PR_PAUSE_PHRASES = (
    "Wait for user response",
    "Ready for feedback",
    "review checkpoints",
    "executor pauses",
)

ACTIVE_PLAN_CONSUMERS = (
    REPO / ".agents" / "skills" / "cmd-j-diff-review" / "SKILL.md",
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
    REPO / ".gemini" / "antigravity-cli" / "skills" / "j-arch" / "SKILL.md",
    REPO / ".claude" / "skills" / "post-ship-doc-sync" / "SKILL.md",
    REPO / ".agents" / "skills" / "post-ship-doc-sync" / "SKILL.md",
    REPO / ".gemini" / "antigravity-cli" / "skills" / "post-ship-doc-sync" / "SKILL.md",
    REPO / ".claude" / "commands" / "j-plan.md",
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
# The user chose these exact Codex models. Keep the exception narrow: a dated variant or another
# generation must still fail the stale-model guard, as must any name in another tool's ladder.
CODEX_SELECTED_MODELS = re.compile(
    r"(?<![\w.-])(?:gpt-6\.2-(?:astra|sol)|gpt-6\.1-terra|gpt-6-luna)(?![\w.-])"
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
    REPO / ".gemini" / "antigravity-cli" / "skills" / "j-diff-review" / "SKILL.md",
)

SKILL_TREE_ROOTS = (
    REPO / ".claude" / "skills",
    REPO / ".agents" / "skills",
    REPO / ".gemini" / "antigravity-cli" / "skills",
)

SHARED_SKILL_ROOTS = (
    REPO / ".agents" / "skills",
    REPO / ".gemini" / "antigravity-cli" / "skills",
)
SHARED_SKILL_FRONTMATTER_KEYS = frozenset({"name", "description"})

# Credit forms, not source names: a name list goes stale and collides with ordinary words.
# `(?!Author\b)` spares placeholders: "First Author et al." in the j-email-analysis templates and
# "(Author, 2024)" in the latex-paper-writing citation gotcha.
SOURCE_CREDIT_PATTERNS = (
    re.compile(r"^\s*(?:[-*>]\s*)?(?:\*\*)?Source:(?:\*\*)?\s"),
    re.compile(r"^\s*(?:[-*>]\s*)?Adapted from "),
    re.compile(r"\(from [A-Z][a-z]+'s "),
    re.compile(r"\b(?!Author\b)[A-Z][a-z]+ et al\."),
    re.compile(r"\((?!Author\b)[A-Z][a-z]+(?: et al\.)?, (?:19|20)\d{2}\)"),
    re.compile(r"\((?:per|from) \[[^\]]*\]\(https?://"),
)

# Both execution modes close a plan boundary through the diff review, whose Step 7 opens the PR.
EXECUTION_CLOSE_OWNERS = tuple(
    root / skill / "SKILL.md"
    for root in SKILL_TREE_ROOTS
    for skill in ("executing-plans", "subagent-driven-development")
)

J_NEXT_COMMANDS = (
    REPO / ".claude" / "commands" / "j-next.md",
    REPO / ".agents" / "skills" / "cmd-j-next" / "SKILL.md",
    REPO / ".gemini" / "antigravity-cli" / "skills" / "j-next" / "SKILL.md",
)

J_BRAINSTORM_COMMANDS = (
    REPO / ".claude" / "commands" / "j-brainstorm.md",
    REPO / ".agents" / "skills" / "cmd-j-brainstorm" / "SKILL.md",
    REPO / ".gemini" / "antigravity-cli" / "skills" / "j-brainstorm" / "SKILL.md",
)

PROSE_TELL_CONSUMERS = (
    REPO / ".claude" / "commands" / "j-paper.md",
    REPO / ".agents" / "skills" / "cmd-j-paper" / "SKILL.md",
    REPO / ".gemini" / "antigravity-cli" / "skills" / "j-paper" / "SKILL.md",
    REPO / ".claude" / "agents" / "documentation-writer.md",
    REPO / ".codex" / "agents" / "documentation-writer.toml",
    REPO / ".gemini" / "config" / "agents" / "documentation-writer.md",
)

WRITING_PLANS_SKILLS = tuple(p for p in PLAN_WORKFLOW_SKILLS if p.parent.name == "writing-plans")

CODE_REVIEW_PR_SIZE_SURFACES = (
    *(root / "code-review-patterns" / "SKILL.md" for root in SKILL_TREE_ROOTS),
    REPO / ".agents" / "skills" / "cmd-j-diff-review" / "SKILL.md",
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

# A rebase's risk is its conflict resolutions; every copy scopes verification to them.
REBASE_COMMANDS = (
    REPO / ".claude" / "commands" / "j-rebase.md",
    REPO / ".agents" / "skills" / "cmd-j-rebase" / "SKILL.md",
    REPO / ".gemini" / "antigravity-cli" / "skills" / "j-rebase" / "SKILL.md",
)

# Two trees, one table of contents. Bodies may differ -- `.claude/` is written for a generation
# `.agents/` does not serve -- but a heading in one tree and not the other means a reader of the
# other tree cannot find the topic at all. See docs/adr/workflow/reference-tree-section-parity.md.
REFERENCE_TREES = (REPO / ".claude" / "references", REPO / ".agents" / "references")


def tracked_tree_files() -> list[str]:
    """Tracked paths under the four KB trees. A failed `git ls-files` raises with git's own stderr,
    which `check=True` alone would drop from the test output."""
    result = subprocess.run(
        ["git", "ls-files", "-z", *TREE_ROOTS],
        cwd=REPO,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"git ls-files failed ({result.returncode}): {result.stderr.strip()}")
    return [name for name in result.stdout.split("\0") if name]


def discipline_body(text: str) -> list[str]:
    """The prose under the existing-code-discipline sections, normalized for cross-copy compare.

    Heading level is dropped -- the command skill nests the same sections one level deeper -- as are
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


# The reference twins and the Codex command skill that embeds their text.
EXISTING_CODE_DISCIPLINE_OWNERS = (
    ".claude/references/workflow/existing-code-discipline.md",
    ".agents/references/workflow/existing-code-discipline.md",
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


FRONTMATTER_BLOCK = re.compile(r"\A---\n(.*?)\n---\n", re.DOTALL)


def frontmatter_block(path: Path) -> str:
    """Return a file's YAML frontmatter text, failing with the file named when it has none."""
    match = FRONTMATTER_BLOCK.match(path.read_text(encoding="utf-8"))
    if match is None:
        raise AssertionError(f"{path.relative_to(REPO)} has no frontmatter block")
    return match.group(1)


def frontmatter_key_lines(path: Path) -> list[str]:
    """Return the key of every unindented, non-comment frontmatter line, duplicates kept.

    A line scan, not a YAML parser: quotes are stripped from a key, and an unindented line with no
    colon (a `? key` complex key) is returned whole, so either form counts as a key.
    """
    return [
        line.split(":", 1)[0].strip().strip("\"'")
        for line in frontmatter_block(path).splitlines()
        if line and not line[0].isspace() and not line.startswith("#")
    ]


def frontmatter_keys(path: Path) -> set[str]:
    """Return the set of top-level frontmatter keys (see `frontmatter_key_lines`)."""
    return set(frontmatter_key_lines(path))


def skill_directories(root: Path) -> set[str]:
    return {path.parent.name for path in root.glob("*/SKILL.md")}


def description(path: Path) -> str:
    match = FRONTMATTER_DESCRIPTION.search(path.read_text())
    if match is None:
        raise AssertionError(f"missing description: {path}")
    return match.group(1)


# Consult commands list their agent's reference index beside optional delegation
# (docs/adr/workflow/consult-commands-offer-references-and-delegation.md).
CONSULT_AGENTS = {
    "j-arch": ("architecture-specialist",),
    "j-cloud": ("cloud-architect",),
    "j-devops": ("devops-engineer",),
    "j-docs": ("documentation-writer",),
    "j-frontend": ("frontend-engineer",),
    "j-lang": ("language-specialist",),
    "j-data": ("data-engineer",),
    "j-system-design": (
        "architecture-specialist",
        "data-engineer",
        "security-reviewer",
        "devops-engineer",
    ),
}
# Agents whose `Reference library at` block lists only part of its directory, and why.
CURATED_REFERENCE_INDEXES: dict[str, str] = {
    "workflow-architect": (
        "single-purpose: works from workflow-specification; "
        "architecture-specialist indexes the directory"
    ),
    "scope-reviewer": (
        "single-purpose: applies scope-review-methodology; product-manager indexes the directory"
    ),
    "code-reviewer": "workflow/ is a mixed directory; the reviewer indexes the review files",
}
AGENT_INDEX_TREES = ((".claude", ".claude"), (".gemini/config", ".agents"))
LIBRARY_BLOCK = re.compile(r"^Reference library at `?([^`\n]+?)`?:$")


def library_gaps(agent_text: str, references: Path) -> dict[str, set[str]]:
    """Return, per `Reference library at` directory, the files the block does not name."""
    gaps: dict[str, set[str]] = {}
    lines = agent_text.splitlines()
    for index, line in enumerate(lines):
        match = LIBRARY_BLOCK.match(line.strip())
        if match is None:
            continue
        directory = match.group(1).rstrip("/").rsplit("/", 1)[-1]
        named = reference_names("\n".join([line.strip(), *lines[index + 1 :]]))
        listed = {name.split("/", 1)[1] for name in named if name.startswith(f"{directory}/")}
        gaps[directory] = {p.stem for p in (references / directory).glob("*.md")} - listed
    return gaps


REFERENCE_BLOCK = re.compile(r"^(?:Reference library at|Also see) `?([^`\n]+?)`?:$")


def reference_blocks(text: str) -> list[str]:
    """Return the path of each `Reference library at` or `Also see` block."""
    matches = (REFERENCE_BLOCK.match(line.strip()) for line in text.splitlines())
    return [match.group(1) for match in matches if match]


def reference_names(text: str) -> set[str]:
    """Return "<dir>/<name>" for each name under a `Reference library at` or `Also see` block."""
    names: set[str] = set()
    lines = text.splitlines()
    for index, line in enumerate(lines):
        match = REFERENCE_BLOCK.match(line.strip())
        if match is None:
            continue
        directory = match.group(1).rstrip("/").rsplit("/", 1)[-1]
        for item in lines[index + 1 :]:
            if not item.startswith("- "):
                break
            names.update(
                f"{directory}/{name.strip()}" for name in item[2:].split(",") if name.strip()
            )
    return names


# The prefix each (tree, kind) cites shared references by, as written in the repo. Sync expands the
# home-anchored `~/` forms to the absolute home path in installed copies. No form resolves against
# the project (docs/adr/workflow/kb-reference-paths-never-resolve-against-the-project.md).
AGENTS_REFERENCES = "~/.agents/references/"
CLAUDE_REFERENCES = "~/.claude/references/"
REFERENCE_FORMS: dict[tuple[str, str], str] = {
    (".agents", "skills"): AGENTS_REFERENCES,
    (".gemini/antigravity-cli", "skills"): AGENTS_REFERENCES,
    (".codex", "agents"): AGENTS_REFERENCES,
    (".gemini/config", "agents"): AGENTS_REFERENCES,
    (".claude", "skills"): "${CLAUDE_SKILL_DIR}/../../references/",
    (".claude", "commands"): CLAUDE_REFERENCES,
    (".claude", "agents"): CLAUDE_REFERENCES,
}
REFERENCE_CATEGORIES = sorted(
    {
        path.name
        for tree in (".agents", ".claude")
        for path in (REPO / tree / "references").iterdir()
        if path.is_dir()
    }
)
# One citation token: everything from the last delimiter up to and including `<category>/`.
REFERENCE_CITATION = re.compile(
    r"[^\s`'\"(\[*<>=,]*references/(?P<category>" + "|".join(REFERENCE_CATEGORIES) + ")/"
)
# A reference directory named with no category: a `Glob(`, `Scan`, or `Reference library at` read
# instruction. The token runs back to the last delimiter, so `~/.claude/references/` is one token.
BARE_REFERENCE_DIR = re.compile(r"[^\s`'\"(\[*<>=,]*(?:\.claude|\.agents)/references/")
# These describe the repo layout or name project-scope surfaces on purpose, so they keep the
# relative form. Exact tracked paths, never a suffix match.
RELATIVE_REFERENCE_EXEMPT = (
    ".agents/skills/config-security-audit/SKILL.md",
    ".agents/skills/skill-audit/SKILL.md",
    ".claude/skills/config-security-audit/SKILL.md",
    ".claude/skills/skill-audit/scripts/audit.py",
    ".gemini/antigravity-cli/skills/config-security-audit/SKILL.md",
    ".gemini/antigravity-cli/skills/skill-audit/SKILL.md",
)


def reference_form_key(rel: str) -> tuple[str, str] | None:
    """Return the REFERENCE_FORMS key a tracked path falls under, or None."""
    for tree, kind in REFERENCE_FORMS:
        if rel.startswith(f"{tree}/{kind}/"):
            return tree, kind
    return None


def project_relative_citations(text: str, prefix: str) -> list[str]:
    """Return each reference citation in `text` that is not written with `prefix`."""
    return [
        match.group(0)
        for match in REFERENCE_CITATION.finditer(text)
        if match.group(0) != f"{prefix}{match.group('category')}/"
    ]


def bare_prefixes(key: tuple[str, str]) -> tuple[str, ...]:
    """The prefixes a bare reference directory may carry under one REFERENCE_FORMS key."""
    if key == (".claude", "skills"):
        return (CLAUDE_REFERENCES,)
    return (REFERENCE_FORMS[key],)


def bare_reference_dirs(text: str, prefixes: tuple[str, ...]) -> list[str]:
    """Return each `.claude/references/` or `.agents/references/` token not written as a prefix."""
    return [m.group(0) for m in BARE_REFERENCE_DIR.finditer(text) if m.group(0) not in prefixes]


def command_copies(command: str) -> tuple[tuple[Path, str], ...]:
    """Return each tree's copy of a command with the reference path prefix that tree uses."""
    agents = REFERENCE_FORMS[(".agents", "skills")]
    return (
        (
            REPO / ".claude" / "commands" / f"{command}.md",
            REFERENCE_FORMS[(".claude", "commands")],
        ),
        (REPO / ".agents" / "skills" / f"cmd-{command}" / "SKILL.md", agents),
        (
            REPO / ".gemini" / "antigravity-cli" / "skills" / command / "SKILL.md",
            REFERENCE_FORMS[(".gemini/antigravity-cli", "skills")],
        ),
    )


def agent_reference_names(agent: str) -> set[str]:
    path = REPO / ".claude" / "agents" / f"{agent}.md"
    return reference_names(path.read_text(encoding="utf-8"))


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
        claude = {path.stem for path in (REPO / ".claude" / "commands").glob("j-*.md")}
        antigravity = {
            name
            for name in skill_directories(REPO / ".gemini" / "antigravity-cli" / "skills")
            if name.startswith("j-")
        }

        self.assertFalse(list((REPO / ".codex" / "prompts").glob("*.md")), "use command skills")
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

    def test_claude_skill_descriptions_are_one_quoted_line_of_200_chars(self):
        """Owner: .claude/skills/writing-skills/SKILL.md (Description = triggering conditions only)."""
        checked = 0
        for path in sorted((REPO / ".claude" / "skills").glob("*/SKILL.md")):
            line = next(
                (x for x in frontmatter_block(path).splitlines() if x.startswith("description:")),
                "",
            )
            value = line[len("description:") :].strip()
            checked += 1
            with self.subTest(skill=path.parent.name):
                self.assertRegex(value, r'^"[^"]*"$', f"{path.parent.name}: one quoted line")
                self.assertLessEqual(
                    len(value) - 2, 200, f"{path.parent.name}: over 200 characters"
                )
        self.assertGreater(checked, 0, "no Claude skills checked; the check is vacuous")

    def test_shared_skill_frontmatter_is_name_and_description_only(self):
        """Shared skills carry only name and description (owner: .agents/skills/writing-skills/SKILL.md)."""
        checked = 0
        for root in SHARED_SKILL_ROOTS:
            for path in sorted(root.glob("*/SKILL.md")):
                rel = path.relative_to(REPO)
                checked += 1
                with self.subTest(skill=str(rel)):
                    lines = frontmatter_key_lines(path)
                    self.assertEqual(
                        sorted(lines),
                        sorted(SHARED_SKILL_FRONTMATTER_KEYS),
                        f"{rel}: shared skill frontmatter must be exactly name and description, found {sorted(lines)}",
                    )
        self.assertGreater(checked, 0, "no shared skills checked; the check is vacuous")

    def test_no_skill_declares_skills_frontmatter(self):
        """No harness reads `skills:` in a skill's own frontmatter; name dependencies in the body."""
        checked = 0
        for root in SKILL_TREE_ROOTS:
            for path in sorted(root.glob("*/SKILL.md")):
                rel = path.relative_to(REPO)
                checked += 1
                with self.subTest(skill=str(rel)):
                    self.assertNotIn(
                        "skills",
                        frontmatter_keys(path),
                        f"{rel}: drop the inert skills: key and name the dependency in the body",
                    )
        self.assertGreater(checked, 0, "no skills checked; the check is vacuous")

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

    def test_delegation_ladders_name_no_unapproved_pinned_model(self):
        """A ladder rung that names a model version goes stale every generation, so rungs describe
        tiers by role. Scoped to bullets on purpose: GEMINI.md's prose names a slug as a worked
        `--model` example and ships `agy models` beside it as the freshness pointer, which is the
        staleness problem already solved rather than an instance of it. The regex needs a digit, so
        floating aliases (opus, sonnet, haiku, fable, flash, pro) stay legal. The user's exact
        Codex selections are the only versioned-model exception."""
        for path in ALWAYS_LOADED_CEILINGS:
            lines = path.read_text(encoding="utf-8").splitlines()
            if path == REPO / ".codex" / "AGENTS.md":
                lines = [CODEX_SELECTED_MODELS.sub("", line) for line in lines]
            pinned = [
                line
                for line in lines
                if line.lstrip().startswith(("-", "*")) and PINNED_MODEL.search(line)
            ]
            with self.subTest(path=path.relative_to(REPO)):
                self.assertEqual(pinned, [], f"{path.name} pins an unapproved model version")

    def test_codex_model_allowlist_matches_only_exact_selected_names(self):
        for name in ("gpt-6.2-astra", "gpt-6.2-sol", "gpt-6.1-terra", "gpt-6-luna"):
            with self.subTest(name=name):
                self.assertEqual(CODEX_SELECTED_MODELS.sub("", f"`{name}`"), "``")
        for name in (
            "gpt-6-sol",
            "gpt-6.1-sol",
            "gpt-6-astra",
            "gpt-5.6-sol",
            "gpt-6.2-sol-2026-09-28",
            "gpt-6.2-astra-pro",
        ):
            with self.subTest(name=name):
                self.assertIsNotNone(PINNED_MODEL.search(CODEX_SELECTED_MODELS.sub("", name)))

    def test_codex_delegation_guidance_pairs_models_with_effort(self):
        guidance = (REPO / ".codex" / "AGENTS.md").read_text()
        for model, effort in (
            ("gpt-6.2-sol", "medium"),
            ("gpt-6.2-astra", "high"),
            ("gpt-6.1-terra", "medium"),
            ("gpt-6-luna", "low"),
        ):
            with self.subTest(model=model):
                self.assertRegex(guidance, rf"`{re.escape(model)}`[^\n]*?`{effort}`")

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

    def test_diff_review_fixes_all_but_large_unrelated_findings(self):
        for path in DIFF_REVIEW_DISPATCHERS:
            body = path.read_text(encoding="utf-8")
            with self.subTest(path=path.relative_to(REPO)):
                self.assertIn("large and unrelated", body)
                self.assertIn("bug and task tracking mechanism", body)
                self.assertIn("security findings included", body)
                self.assertIn("another PR", body)
                self.assertNotIn("future-work mechanism", body)
                self.assertNotIn("Never create a new plan file", body)
                self.assertNotIn("public tracker", body)

    def test_diff_review_diffs_against_origin_main(self):
        # Plan branches are cut from origin/main; a stale local main drags landed commits into review.
        for path in DIFF_REVIEW_DISPATCHERS:
            body = path.read_text(encoding="utf-8")
            with self.subTest(path=path.relative_to(REPO)):
                self.assertIn("git diff origin/main...HEAD", body)
                self.assertIsNone(re.search(r"(?<!origin/)\bmain\.\.\.?HEAD", body))

    def test_diff_review_reviewers_start_from_a_clean_context(self):
        # The orchestrating session is often the one that wrote the code; its history and intent stay out.
        for path in DIFF_REVIEW_DISPATCHERS:
            body = path.read_text(encoding="utf-8")
            with self.subTest(path=path.relative_to(REPO)):
                self.assertIn("fresh context", body)
                self.assertIn("a reviewer handed the author's intent tends to confirm it", body)
                self.assertIn("Run `git diff origin/main...<HEAD SHA>` yourself", body)
                self.assertNotIn("Read every changed file in full", body)

    def test_plan_boundaries_close_with_the_diff_review(self):
        """One close per boundary: the diff review runs, then its Step 7 opens the PR."""
        for path in (
            *EXECUTION_CLOSE_OWNERS,
            *PLAN_EXECUTION_CONSUMERS,
            *CLAUDE_PLAN_EXECUTION_COMMANDS,
        ):
            body = path.read_text(encoding="utf-8")
            with self.subTest(path=path.relative_to(REPO)):
                self.assertIn("j-diff-review", body)
                self.assertNotIn("Load `finishing-branch` skill and follow it", body)
                self.assertNotIn("Final Whole-Branch Review", body)
        for path in EXECUTION_CLOSE_OWNERS:
            if path.parent.name == "executing-plans":
                with self.subTest(path=path.relative_to(REPO)):
                    self.assertIn("follow it through Step 7", path.read_text(encoding="utf-8"))
        for path in J_NEXT_COMMANDS:
            with self.subTest(path=path.relative_to(REPO)):
                self.assertNotIn("create-pr", path.read_text(encoding="utf-8"))
        self.assertEqual(len(WRITING_PLANS_SKILLS), 3)
        for path in WRITING_PLANS_SKILLS:
            body = path.read_text(encoding="utf-8")
            with self.subTest(path=path.relative_to(REPO)):
                self.assertIn("bug and task tracking mechanism", body)
                self.assertNotIn("public tracker", body)

    def test_between_pr_cleanup_deletes_only_merged_branches(self):
        blocks = []
        for path in (*J_NEXT_COMMANDS, *WRITING_PLANS_SKILLS):
            lines = path.read_text(encoding="utf-8").splitlines()
            with self.subTest(path=path.relative_to(REPO)):
                self.assertNotIn("branch --merged", "\n".join(lines))
                start = next(
                    (i for i, ln in enumerate(lines) if ln.startswith("git for-each-ref")), -1
                )
                self.assertNotEqual(start, -1, "'git for-each-ref' not found")
                end = lines.index("done", start)
                blocks.append("\n".join(lines[start : end + 1]))
        self.assertEqual(len(blocks), len(J_NEXT_COMMANDS) + len(WRITING_PLANS_SKILLS))
        self.assertEqual(len(set(blocks)), 1)
        env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
        env.update(
            {
                "GIT_CONFIG_GLOBAL": "/dev/null",
                "GIT_CONFIG_NOSYSTEM": "1",
                "GIT_AUTHOR_NAME": "t",
                "GIT_AUTHOR_EMAIL": "t@example.com",
                "GIT_COMMITTER_NAME": "t",
                "GIT_COMMITTER_EMAIL": "t@example.com",
            }
        )

        def run(argv, cwd):
            try:
                return subprocess.run(
                    argv,
                    cwd=cwd,
                    env=env,
                    check=True,
                    capture_output=True,
                    text=True,
                    stdin=subprocess.DEVNULL,
                ).stdout
            except subprocess.CalledProcessError as exc:
                self.fail(
                    f"{argv} failed ({exc.returncode})\nstdout: {exc.stdout}\nstderr: {exc.stderr}"
                )

        def git(*args, cwd):
            return run(["git", *args], cwd)

        def commit_file(tmp, name, label):
            Path(tmp, name).write_text(label, encoding="utf-8")
            git("add", ".", cwd=tmp)
            git("commit", "-q", "-m", label, cwd=tmp)

        with tempfile.TemporaryDirectory() as tmp:
            git("init", "-q", "-b", "main", cwd=tmp)
            git("commit", "-q", "--allow-empty", "-m", "base", cwd=tmp)
            for name in ("feat/ff", "feat/squash", "feat/open"):
                git("checkout", "-q", "-b", name, "main", cwd=tmp)
                slug = name.replace("/", "_")
                for i in (1, 2) if name == "feat/squash" else (1,):
                    commit_file(tmp, f"{slug}{i}.txt", f"{name} {i}")
            git("checkout", "-q", "-b", "feat/squash-plus", "feat/squash", cwd=tmp)
            commit_file(tmp, "extra.txt", "extra unmerged")
            git("checkout", "-q", "main", cwd=tmp)
            git("merge", "-q", "--ff-only", "feat/ff", cwd=tmp)
            git("merge", "-q", "--squash", "feat/squash", cwd=tmp)
            git("commit", "-q", "-m", "squash", cwd=tmp)
            git("update-ref", "refs/remotes/origin/main", "HEAD", cwd=tmp)
            out = run(["bash", "-c", f"set -eo pipefail\n{blocks[0]}"], tmp)
            self.assertIn("kept: feat/open", out)
            left = git("for-each-ref", "--format=%(refname:short)", "refs/heads/", cwd=tmp).split()
        self.assertEqual(set(left), {"main", "feat/open", "feat/squash-plus"})

        with tempfile.TemporaryDirectory() as tmp:
            git("init", "-q", "-b", "main", cwd=tmp)
            git("commit", "-q", "--allow-empty", "-m", "base", cwd=tmp)
            git("update-ref", "refs/remotes/origin/main", "HEAD", cwd=tmp)
            run(["bash", "-c", f"set -eo pipefail\n{blocks[0]}"], tmp)

    def test_j_next_edits_the_plan_after_the_new_branch_exists(self):
        for path in J_NEXT_COMMANDS:
            text = path.read_text(encoding="utf-8")
            with self.subTest(path=path.relative_to(REPO)):
                self.assertGreater(text.index("collapse its phases"), text.index("git checkout -b"))

    def test_create_pr_branch_case_grades_the_convention_not_the_prompt(self):
        case = (
            REPO / "evals" / "cases" / "effects" / "create-pr--effect-derive-branch-and-skip-debris"
        )
        prompt = (case / "prompt.md").read_text(encoding="utf-8")
        grader = (case / "graders" / "branch.md").read_text(encoding="utf-8")
        self.assertNotIn("any prefix is fine", prompt)
        self.assertNotIn("kebab-case", prompt)
        self.assertIn("[a-z]+/[a-z0-9]", grader)

    def test_create_pr_agents_use_the_branch_convention_and_remote_base(self):
        agents = (
            REPO / ".claude" / "agents" / "create-pr.md",
            REPO / ".gemini" / "config" / "agents" / "create-pr.md",
            REPO / ".codex" / "agents" / "create-pr.toml",
        )
        for path in agents:
            text = path.read_text(encoding="utf-8")
            with self.subTest(path=path.relative_to(REPO)):
                self.assertIn("type/short-description", text)
                self.assertIn("origin/main..", text)
                self.assertIn("git fetch origin main", text)
                self.assertIn(
                    "`type` is `feat`, `fix`, `docs`, `chore`, `refactor`, or `test`", text
                )
                self.assertNotIn("<username>", text)
                self.assertIsNone(re.search(r"(?<![\w/])main\.\.", text))

    def test_plans_size_prs_for_review(self):
        self.assertEqual(len(WRITING_PLANS_SKILLS), 3)
        for path in WRITING_PLANS_SKILLS:
            body = path.read_text(encoding="utf-8")
            with self.subTest(path=path.relative_to(REPO)):
                self.assertIn("3-5k changed lines", body)
                self.assertIn("not a hard requirement", body)
                self.assertNotIn("Default to one PR per plan", body)
        for path in (*J_PLAN_COMMANDS, CLAUDE_J_PLAN_COMMAND):
            with self.subTest(path=path.relative_to(REPO)):
                self.assertIn("size target", path.read_text(encoding="utf-8"))
        for path in CODE_REVIEW_PR_SIZE_SURFACES:
            body = path.read_text(encoding="utf-8")
            with self.subTest(path=path.relative_to(REPO)):
                self.assertIn("3-5k", body)
        for path in (*CODE_REVIEW_PR_SIZE_SURFACES, *DIFF_REVIEW_DISPATCHERS):
            with self.subTest(path=path.relative_to(REPO)):
                self.assertNotRegex(path.read_text(encoding="utf-8"), r"\b400 lines")

    def test_plan_surfaces_state_the_bare_minimum_rule(self):
        # Scope, never quality: each statement cites the axis boundary the ladder ADR records.
        execute = [
            p
            for p in (*PLAN_EXECUTION_CONSUMERS, *CLAUDE_PLAN_EXECUTION_COMMANDS)
            if "execute-plan" in str(p)
        ]
        self.assertEqual(len(execute), 3)
        for path in (*PLAN_WORKFLOW_SKILLS, *J_PLAN_COMMANDS, CLAUDE_J_PLAN_COMMAND, *execute):
            body = path.read_text(encoding="utf-8")
            with self.subTest(path=path.relative_to(REPO)):
                self.assertIn("bare minimum", body)
                self.assertIn("Where Thoroughness Stops", body)

    def test_planning_reads_project_conventions(self):
        plan_phrases = (
            "**Project conventions**",
            "never filtered by spec keywords",
            "Flag any design input that conflicts with a listed convention; cite its ID.",
            "without an ADR-backed exception",
            "conformance to the conventions digest",
        )
        for path in (*J_PLAN_COMMANDS, CLAUDE_J_PLAN_COMMAND):
            body = path.read_text(encoding="utf-8")
            for phrase in plan_phrases:
                with self.subTest(path=path.relative_to(REPO), phrase=phrase):
                    self.assertIn(phrase, body)
        self.assertEqual(len(WRITING_PLANS_SKILLS), 3)
        for path in (*J_BRAINSTORM_COMMANDS, *WRITING_PLANS_SKILLS):
            with self.subTest(path=path.relative_to(REPO)):
                self.assertIn(
                    "conformance to the conventions digest", path.read_text(encoding="utf-8")
                )

    def test_writing_surfaces_apply_the_ai_writing_tells_reference(self):
        for tree in (".claude", ".agents"):
            with self.subTest(tree=tree):
                self.assertTrue(
                    (REPO / tree / "references" / "documentation" / "ai-writing-tells.md").is_file()
                )
        for path in PROSE_TELL_CONSUMERS:
            with self.subTest(path=path.relative_to(REPO)):
                body = path.read_text(encoding="utf-8")
                self.assertIn("ai-writing-tells", body)
                self.assertIn("Revision Pass", body)

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
                    # The command needs a resolvable file for each language.
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
        skill owns mindset, severity labels, and feedback. The Codex command skill owns its disposition ladder."""
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
        command = REPO / ".agents" / "skills" / "cmd-j-diff-review" / "SKILL.md"
        self.assertIn("Disposition Ladder", command.read_text(encoding="utf-8"))

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

    def test_rebase_verification_is_scoped_to_conflicted_files(self):
        for path in REBASE_COMMANDS:
            text = path.read_text(encoding="utf-8")
            with self.subTest(path=path.relative_to(REPO)):
                capture = "git diff --name-only --diff-filter=U"
                self.assertIn(capture, text)
                # `git add` clears the unmerged list, so the capture has to come first.
                self.assertLess(text.index(capture), text.index("git add <files>"))
                self.assertIn("The test set is `CONFLICTED` plus `DRIFT`", text)
                # After the rebase, `origin/main...HEAD` shows only the branch's side.
                self.assertIn('git diff "$BASE" origin/main', text)
                self.assertIn("whether the full suite ran and why", text)
                self.assertNotIn("git diff --name-only origin/main..HEAD", text)
                self.assertNotIn("the merged file set", text)

    def test_shared_skills_carry_no_upstream_superpowers_paths(self):
        """`.agents/` was seeded from obra/superpowers; its paths and cross-reference
        syntax are ours now. REFERENCES.md keeps the upstream attribution."""
        for path in sorted((REPO / ".agents" / "skills").glob("*/SKILL.md")):
            leftovers = [
                line.strip() for line in path.read_text().splitlines() if "superpowers" in line
            ]
            with self.subTest(skill=path.parent.name):
                self.assertEqual(leftovers, [], f"{path.parent.name}: upstream leftover")

    def test_no_hook_restates_always_loaded_guidance(self):
        settings = json.loads((REPO / ".claude" / "settings.json").read_text())
        self.assertFalse(settings.get("hooks", {}))
        codex = (REPO / ".codex" / "config.toml").read_text()
        for event in ("PreToolUse", "SessionStart"):
            self.assertNotIn(event, codex)
        for tree in (".claude", ".codex"):
            self.assertFalse((REPO / tree / "hooks" / "guard-bash-output.sh").exists())

    def test_prompt_loggers_are_retired(self):
        for root, config in (
            (".claude", "settings.json"),
            (".codex", "config.toml"),
            (".gemini", "antigravity-cli/settings.json"),
        ):
            with self.subTest(root=root):
                self.assertNotIn("log-prompt.sh", (REPO / root / config).read_text())
                self.assertFalse((REPO / root / "hooks" / "log-prompt.sh").exists())

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
                self.assertIn("owned by the current user", content)
                self.assertIn("confirm a discovered plan path", content)
                # Codex and Gemini have no Claude plan directory to fall back on.
                self.assertNotIn("~/.claude/plans/", content)

    def test_plan_execution_runs_to_the_pr_without_pausing(self):
        # A mid-PR "ready for feedback" stop asks for review the PR already provides.
        for path in PLAN_WORKFLOW_SKILLS:
            content = path.read_text()
            with self.subTest(path=path.relative_to(REPO)):
                for phrase in MID_PR_PAUSE_PHRASES:
                    self.assertNotIn(phrase, content)
        for path in (*PLAN_EXECUTION_CONSUMERS, *CLAUDE_PLAN_EXECUTION_COMMANDS):
            with self.subTest(path=path.relative_to(REPO)):
                self.assertIn("blocking question", path.read_text())

    def test_plan_execution_mode_is_chosen_not_asked(self):
        # The executor picks inline or subagent mode from the plan; one section owns the rule.
        for path in EXECUTION_MODE_OWNERS:
            content = path.read_text()
            with self.subTest(path=path.relative_to(REPO)):
                self.assertIn("## Execution Mode\n", content)
                self.assertIn("without asking", content)
                self.assertIn("Decision Log", content)
        for path in (*PLAN_EXECUTION_CONSUMERS, *CLAUDE_PLAN_EXECUTION_COMMANDS):
            content = path.read_text()
            with self.subTest(path=path.relative_to(REPO)):
                self.assertRegex(content, r"`subagent-driven-development`,\s+Execution Mode")
                self.assertIn("without asking", content)
        for path in (*PLAN_WORKFLOW_SKILLS, *J_PLAN_COMMANDS, CLAUDE_J_PLAN_COMMAND):
            content = path.read_text()
            with self.subTest(path=path.relative_to(REPO)):
                for phrase in MODE_MENU_PHRASES:
                    self.assertNotIn(phrase, content)

    def test_codex_surfaces_name_codex_command_syntax(self):
        # Codex invokes commands as `$cmd-j-*`; a bare `/j-*` names nothing it can run.
        for path in (
            REPO / ".codex" / "AGENTS.md",
            REPO / ".agents" / "skills" / "writing-plans" / "SKILL.md",
        ):
            with self.subTest(path=path.relative_to(REPO)):
                self.assertNotRegex(
                    path.read_text(), r"`/j-execute-plan|`/j-next|Run /j-execute-plan"
                )

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

    def test_kb_assets_carry_no_source_credit(self):
        """Credit lives in REFERENCES.md, one line per source (owner: writing-skills, Claims and
        evidence). Tracked `.md` and `.toml` files under the four trees only."""
        tracked = tracked_tree_files()
        scanned = 0
        credits = []
        for name in tracked:
            path = REPO / name
            if path.suffix not in {".md", ".toml"} or not path.is_file():
                continue
            scanned += 1
            for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                if any(pattern.search(line) for pattern in SOURCE_CREDIT_PATTERNS):
                    credits.append(
                        f"{name}:{lineno}: source credit belongs in REFERENCES.md: {line.strip()}"
                    )
        self.assertGreater(scanned, 0, "no KB files scanned; the check is vacuous")
        self.assertEqual(credits, [], "\n".join(credits))

    def test_kb_assets_contain_no_em_dash(self):
        """KB assets use a comma, semicolon, colon, or parentheses instead (rule: CLAUDE.md,
        Editing Guidelines). Tracked files only; a file that does not decode as UTF-8 is reported
        rather than skipped silently."""
        self.assertIn(EM_DASH, "a \u2014 b", "EM_DASH is not the em dash; the check is vacuous")
        self.assertNotIn(EM_DASH, "a \u2013 b - c", "EM_DASH matches an en dash or a hyphen")
        tracked = tracked_tree_files()
        scanned = 0
        dashes = []
        undecodable = []
        for name in tracked:
            path = REPO / name
            if not path.is_file():
                continue
            try:
                text = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                undecodable.append(name)
                continue
            scanned += 1
            for lineno, line in enumerate(text.splitlines(), 1):
                if EM_DASH in line:
                    dashes.append(
                        f"{name}:{lineno}: em dash; use a comma, semicolon, colon, or parentheses"
                    )
        self.assertGreater(scanned, 0, "no KB files scanned; the check is vacuous")
        self.assertEqual(undecodable, [], "tracked KB files that are not UTF-8 text")
        self.assertEqual(dashes, [], "\n".join(dashes))

    def test_kb_prose_contains_no_double_hyphen_dash(self):
        """A prose ` -- ` is a dash typed as two hyphens, padded or wrapped at a line end; use a
        comma, semicolon, colon, or parentheses. Code spans, fenced blocks, and table placeholder
        cells are exempt, so a CLI argument separator stays in a code span. Frontmatter
        descriptions and TOML strings are prose. Scope: tracked `.md` and `.toml` files under the
        four trees only; shell and Python comments are out of scope."""

        def flagged(text):
            hits = []
            for lineno, line in lines_outside_fences(text):
                line = CODE_SPAN.sub("C", line)
                line = TABLE_PLACEHOLDER.sub("", line)
                if PROSE_DOUBLE_HYPHEN.search(line):
                    hits.append(lineno)
            return hits

        for bad in (
            "A prose dash -- here",
            "## Phase 1 -- Scope",
            "run `git log -- path` -- then",
            "text --",
            "a  --  b",
        ):
            self.assertTrue(flagged(bad), f"control not rejected: {bad!r}")
        for good in (
            "Use `git diff -- path` here",
            "| Right | -- | x |",
            "```\na -- b\n```",
            "a --flag",
            "-- list item",
        ):
            self.assertFalse(flagged(good), f"control wrongly rejected: {good!r}")
        tracked = tracked_tree_files()
        scanned = 0
        problems = []
        for name in tracked:
            if not name.endswith((".md", ".toml")):
                continue
            path = REPO / name
            if not path.is_file():
                continue
            text = path.read_text(encoding="utf-8")
            scanned += 1
            if has_unterminated_fence(text):
                problems.append(f"{name}: unterminated fence hides prose from the check")
            problems.extend(
                f"{name}:{n}: prose ' -- '; use a comma, colon, semicolon, or parentheses"
                for n in flagged(text)
            )
        self.assertGreater(scanned, 0, "no KB files scanned; the check is vacuous")
        self.assertEqual(problems, [], "\n".join(problems))

    def test_source_credit_patterns_catch_each_credit_form(self):
        """Positive and negative controls, so a loosened pattern cannot pass on a clean tree."""
        credits = (
            "Source: Google Python Style Guide. Only rules linters cannot enforce.",
            "- Source: a style guide",
            "**Source:** a style guide",
            "Adapted from [x](https://example.com), the origin of the form.",
            "### Skill Categories (from Anthropic's Guide)",
            "Meincke et al. (2025) tested 7 principles.",
            "*Analysis generated using the three-pass method (Keshav, 2016).*",
            "**API specifics** (per [the caching docs](https://example.com/docs)).",
        )
        plain = (
            "First Author et al. (2026)",
            '`\\citep` for "(Author, 2024)"',
            "**Source email:** {email subject line}",
            "Source Confidence: [0.0-1.0 per source]",
            "The source of truth is the plan file.",
            "Sources only in REFERENCES.md.",
            "XFetch: recompute early.",
        )
        for line in credits:
            with self.subTest(credit=line):
                self.assertTrue(any(p.search(line) for p in SOURCE_CREDIT_PATTERNS))
        for line in plain:
            with self.subTest(plain=line):
                self.assertFalse(any(p.search(line) for p in SOURCE_CREDIT_PATTERNS))

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
        tracked = tracked_tree_files()
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

        # `${CLAUDE_SKILL_DIR}` resolves to the skill's own folder, and Claude Code substitutes it
        # only in a Claude skill. A prefixed path must name a file that skill ships, and no other
        # file in the agent trees may carry the token, where it would reach the reader
        # unsubstituted.
        skill_dir_claims = 0
        for skill in sorted((REPO / ".claude" / "skills").glob("*/SKILL.md")):
            for claimed in SKILL_DIR_SCRIPT_PATH.findall(skill.read_text(encoding="utf-8")):
                skill_dir_claims += 1
                if not (skill.parent / claimed).is_file():
                    missing.append(f"{skill.relative_to(REPO)}: ${{CLAUDE_SKILL_DIR}}/{claimed}")
        self.assertGreater(skill_dir_claims, 0, "no ${CLAUDE_SKILL_DIR} script path found")
        self.assertEqual(
            missing, [], "skills name scripts they do not ship:\n" + "\n".join(missing)
        )
        stray = [
            name
            for name in tracked
            if name
            and not (name.startswith(".claude/skills/") and name.endswith("/SKILL.md"))
            and (REPO / name).is_file()
            and "${CLAUDE_SKILL_DIR}" in (REPO / name).read_text(encoding="utf-8", errors="ignore")
        ]
        self.assertEqual(stray, [], "${CLAUDE_SKILL_DIR} outside a Claude SKILL.md")

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

    def test_codex_parent_and_subagent_settings(self):
        config = (REPO / ".codex" / "config.toml").read_text()
        root = config.split("[agents]", 1)[0]
        for setting in (
            'model = "gpt-6.2-astra"',
            'model_reasoning_effort = "medium"',
            'plan_mode_reasoning_effort = "xhigh"',
        ):
            self.assertIn(setting, root.splitlines())
        for section, settings in (
            (
                "agents",
                (
                    'default_subagent_model = "gpt-6.2-sol"',
                    'default_subagent_reasoning_effort = "medium"',
                    "max_concurrent_threads_per_session = 6",
                ),
            ),
            (
                "features.multi_agent_v2",
                ("enabled = true", "expose_spawn_agent_model_overrides = true"),
            ),
        ):
            match = re.search(rf"(?ms)^\[{re.escape(section)}\]\n(.*?)(?=^\[|\Z)", config)
            self.assertIsNotNone(match, section)
            assert match is not None
            for setting in settings:
                self.assertIn(setting, match[1].splitlines())
        self.assertNotRegex(config, re.compile(r"^(?:max_threads|max_depth)\s*=", re.MULTILINE))

    def test_agent_guidance_matches_the_stated_spawn_depth(self):
        settings = json.loads((REPO / ".claude" / "settings.json").read_text())
        depth = settings["env"]["CLAUDE_CODE_MAX_SUBAGENT_SPAWN_DEPTH"]
        self.assertEqual(depth, "3")
        for config in ALWAYS_LOADED_CEILINGS:
            with self.subTest(path=config):
                self.assertEqual(
                    re.findall(r"Max spawn depth is (\d+)", config.read_text()), [depth]
                )

    def test_gemini_md_names_only_shipped_gemini_paths(self):
        """Setup deletes the legacy agents/ and commands/ dirs, so GEMINI.md must not cite them."""
        text = (REPO / ".gemini" / "GEMINI.md").read_text()
        for legacy in ("~/.gemini/agents/", "~/.gemini/commands/"):
            with self.subTest(path=legacy):
                self.assertNotIn(legacy, text)

    def test_codex_roles_inherit_spawn_model_and_effort(self):
        agents = sorted((REPO / ".codex" / "agents").glob("*.toml"))
        self.assertTrue(agents)
        for path in agents:
            with self.subTest(path=path):
                self.assertNotRegex(
                    path.read_text(),
                    re.compile(r"^\s*(?:model|model_reasoning_effort)\s*=", re.MULTILINE),
                )

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

        translated_claude_allow = set()
        for rule in claude_allow:
            if rule.startswith("Bash(") and rule.endswith(")"):
                cmd = rule[len("Bash(") : -1].removesuffix(" *")
                if cmd == "grep:*":
                    cmd = "grep"
                translated_claude_allow.add(f"command({cmd})")
        self.assertEqual(agy_allow, translated_claude_allow)

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

    def test_no_agent_config_allows_a_retired_entry(self):
        """No allow list or Codex allow rule may hold an entry that runs arbitrary code."""
        from macos_setup.dotfiles import RETIRED_PERMISSIONS

        for relative, retired in RETIRED_PERMISSIONS.items():
            allow = json.loads((REPO / relative).read_text())["permissions"]["allow"]
            for entry in retired:
                self.assertNotIn(entry, allow, f"{relative} still allows {entry}")

        # Every word of a retired command is banned from a Codex allow pattern, except the
        # families that keep audited subcommands (git, gh, npm) and `test` as in `npm test`
        # (a bare "test" never appears in an allow pattern; `just` and `ci` stay listed by hand).
        # `python`, `uv`, `make`, `ruff`, `rg`, `find` come from the Antigravity entries.
        words: set[str] = set()
        for entries in RETIRED_PERMISSIONS.values():
            for entry in entries:
                command = entry[entry.index("(") + 1 : -1].removesuffix(" *").removesuffix(":*")
                words.update(command.split())
        words -= {"git", "gh", "npm"}
        words.update(["just", "ci"])
        banned = [rf"[\"']{re.escape(t)}[\"']" for t in sorted(words)]
        rules = (REPO / ".codex" / "rules" / "default.rules").read_text()
        blocks = re.split(r"(?m)^(?=prefix_rule\()", rules)
        gh_patterns = []
        for block in blocks:
            if not block.startswith("prefix_rule(") or not re.search(
                r'decision\s*=\s*"allow"', block
            ):
                continue
            start = block.index("pattern")
            decision = re.search(r"decision\s*=", block)
            if decision is None:
                self.fail("allow rule has no decision")
            pattern = block[start : decision.start()]
            for token in banned:
                self.assertIsNone(
                    re.search(token, pattern), f"allow rule holds {token}: {pattern.strip()}"
                )
            for path in ("/private/", "/tmp/", ".venv/"):
                self.assertNotIn(path, pattern, f"allow rule names {path}: {pattern.strip()}")
            if re.search(r'pattern\s*=\s*\[\s*["\']gh["\']', pattern):
                gh_patterns.append(pattern)
        self.assertTrue(gh_patterns, "no gh allow rule found")
        for pattern in gh_patterns:
            self.assertRegex(
                pattern, r'["\']gh["\']\s*,', f"gh rule pattern is a bare prefix: {pattern}"
            )

    # Subcommands of `gh` the KB runs. A new entry here is a deliberate decision, because every
    # `gh` subcommand is a remote call (see docs/adr/workflow/skills-do-not-pre-approve-tools.md).
    GH_ALLOWED = frozenset(
        {
            "pr view",
            "pr list",
            "pr diff",
            "pr checks",
            "pr create",
            "pr edit",
            "issue view",
            "issue list",
            "run list",
            "run view",
            "repo view",
        }
    )

    def test_gh_allow_set_is_the_same_exact_set_in_all_three_configs(self):
        claude = json.loads((REPO / ".claude" / "settings.json").read_text())
        agy = json.loads((REPO / ".gemini" / "antigravity-cli" / "settings.json").read_text())
        claude_gh = {
            rule[len("Bash(gh ") : -1].removesuffix(" *")
            for rule in claude["permissions"]["allow"]
            if rule.startswith("Bash(gh")
        }
        agy_gh = {
            rule[len("command(gh ") : -1]
            for rule in agy["permissions"]["allow"]
            if rule.startswith("command(gh")
        }
        self.assertEqual(claude_gh, self.GH_ALLOWED)
        self.assertEqual(agy_gh, self.GH_ALLOWED)

        rules = (REPO / ".codex" / "rules" / "default.rules").read_text()
        codex_gh: set[str] = set()
        for block in re.split(r"(?m)^(?=prefix_rule\()", rules):
            if not block.startswith("prefix_rule(") or not re.search(
                r'decision\s*=\s*"allow"', block
            ):
                continue
            match = re.search(r"pattern\s*=\s*(\[.*?\]),\s*\n\s*decision", block, re.S)
            if match is None:
                self.fail("allow rule has no pattern")
            pattern = match.group(1)
            if not re.match(r'\[\s*["\']gh["\']', pattern):
                continue
            head = re.match(r'\[\s*["\']gh["\']\s*,\s*["\'](\w+)["\']', pattern)
            if head is None:
                self.fail(f"gh rule has no subcommand: {pattern}")
            first = head.group(1)
            rest = re.match(r'\[\s*["\']gh["\']\s*,\s*["\']\w+["\']\s*,\s*(.+)\]$', pattern, re.S)
            if rest is None:
                self.fail(f"gh rule has no third element: {pattern}")
            verbs = re.findall(r'["\'](\w+)["\']', rest.group(1))
            codex_gh.update(f"{first} {verb}" for verb in verbs)
        self.assertEqual(codex_gh, self.GH_ALLOWED)

    def test_codex_git_allow_rules_cover_the_claude_git_subcommands(self):
        claude = json.loads((REPO / ".claude" / "settings.json").read_text())
        claude_git = {
            rule[len("Bash(git ") : -1].removesuffix(" *")
            for rule in claude["permissions"]["allow"]
            if rule.startswith("Bash(git ")
        }
        rules = (REPO / ".codex" / "rules" / "default.rules").read_text()
        codex_git: set[str] = set()
        for block in re.split(r"(?m)^(?=prefix_rule\()", rules):
            if not re.search(r'decision\s*=\s*"allow"', block):
                continue
            match = re.search(
                r'pattern\s*=\s*\[\s*["\']git["\']\s*,\s*(.+?)\],\s*\n\s*decision', block, re.S
            )
            if match:
                codex_git.update(re.findall(r'["\']([\w-]+)["\']', match.group(1)))
        self.assertEqual(codex_git, claude_git)

    def test_npm_audit_fix_is_denied_or_prompted_in_every_tree(self):
        """`npm audit *` also matches `npm audit fix`, which runs an install and its scripts."""
        claude = json.loads((REPO / ".claude" / "settings.json").read_text())
        agy = json.loads((REPO / ".gemini" / "antigravity-cli" / "settings.json").read_text())
        for rule in ("Bash(npm audit fix *)", "Bash(npm audit fix)"):
            self.assertIn(rule, claude["permissions"]["deny"])
        self.assertIn("command(npm audit fix)", agy["permissions"]["deny"])
        rules = (REPO / ".codex" / "rules" / "default.rules").read_text()
        self.assertRegex(
            rules,
            r'(?s)prefix_rule\(\s*pattern\s*=\s*\[\s*["\']npm["\']\s*,\s*["\']audit["\']\s*,'
            r'\s*["\']fix["\']\s*\],\s*decision\s*=\s*"prompt"',
        )

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

    def test_shared_reference_citations_resolve_from_the_reference_root(self):
        """XR-7 validates `references/<category>/<file>.md` citations in the Claude tree only.
        The shared tree carries the same citations, so check them here: each must resolve from
        `.agents/references/`, not from the citing file's own directory."""
        _, agents = REFERENCE_TREES
        citation = re.compile(r"(?<![\w./-])references/([\w./-]+\.md)")
        dangling = []
        cited = 0
        for agents_file in sorted(agents.rglob("*.md")):
            rel = agents_file.relative_to(REPO)
            lines = agents_file.read_text(encoding="utf-8").splitlines()
            for n, line in enumerate(lines, 1):
                for target in citation.findall(line):
                    cited += 1
                    if not (agents / target).is_file():
                        dangling.append(f"{rel}:{n}: references/{target}")
        self.assertGreater(cited, 0, "no shared citations found; the check is vacuous")
        self.assertEqual(dangling, [], "shared citations do not resolve:\n" + "\n".join(dangling))

    def test_prefixed_citations_name_a_real_reference_or_skill(self):
        """`<category>:<name>` names `references/<category>/<name>.md` (an optional `#anchor` must
        match a heading there), and `skill:<name>` names a skill folder holding `SKILL.md`, both in
        the citing file's own tree: `.claude` cites the Claude tree; `.agents` and `.codex` cite
        `.agents`; `.gemini` cites its own skills or `.agents`. A citation counts in backticks, in
        bold, or bare in prose outside fences and code spans. A skill written in category form, or a
        `languages:*` wildcard, resolves nowhere; that is the drift #157 swept."""
        categories = sorted(
            {entry.name for root in REFERENCE_TREES for entry in root.iterdir() if entry.is_dir()}
        )
        kinds = "|".join(["skill", *map(re.escape, categories)])
        segment = r"(?:\.\.?|[\w*-]+)"
        target = r"((?:\.\.?|[A-Za-z*][\w*-]*)(?:/" + segment + r")*)(?:#([\w-]+))?"
        delimited = re.compile(r"(`|\*\*)(" + kinds + r"):" + target + r"\1")
        bare = re.compile(r"(?<![\w:/.`*#'\"-])(" + kinds + r"):" + target + r"(?![\w:/`*])")
        bold = re.compile(r"\*\*[^*]+\*\*")

        def slug(heading: str) -> str:
            kept = re.sub(r"[^\w\- ]", "", heading.strip().lower())
            return kept.replace(" ", "-")

        def resolves(tree: str, kind: str, name: str, anchor: str) -> bool:
            if {".", ".."} & set(name.split("/")):
                return False
            if kind == "skill":
                roots = {
                    ".claude": [REPO / ".claude" / "skills"],
                    ".gemini": [REPO / ".gemini" / "antigravity-cli" / "skills"],
                }.get(tree, []) + ([] if tree == ".claude" else [REPO / ".agents" / "skills"])
                folders = [name] if tree == ".claude" else [name, f"cmd-{name}"]
                return any((root / f / "SKILL.md").is_file() for root in roots for f in folders)
            base = REPO / (".claude" if tree == ".claude" else ".agents") / "references"
            reference = base / kind / f"{name}.md"
            if not reference.is_file():
                return False
            if not anchor:
                return True
            text = reference.read_text(encoding="utf-8")
            return anchor in {slug(h) for h in heading_texts(text)}

        counts = {"skill": 0, "reference": 0}
        dangling = []
        for name in tracked_tree_files():
            path = REPO / name
            if path.suffix not in {".md", ".toml"} or not path.is_file():
                continue
            tree = name.split("/", 1)[0]
            try:
                text = path.read_text(encoding="utf-8")
            except UnicodeDecodeError as exc:
                self.fail(f"{name}: not UTF-8 ({exc})")
            for lineno, line in lines_outside_fences(text):
                found = [m.group(2, 3, 4) for m in delimited.finditer(line)]
                prose = CODE_SPAN.sub("", bold.sub("", line))
                found += [m.group(1, 2, 3) for m in bare.finditer(prose)]
                for kind, cited, anchor in found:
                    counts["skill" if kind == "skill" else "reference"] += 1
                    if not resolves(tree, kind, cited, anchor or ""):
                        suffix = f"#{anchor}" if anchor else ""
                        dangling.append(f"{name}:{lineno}: {kind}:{cited}{suffix}")
        for kind, count in counts.items():
            self.assertGreater(count, 0, f"no {kind} citations found; that half is vacuous")
        self.assertEqual(
            dangling,
            [],
            "prefixed citations resolve to no reference or skill; cite a skill by bare name or "
            "`skill:<name>`, and a reference by path:\n" + "\n".join(dangling),
        )

    def test_existing_code_discipline_is_one_document_in_every_tree(self):
        """The embedded diff-review guidance tracks the reference bodies as well as the headings."""
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
                    key = reference_form_key(rel)
                    self.assertIsNotNone(key, f"{rel} is outside REFERENCE_FORMS")
                    assert key is not None
                    prefix = REFERENCE_FORMS[key]
                    pointer = f"{prefix}workflow/existing-code-discipline.md"
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
        supporting = 0
        for skill_dir in sorted(path.parent for path in agents_skills.glob("*/SKILL.md")):
            name = skill_dir.name
            if name.startswith("cmd-j-"):
                continue
            # Every file ships to Gemini, not just SKILL.md: a skill's references drift the same way.
            for source in sorted(path for path in skill_dir.rglob("*") if path.is_file()):
                rel = source.relative_to(agents_skills)
                mirror = gemini_skills / rel
                with self.subTest(file=str(rel)):
                    self.assertTrue(mirror.is_file(), f"{rel} has no Gemini copy")
                    self.assertEqual(
                        source.read_bytes(),
                        mirror.read_bytes(),
                        f"{rel}: the .agents and .gemini copies have diverged",
                    )
                if source.name != "SKILL.md":
                    supporting += 1
            compared += 1
        self.assertGreater(compared, 0, "no shared skills compared; the check is vacuous")
        self.assertGreater(supporting, 0, "no supporting files compared; the rglob is vacuous")

    def test_fix_loops_stop_after_two_rounds(self):
        """Fix loops end after two open rounds and halt rather than mark the task done.

        Matches CLAUDE.md's stop-after-two-attempts rule. Each surface pins its own stop sentence,
        counted, so deleting one of a checklist's two bounds fails too.
        """
        sdd = ("skills", "subagent-driven-development", "SKILL.md")
        checklist = ("references", "workflow", "task-execution-checklists.md")
        sdd_stop = "Stop after two fix rounds that leave a Critical/Important finding open"
        sdd_halt = "do not mark the task done, start the next task, or commit the phase"
        checklist_stop = (
            "If issues remain after two fix rounds, stop, report them, and do not mark the task "
            "complete."
        )
        surfaces = {
            REPO.joinpath(".claude", *sdd): {sdd_stop: 1, sdd_halt: 1},
            REPO.joinpath(".agents", *sdd): {sdd_stop: 1, sdd_halt: 1},
            REPO.joinpath(".gemini", "antigravity-cli", *sdd): {sdd_stop: 1, sdd_halt: 1},
            REPO.joinpath(".claude", *checklist): {checklist_stop: 2},
            REPO.joinpath(".agents", *checklist): {checklist_stop: 2},
        }
        for path, expected in surfaces.items():
            text = path.read_text(encoding="utf-8")
            with self.subTest(path=str(path.relative_to(REPO))):
                self.assertIsNone(re.search(r"(?i)repeat until (clean|pass)", text))
                for sentence, count in expected.items():
                    self.assertEqual(text.count(sentence), count, sentence)

    def test_plans_record_decisions_not_code(self):
        """See docs/adr/workflow/plans-record-decisions-not-code.md."""
        for rel in WRITING_PLANS_OWNERS:
            text = (REPO / rel).read_text(encoding="utf-8")
            with self.subTest(path=rel):
                for phrase in (
                    "set of decisions the implementer cannot make alone",
                    "**Proportion**",
                    "**Interfaces:**",
                    "expand-contract",
                ):
                    self.assertIn(phrase, text)
                for retired in ("**Complete code**", "Junior-engineer bar", "2-5 minutes"):
                    self.assertNotIn(retired, text)

    def test_plan_red_team_checks_cited_paths_and_comprehension(self):
        for path in (*J_PLAN_COMMANDS, CLAUDE_J_PLAN_COMMAND):
            text = path.read_text(encoding="utf-8")
            with self.subTest(path=str(path.relative_to(REPO))):
                self.assertIn("cites as existing", text)
                self.assertIn("cannot paraphrase", text)

    def test_fix_loops_rereview_only_the_fix(self):
        for path in EXECUTION_MODE_OWNERS:
            text = path.read_text(encoding="utf-8")
            with self.subTest(path=str(path.relative_to(REPO))):
                self.assertIn("NOT ADDRESSED", text)
                self.assertIn("one fix subagent", text)
                self.assertIn("never downgrades a finding's severity", text)

    def _assert_all_contain(self, paths, phrase):
        for path in paths:
            with self.subTest(path=str(path.relative_to(REPO)), phrase=phrase):
                self.assertIn(phrase, path.read_text(encoding="utf-8"))

    def test_review_surfaces_name_the_weakened_bar(self):
        self._assert_all_contain(
            (
                REPO / ".claude" / "agents" / "code-reviewer.md",
                REPO / ".codex" / "agents" / "code-reviewer.toml",
                REPO / ".gemini" / "config" / "agents" / "code-reviewer.md",
            ),
            "Weakened Quality Bar",
        )
        self._assert_all_contain(
            tuple(
                REPO / tree / "references" / "workflow" / "review-checklists.md"
                for tree in (".claude", ".agents")
            ),
            "## Weakened Quality Bar",
        )

    def test_diff_review_reads_review_md_from_the_base_branch(self):
        self._assert_all_contain(DIFF_REVIEW_DISPATCHERS, "git show origin/main:REVIEW.md")

    def test_debugging_starts_from_a_red_capable_loop(self):
        self._assert_all_contain(
            tuple(root / "debugging-methodology" / "SKILL.md" for root in SKILL_TREE_ROOTS),
            "No red-capable command, no Phase 2",
        )

    def test_pr_bodies_state_merge_danger(self):
        skills = tuple(
            root / name / "SKILL.md"
            for root in SKILL_TREE_ROOTS
            for name in ("finishing-branch", "github-issue-resolution")
        )
        agents = (
            REPO / ".claude" / "agents" / "create-pr.md",
            REPO / ".codex" / "agents" / "create-pr.toml",
            REPO / ".gemini" / "config" / "agents" / "create-pr.md",
        )
        diff_review = (REPO / ".agents" / "skills" / "cmd-j-diff-review" / "SKILL.md",)
        self._assert_all_contain((*skills, *agents, *diff_review), "Merge Danger")

    def test_security_findings_carry_a_state(self):
        self._assert_all_contain(
            tuple(
                REPO / tree / "references" / "security" / "vulnerability-review-pipeline.md"
                for tree in (".claude", ".agents")
            ),
            "## Finding States",
        )
        self._assert_all_contain(
            (
                REPO / ".claude" / "agents" / "security-reviewer.md",
                REPO / ".codex" / "agents" / "security-reviewer.toml",
                REPO / ".gemini" / "config" / "agents" / "security-reviewer.md",
            ),
            "demonstrated impact",
        )

    def test_paper_surfaces_carry_harvest_checks(self):
        def command(name):
            return (
                REPO / ".claude" / "commands" / f"{name}.md",
                REPO / ".agents" / "skills" / f"cmd-{name}" / "SKILL.md",
                REPO / ".gemini" / "antigravity-cli" / "skills" / name / "SKILL.md",
            )

        def research(name):
            return tuple(
                REPO / tree / "references" / "research" / name for tree in (".claude", ".agents")
            )

        analysis = command("j-paper-analysis")
        self._assert_all_contain(
            (*research("paper-analysis-methodology.md"), *analysis), "Number provenance"
        )
        self._assert_all_contain((*research("output-template.md"), *analysis), "Linchpin check")
        for phrase in ("Keep each claim's strength", "AI-use disclosure"):
            self._assert_all_contain(command("j-paper"), phrase)
        self._assert_all_contain(command("j-paper"), "Citation Verification")
        self._assert_all_contain(research("literature-review.md"), "## Citation Verification")
        self._assert_all_contain(
            (*research("rebuttal-writing.md"), *command("j-paper")), "Rebuttal"
        )
        self._assert_all_contain(
            (*research("paper-analysis-methodology.md"), *analysis), "Confidential manuscript"
        )
        self._assert_all_contain(
            (*research("output-template.md"), *analysis), "the change that would resolve it"
        )
        self._assert_all_contain(command("j-paper"), "rebuttal-writing.md")
        for phrase in ("| Limitations |", "| Reproducibility |", "| Error bars |", "upon request"):
            self._assert_all_contain(command("j-paper"), phrase)
        self._assert_all_contain(
            (*research("output-template.md"), *analysis), "Proportionate requests"
        )
        self._assert_all_contain(analysis, "do not search by title")

    def test_ui_guidance_is_scoped_and_diagrams_accessible(self):
        def frontend(name):
            return tuple(
                REPO / tree / "references" / "frontend" / name for tree in (".claude", ".agents")
            )

        self._assert_all_contain(frontend("premium-design-aesthetics.md"), "**The brief wins.**")
        self._assert_all_contain(frontend("web-animation-patterns.md"), "fewer and gentler")
        self._assert_all_contain(
            tuple(root / "html-report-authoring" / "SKILL.md" for root in SKILL_TREE_ROOTS),
            "aria-labelledby",
        )

    def test_claude_command_effort_only_lowers_the_session_level(self):
        """`effort:` is free text to YAML, so a typo loads silently and the command runs at the
        session default. Commands only lower effort (docs/adr/workflow/
        commands-inherit-the-session-model.md): raising it was never measured to pay, so `high`,
        `xhigh`, and `max` fail here. Pin the value set, not each command's choice."""
        allowed = {"low", "medium"}
        effort = re.compile(r"^effort:\s*(\S+)\s*$", re.MULTILINE)
        checked = 0
        for path in sorted((REPO / ".claude" / "commands").glob("*.md")):
            frontmatter = path.read_text(encoding="utf-8").split("---", 2)[1]
            for value in effort.findall(frontmatter):
                with self.subTest(command=path.name):
                    self.assertIn(value, allowed, f"{path.name}: unsupported effort {value!r}")
                checked += 1
        self.assertGreater(checked, 0, "no command sets effort; the check is vacuous")

    def test_agents_carry_no_persistent_memory(self):
        """`memory:` auto-grants Write and Edit and injects an uncurated index outside the frozen
        review packet, so no Claude agent sets it (docs/adr/workflow/
        agents-carry-no-persistent-memory.md)."""
        memory = re.compile(r"(?m)^memory:")
        agents = sorted((REPO / ".claude" / "agents").glob("*.md"))
        self.assertTrue(agents, "no Claude agents found; the check is vacuous")
        offenders = [path.name for path in agents if memory.search(frontmatter_block(path))]
        self.assertEqual(offenders, [], f"agents set memory: {offenders}")

    def test_side_effect_commands_nothing_invokes_are_user_only(self):
        """Exactly the side-effect commands no skill or coordinator invokes set
        `disable-model-invocation: true`. The set comes from observed invocations recorded in
        docs/adr/workflow/side-effect-commands-are-user-invoked.md: flagging a command that a skill
        chains to would break that flow."""
        flag = re.compile(r"(?m)^disable-model-invocation:\s*true\s*$")
        commands = sorted((REPO / ".claude" / "commands").glob("*.md"))
        self.assertTrue(commands, "no Claude commands found; the check is vacuous")
        flagged = {path.stem for path in commands if flag.search(frontmatter_block(path))}
        self.assertEqual(flagged, USER_ONLY_COMMANDS)

    def test_codex_user_only_commands_disable_implicit_invocation(self):
        """Codex ignores `disable-model-invocation` and reads `agents/openai.yaml` instead, so the
        user-only commands' `.agents` twins turn implicit invocation off there, and no `.agents`
        skill keeps the inert key (docs/adr/workflow/side-effect-commands-are-user-invoked.md)."""
        policy = re.compile(r"(?m)^policy:\s*\n\s+allow_implicit_invocation:\s*false\s*$")
        command_skills = sorted((REPO / ".agents" / "skills").glob("cmd-j-*"))
        self.assertTrue(command_skills, "no .agents command skills found; the check is vacuous")
        with_policy = set()
        for folder in command_skills:
            config = folder / "agents" / "openai.yaml"
            if config.is_file():
                with_policy.add(folder.name.removeprefix("cmd-"))
                with self.subTest(skill=folder.name):
                    self.assertRegex(config.read_text(encoding="utf-8"), policy)
        self.assertEqual(with_policy, USER_ONLY_COMMANDS)
        inert = [
            str(path.relative_to(REPO))
            for path in sorted((REPO / ".agents" / "skills").glob("*/SKILL.md"))
            if "disable-model-invocation" in frontmatter_block(path)
        ]
        self.assertEqual(inert, [], "inert disable-model-invocation in .agents skills")

    def test_llm_judges_grade_checkable_claims(self):
        """A 1-5 rating is not checkable and drifts between judge runs; judges grade per-task
        PASS/FAIL claims instead, and the eval reference owns task design and hillclimbing."""
        banned = ("scale of 1-5", "1-5 scale", "<1-5>", "both orders and average")
        for tree in (".claude", ".agents"):
            root = REPO / tree / "references" / "ai-ml"
            for rel in (
                "eval-and-benchmarking.md",
                "dataset-management/synthetic-data-generation.md",
            ):
                text = (root / rel).read_text(encoding="utf-8")
                for phrase in banned:
                    with self.subTest(file=f"{tree}/{rel}", phrase=phrase):
                        self.assertNotIn(phrase, text)
            names = heading_texts((root / "eval-and-benchmarking.md").read_text(encoding="utf-8"))
            for heading in (
                "Eval Task Design",
                "Validity Checks",
                "Golden Sets",
                "Hillclimbing",
                "Overfitting",
            ):
                with self.subTest(tree=tree, heading=heading):
                    self.assertIn(heading, names)

    def test_scoped_hook_section_exists_in_both_trees(self):
        """Skill- and agent-frontmatter hooks have their own lifetime and trust rules; both hook
        references carry the section so the twins stay in parity."""
        for tree in (".claude", ".agents"):
            text = (REPO / tree / "references" / "workflow" / "hook-patterns.md").read_text(
                encoding="utf-8"
            )
            with self.subTest(tree=tree):
                self.assertIn("Skill and Agent Scoped Hooks", heading_texts(text))

    def test_session_boundaries_section_exists_in_both_trees(self):
        """Both context-efficiency twins tell the reader when a session reset or model switch costs
        the cache."""
        for tree in (".claude", ".agents"):
            text = (REPO / tree / "references" / "workflow" / "context-efficiency.md").read_text(
                encoding="utf-8"
            )
            with self.subTest(tree=tree):
                self.assertIn("Session Boundaries", heading_texts(text))

    def test_consult_commands_list_their_agent_references(self):
        for command, agents in CONSULT_AGENTS.items():
            want = set().union(*(agent_reference_names(agent) for agent in agents))
            for path, prefix in command_copies(command):
                with self.subTest(path=str(path.relative_to(REPO))):
                    self.assertTrue(want)
                    text = path.read_text(encoding="utf-8")
                    self.assertEqual(reference_names(text), want)
                    for block in reference_blocks(text):
                        self.assertTrue(block.startswith(prefix), f"{block} lacks {prefix}")
        for agent in sorted({a for agents in CONSULT_AGENTS.values() for a in agents}):
            gemini = REPO / ".gemini" / "config" / "agents" / f"{agent}.md"
            with self.subTest(path=str(gemini.relative_to(REPO))):
                got = reference_names(gemini.read_text(encoding="utf-8"))
                self.assertEqual(got, agent_reference_names(agent))

    def test_agent_reference_indexes_cover_their_directory(self):
        for tree, refs in AGENT_INDEX_TREES:
            for path in sorted((REPO / tree / "agents").glob("*.md")):
                if path.stem in CURATED_REFERENCE_INDEXES:
                    continue
                gaps = library_gaps(path.read_text(encoding="utf-8"), REPO / refs / "references")
                with self.subTest(path=str(path.relative_to(REPO))):
                    self.assertEqual({k: v for k, v in gaps.items() if v}, {})

    def test_curated_reference_indexes_are_live(self):
        for agent in CURATED_REFERENCE_INDEXES:
            path = REPO / ".claude" / "agents" / f"{agent}.md"
            with self.subTest(agent=agent):
                gaps = library_gaps(
                    path.read_text(encoding="utf-8"), REPO / ".claude" / "references"
                )
                self.assertTrue(any(gaps.values()), f"{agent} indexes its whole directory")

    def test_no_kb_asset_cites_a_project_relative_reference(self):
        """A tool resolves a relative path against the user's project, where a cloned repo can
        plant a same-named file. Each tree and kind cites references in its one chosen form."""
        tracked = subprocess.run(
            [
                "git",
                "ls-files",
                "-z",
                *sorted({f"{tree}/{kind}" for tree, kind in REFERENCE_FORMS}),
            ],
            cwd=REPO,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.split("\0")
        scanned = 0
        offenders = []
        exempt_hits = set()
        for rel in tracked:
            key = reference_form_key(rel)
            path = REPO / rel
            if key is None or not path.is_file():
                continue
            try:
                text = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            scanned += 1
            prefix = REFERENCE_FORMS[key]
            bad = project_relative_citations(text, prefix)
            bad += bare_reference_dirs(text, bare_prefixes(key))
            if rel in RELATIVE_REFERENCE_EXEMPT:
                if bad:
                    exempt_hits.add(rel)
                continue
            offenders.extend(f"{rel}: {citation}" for citation in bad)
            if path.suffix not in EXPANDED_SUFFIXES and any(a in text for a in HOME_ANCHORS):
                offenders.append(f"{rel}: home anchor in a file sync does not expand")
        self.assertGreater(scanned, 0, "no KB files scanned; the check is vacuous")
        self.assertEqual(
            sorted(exempt_hits),
            sorted(RELATIVE_REFERENCE_EXEMPT),
            "the exempt copies no longer hold the relative form",
        )
        self.assertEqual(offenders, [], "\n".join(offenders[:40]))

    def test_project_relative_reference_scan_controls(self):
        for (tree, kind), prefix in REFERENCE_FORMS.items():
            with self.subTest(form=f"{tree}/{kind}"):
                self.assertEqual(
                    project_relative_citations(f"Read `{prefix}cloud/x.md`.", prefix), []
                )
        agents = AGENTS_REFERENCES
        self.assertEqual(
            project_relative_citations("Read `.agents/references/cloud/x.md`.", agents),
            [".agents/references/cloud/"],
        )
        self.assertEqual(
            project_relative_citations("Read `references/cloud/x.md`.", agents),
            ["references/cloud/"],
        )
        self.assertEqual(
            project_relative_citations("Read `./references/cloud/x.md`.", agents),
            ["./references/cloud/"],
        )
        self.assertEqual(
            project_relative_citations("Read `../references/cloud/x.md`.", agents),
            ["../references/cloud/"],
        )
        self.assertEqual(
            project_relative_citations("Read `~/.claude/references/cloud/x.md`.", agents),
            ["~/.claude/references/cloud/"],
        )
        self.assertEqual(
            project_relative_citations("See [x](.agents/references/cloud/x.md).", agents),
            [".agents/references/cloud/"],
        )
        self.assertEqual(
            project_relative_citations("Read `code-review-patterns/references/x.md`.", agents), []
        )
        for text, token in (
            ('Glob(".claude/references/*/")', ".claude/references/"),
            ("Scan .agents/references/ for patterns", ".agents/references/"),
            ("Reference library at .claude/references/{category}/:", ".claude/references/"),
            ("Scan `~/.claude/references/` here", "~/.claude/references/"),
            ("Scan `~/.agents/references/` here", None),
        ):
            with self.subTest(text=text):
                self.assertEqual(bare_reference_dirs(text, (agents,)), [token] if token else [])
        self.assertEqual(bare_reference_dirs('Glob("~/.agents/references/*/")', (agents,)), [])

    def test_consult_cases_measure_reference_reads(self):
        """Each consult case carries a read-refs grader on its agents' reference directories.

        A reference-grounded case (its description names one reference file) is graded on that
        file's directory alone, so a read elsewhere does not count as reaching it.
        """
        cases_root = REPO / "evals" / "cases" / "effects"
        for command, agents in CONSULT_AGENTS.items():
            dirs = {name.split("/")[0] for a in agents for name in agent_reference_names(a)}
            dirs.discard("workflow")
            case_dirs = sorted(cases_root.glob(f"{command}--effect-*"))
            self.assertTrue(case_dirs, command)
            for case in case_dirs:
                with self.subTest(case=case.name):
                    grader = (case / "graders" / "read-refs.md").read_text(encoding="utf-8")
                    self.assertIn("tool: Read", grader)
                    prompt = (case / "prompt.md").read_text(encoding="utf-8")
                    described = re.search(r"^description:.*$", prompt, re.M)
                    line = described.group(0) if described else ""
                    if ".claude/references/" not in line:
                        for directory in dirs:
                            self.assertIn(directory, grader, "read-refs omits an agent directory")
                        continue
                    grounded = re.search(r"; \.claude/references/([\w-]+)/[\w.-]+; ", line)
                    if grounded is None:
                        self.fail(f"unreadable reference path in: {line}")
                    own = grounded.group(1)
                    self.assertIn(
                        own, dirs, "the named reference is outside the agents' directories"
                    )
                    self.assertIn(
                        f"/{own}/", grader, "read-refs omits the named reference's directory"
                    )
                    for other in dirs - {own}:
                        self.assertNotIn(
                            other, grader, "a grounded case grades one directory alone"
                        )

    def test_j_plan_states_an_outside_plan_mode_path(self):
        text = CLAUDE_J_PLAN_COMMAND.read_text(encoding="utf-8")
        self.assertGreaterEqual(text.count("Outside plan mode"), 2)
        self.assertIn("~/.claude/plans/<repo-name>-", text)
        self.assertIn('basename "$(git rev-parse --show-toplevel)"', text)
        self.assertIn("[a-z0-9-]", text)

    def test_fetched_citation_fields_are_untrusted_tex(self):
        for tree in (".claude", ".agents"):
            path = REPO / tree / "references" / "research" / "literature-review.md"
            text = path.read_text(encoding="utf-8")
            needles = (
                "\\write18",
                "\\input",
                "shell-escape",
                '{\\"o}',
                "@preamble",
                "Reject anything outside the allowlist",
                "every fetched field (title, author, venue, note, abstract, and any other)",
                "`url` or `doi` value verbatim",
                "depth never goes below 0",
                "line-leading `@`",
                "`\\mathbb{<letters>}`",
                "bidi",
                "restricted `\\write18`",
            )
            for needle in needles:
                with self.subTest(path=path.relative_to(REPO), needle=needle):
                    self.assertIn(needle, text)
        paper = [p for p in PROSE_TELL_CONSUMERS if "j-paper" in str(p)]
        self.assertTrue(paper)
        for path in paper:
            with self.subTest(path=path.relative_to(REPO)):
                self.assertIn("untrusted TeX", path.read_text(encoding="utf-8"))

    def test_no_kb_asset_recommends_uv_sync_frozen(self):
        out = subprocess.run(
            ["git", "ls-files", ".claude", ".agents", ".codex", ".gemini"],
            cwd=REPO,
            capture_output=True,
            text=True,
            check=True,
            stdin=subprocess.DEVNULL,
        ).stdout.split()
        for rel in out:
            path = REPO / rel
            if not path.is_file():
                continue
            with self.subTest(path=rel):
                text = path.read_text(encoding="utf-8", errors="ignore")
                self.assertNotRegex(text, r"uv sync --frozen")

    def test_eval_audit_asset_fixes_hold(self):
        def read(rel):
            return (REPO / rel).read_text(encoding="utf-8")

        for tree in (".claude", ".agents", ".gemini/antigravity-cli"):
            rel = f"{tree}/skills/using-git-worktrees/SKILL.md"
            with self.subTest(row="using-git-worktrees", path=rel):
                text = read(rel)
                self.assertIn("-- .gitignore", text)
                self.assertIn(".git/info/exclude", text)
                self.assertIn("git rev-parse --git-path info/exclude", text)
                self.assertNotIn("Add to .gitignore, commit, then proceed", text)
                self.assertNotIn("Add to .gitignore + commit", text)
        for tree in (".claude", ".agents"):
            ref = f"{tree}/references"
            with self.subTest(row="uv-workflows", tree=tree):
                text = read(f"{ref}/languages/uv-workflows.md")
                self.assertIn("`--locked` in CI", text)
                self.assertNotIn("MUST use --frozen", text)
            with self.subTest(row="python-packaging", tree=tree):
                text = read(f"{ref}/languages/python-packaging-and-distribution.md")
                self.assertNotIn("pydantic>=2.0,<3.0", text)
                self.assertNotRegex(text, r"pydantic>=2[^\"\n]*<3")
            with self.subTest(row="js-ts-patterns", tree=tree):
                text = read(f"{ref}/languages/js-ts-patterns.md")
                lines = [ln for ln in text.splitlines() if '"import"' in ln and '"types"' in ln]
                self.assertTrue(lines)
                for line in lines:
                    self.assertLess(line.index('"types"'), line.index('"import"'))
            with self.subTest(row="scope-review", tree=tree):
                text = read(f"{ref}/product/scope-review-methodology.md")
                self.assertIn("missing capabilities", text)
                self.assertNotIn("10x version", text)
            with self.subTest(row="file-storage-patterns", tree=tree):
                text = read(f"{ref}/cloud/file-storage-patterns.md")
                self.assertIn("sign each part on demand", text)


if __name__ == "__main__":
    unittest.main()
