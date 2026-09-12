"""Architecture checks for the tool-specific agent configuration trees."""

from __future__ import annotations

import json
import re
import subprocess
import unittest
from pathlib import Path

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

PLAN_STORAGE_SKILLS = (
    REPO / ".agents" / "skills" / "writing-plans" / "SKILL.md",
)

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
    "If information can be a table, make it a table",
)

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


def skill_directories(root: Path) -> set[str]:
    return {path.parent.name for path in root.glob("*/SKILL.md")}


def description(path: Path) -> str:
    match = FRONTMATTER_DESCRIPTION.search(path.read_text())
    if match is None:
        raise AssertionError(f"missing description: {path}")
    return match.group(1)


class AgentConfigArchitectureTests(unittest.TestCase):
    def test_shared_skills_do_not_contain_agent_wrappers(self):
        wrappers = sorted((REPO / ".agents" / "skills").glob("agent-*/SKILL.md"))
        self.assertEqual(wrappers, [])

    def test_command_skills_match_all_native_command_sets(self):
        command_skills = {
            name.removeprefix("cmd-")
            for name in skill_directories(REPO / ".agents" / "skills")
            if name.startswith("cmd-j-")
        }
        codex = {
            path.stem for path in (REPO / ".codex" / "prompts").glob("j-*.md")
        }
        claude = {
            path.stem for path in (REPO / ".claude" / "commands").glob("j-*.md")
        }
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
        codex = {
            path.stem for path in (REPO / ".codex" / "agents").glob("*.toml")
        }
        claude = {
            path.stem for path in (REPO / ".claude" / "agents").glob("*.md")
        }
        antigravity = {
            path.stem for path in (REPO / ".gemini" / "config" / "agents").glob("*.md")
        }

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

    def test_structure_rule_has_one_owner_carrying_the_conditional(self):
        """The owner states BOTH halves: the density ordering, and that structure the question did
        not ask for is a net cost anyway. Half the rule reads as a licence for the other half."""
        for path in STRUCTURE_RULE_OWNERS:
            content = path.read_text()
            with self.subTest(path=path.relative_to(REPO)):
                self.assertIn("per unit of\ninformation carried", content)
                self.assertIn("altitude", content.lower())

    def test_always_loaded_configs_do_not_restate_the_unconditional_structure_rule(self):
        """A measured-wrong rule must not survive in a file re-sent on every request. The configs
        point at context-efficiency instead, same convention as REPORT_POINTERS."""
        for path in REPORT_POINTERS:
            content = path.read_text()
            with self.subTest(path=path.relative_to(REPO)):
                for phrase in UNCONDITIONAL_STRUCTURE_PHRASES:
                    self.assertNotIn(phrase, content)
                self.assertIn("context-efficiency", content)

    def test_diff_review_hands_reviewers_the_head_sha(self):
        for path in DIFF_REVIEW_DISPATCHERS:
            content = path.read_text()
            with self.subTest(path=path.relative_to(REPO)):
                self.assertIn("git rev-parse HEAD", content)
                self.assertIn("HEAD SHA", content)
                self.assertIn("do not create or request a worktree", content)

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
                line.strip()
                for line in path.read_text().splitlines()
                if "superpowers" in line
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
        self.assertEqual(missing, [], "assets name scripts that do not exist:\n" + "\n".join(missing))

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
                cmd = rule[len("Bash("):-1].removesuffix(" *")
                if cmd == "grep:*":
                    cmd = "grep"
                expected = f"command({cmd})"
                self.assertIn(expected, agy_allow, f"Missing allowed command: {expected}")

        for rule in claude_deny:
            if rule.startswith("Bash(") and rule.endswith(")"):
                cmd = rule[len("Bash("):-1].removesuffix(" *")
                expected = f"command({cmd})"
                self.assertIn(expected, agy_deny, f"Missing denied command: {expected}")
            elif rule.startswith("Read(") and rule.endswith(")"):
                path = rule[len("Read("):-1].removesuffix("/**")
                if path.startswith("./"):
                    continue  # non-absolute paths are rejected by the Antigravity sandbox
                expected = f"read_file({path})"
                self.assertIn(expected, agy_deny, f"Missing denied read_file: {expected}")

    def test_legacy_gemini_artifacts_are_not_tracked(self):
        """Legacy Gemini CLI policy engine, hooks, commands, and agents must not be present in .gemini/."""
        self.assertFalse((REPO / ".gemini" / "policies").exists(), "legacy policies/ must not exist")
        self.assertFalse((REPO / ".gemini" / "hooks").exists(), "legacy hooks/ must not exist")
        self.assertFalse((REPO / ".gemini" / "settings.json").exists(), "root .gemini/settings.json must not exist")
        self.assertFalse((REPO / ".gemini" / "commands").exists(), "legacy commands/ must not exist")
        self.assertFalse((REPO / ".gemini" / "agents").exists(), "legacy agents/ must not exist")


if __name__ == "__main__":
    unittest.main()
