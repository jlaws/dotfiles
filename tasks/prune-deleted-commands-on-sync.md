# Task: Prune Deleted Commands From Every Synced Tree

## Context
Found while planning `feat/plan-close-review-and-lean-planning`.

## Problem
`_agent_removal_paths` (`macos_setup/dotfiles.py:292-305`) prunes only `.agents/skills/cmd-j-*`. Commands deleted from `.claude/commands/`, `.codex/prompts/`, and `.gemini/antigravity-cli/skills/` stay deployed. Observed: `~/.claude/commands/j-write-plan.md` and `~/.codex/prompts/j-write-plan.md`, both deleted from the repo in #56. This contradicts `docs/adr/workflow/merge-agent-configs-during-setup.md:30`, which says stale removed commands are cleaned up.

## Scope
1. Add a failing test in `tests/test_dotfiles.py` that syncs a target holding a command the repo no longer ships in each of those trees.
2. Extend the prune in `_agent_removal_paths` to `.claude/commands/`, `.codex/prompts/`, and `.gemini/antigravity-cli/skills/`.
3. Run `make verify`.
