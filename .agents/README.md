# Shared Agent Knowledge Base

This tree is the shared Codex/Gemini knowledge base following the [Agent Skills specification](https://agentskills.io/specification). `setup.sh` syncs it to `~/.agents/`.

- `skills/` contains `$cmd-j-*` command entry points and reusable workflows with bare names (for example, `design-first`, `code-quality`, and `test-driven-development`). Specialist agents live in each tool's native agent directory.
- `references/` contains shared domain knowledge organized by category.

Tool-specific agents, commands, config, hooks, and rules remain in `.claude/`, `.codex/`, and `.gemini/`. Antigravity reads its skills from `.gemini/antigravity-cli/skills/`, a byte-identical copy of the shared skills that `setup.sh` syncs; `tests/test_agent_config.py` enforces the match. Claude remains self-contained under `.claude/`.
