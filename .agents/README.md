# Shared Agent Knowledge Base

This tree follows the [Agent Skills specification](https://agentskills.io/specification). `setup.sh` syncs it to `~/.agents/`.

- `skills/` contains Codex `$cmd-j-*` command entry points and Codex-tailored workflow skills (for example, `design-first`, `code-quality`, and `test-driven-development`). Specialist agents live in each tool's native agent directory.
- `references/` contains shared Codex and Google Antigravity domain knowledge organized by category.

Tool-specific agents, commands, config, and rules live in `.claude/`, `.codex/`, and `.gemini/`. Antigravity reads its workflow and `j-*` command skills from `.gemini/antigravity-cli/skills/`, keeping skill-set and supporting-file parity with `.agents/skills/` while adapting bodies and descriptions per harness (`tests/test_agent_config.py`). Claude remains self-contained under `.claude/`.
