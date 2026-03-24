# Shared Agent Knowledge Base

This tree is the canonical, tool-agnostic knowledge base following the [Agent Skills specification](https://agentskills.io/specification). `setup.sh` syncs it to `~/.agents/` and into tool-specific directories (`~/.cursor/skills/`, etc.).

- `skills/` contains all skills: `agent-*` (specialist roles), `cmd-*` (command entry points), and `workflow/`/`testing/`/`migration/` (procedural workflows).
- `references/` contains shared domain knowledge organized by category.

Tool-specific config, hooks, and rules remain in `.claude/`, `.codex/`, and `.cursor/`.
