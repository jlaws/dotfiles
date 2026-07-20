# Shared Agent Knowledge Base

This tree is the shared Codex/Gemini knowledge base following the [Agent Skills specification](https://agentskills.io/specification). `setup.sh` syncs it to `~/.agents/`.

- `skills/` contains `$cmd-j-*` command entry points and reusable workflows with bare names (for example, `design-first`, `code-quality`, and `test-driven-development`). Specialist agents live in each tool's native agent directory.
- `references/` contains shared domain knowledge organized by category.

Tool-specific agents, commands, config, hooks, and rules remain in `.claude/`, `.codex/`, and `.gemini/`. Gemini CLI auto-discovers `.agents/skills/` natively, so no separate skill copy under `.gemini/` is needed. Claude remains self-contained under `.claude/`.
