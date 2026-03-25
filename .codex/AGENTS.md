# Codex Working Agreements

## Core Defaults

- Verify before claiming success. Run the relevant command in the current session and report the actual result.
- Prefer search-first exploration. Use fast file discovery before reading large files.
- Keep work complete. Avoid partial implementations, stubbed code, and unfinished docs.
- Push back on weak assumptions with concrete technical reasoning.
- Prefer small, atomic changes and review the diff before committing.

## Knowledge Base

- Global Codex config lives under `~/.codex/`.
- User skills live under `~/.agents/skills/` and domain references under `~/.agents/references/`.
- Codex slash prompts live in `~/.codex/prompts/`, legacy command source files live in `~/.codex/commands/`, and agent definitions live in `~/.codex/agents/`.
- Before implementation, check whether an applicable skill already exists.

## Execution Defaults

- Prefer `rg` and `rg --files` for search.
- Avoid destructive git commands unless explicitly requested.
- Ask before adding new dependencies or changing external services.
- For reviews, lead with findings, include file paths and line numbers, and prioritize correctness, regressions, security, and missing tests.

## Execution Model

- Do NOT use subagents or agent teams. Process all work linearly in a single context.
