---
name: review-claudemd
description: "Analyze recent conversation history to improve CLAUDE.md files — find violated instructions, missing patterns, and outdated rules. Use when tuning Claude Code behavior or after a batch of sessions."
argument-hint: "<focus-area-or-session-notes>"
---

Before invoking analysis, gather context and resolve scope from `$ARGUMENTS`:

1. **Resolve scope**: Parse $ARGUMENTS — `global` limits to `~/.claude/CLAUDE.md` only, `local` limits to `./CLAUDE.md` only, a number (e.g., `5`) caps conversation count, empty means full analysis (both files, last 20 conversations).
2. **Read CLAUDE.md files**: Read global (`~/.claude/CLAUDE.md`) and local (`./CLAUDE.md`) per scope. Fail fast if neither exists.
3. **Locate conversation history**: Derive the project folder — path is `~/.claude/projects/` with the project's absolute path slash-replaced by dashes (e.g., `-Users-me-project`). List `.jsonl` files sorted by recency. Fail fast if none found.
4. **Extract conversations**: For each of the N most recent `.jsonl` files, extract user/assistant turns using `jq` into readable text files in a temp directory. Skip empty assistant turns.
5. **Assess batch sizes**: List extracted files by size to plan agent batching.

Then orchestrate the review team:

1. **Create a team** with TeamCreate.
2. **Enter delegate mode** (Shift+Tab) — lead coordinates and synthesizes only, does not edit files.
3. **Batch conversations by size**: large (>100KB) 1-2 per agent, medium (10-100KB) 3-5 per agent, small (<10KB) 5-10 per agent.
4. **Create tasks** with TaskCreate — one per batch, each declaring:
   - The CLAUDE.md file contents (global and/or local per scope)
   - The conversation file paths to analyze
   - File ownership: **read-only for all analysts** (no edits)
   - Instructions: analyze conversations against the CLAUDE.md files and find (a) existing instructions that were violated, (b) patterns to add to LOCAL CLAUDE.md, (c) patterns to add to GLOBAL CLAUDE.md, (d) anything outdated or unnecessary. Be specific, output bullet points only.
5. **Spawn analyst agents** (Explore type, Sonnet model, read-only), assign tasks via TaskUpdate.
6. **Collect findings** from all analysts via TaskList, deduplicate, and merge into a single report with these sections:
   - **Instructions violated** — existing rules that weren't followed, need stronger wording
   - **Suggested additions — LOCAL** — project-specific patterns worth codifying
   - **Suggested additions — GLOBAL** — patterns that apply across all projects
   - **Potentially outdated** — items that may no longer be relevant
7. **Shut down team** after the report is assembled.

Present the report as severity-ranked bullet points. Ask the user whether to draft edits to the CLAUDE.md files: $ARGUMENTS
