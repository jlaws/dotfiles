---
name: review-claudemd
description: "Analyze recent conversation history to improve CLAUDE.md files — find violated instructions, missing patterns, and outdated rules. Use when tuning Claude Code behavior or after a batch of sessions. Do NOT use for quick questions (edit files directly instead)."
argument-hint: "<scope: global|local|number|empty>"
---

Scope: $ARGUMENTS

If no arguments provided, analyzes both global and local CLAUDE.md against last 20 conversations.

## CLAUDE.md Review Process

### Step 1: Gather Context

1. Read the CLAUDE.md file(s) in scope (global `~/.claude/CLAUDE.md` and/or local `CLAUDE.md`)
2. Identify all rules, conventions, and behavioral instructions

### Step 2: Dispatch Parallel Explore Subagents

Dispatch parallel Explore subagents to analyze conversation history from different angles:

1. **violation-detector** (Explore) — Find conversations where CLAUDE.md instructions were violated. Report which rules were broken and how.
2. **pattern-detector** (Explore) — Find recurring corrections or guidance the user had to give repeatedly. These are candidates for new CLAUDE.md rules.
3. **staleness-detector** (Explore) — Compare CLAUDE.md rules against current codebase state. Find rules that reference deleted files, renamed functions, or outdated patterns.

### Step 3: Synthesize Findings

After all subagents return, produce:

```markdown
## CLAUDE.md Review — {scope}

### Violated Rules
- {rule} — violated in {N} conversations — {pattern of violation}

### Missing Rules (Repeated Corrections)
- {correction pattern} — given {N} times — suggested rule: {draft}

### Stale Rules
- {rule} — {reason it's outdated}

### Recommended Changes
1. {specific edit to CLAUDE.md with rationale}
```

### Step 4: Decision Gate

Present findings and ask:
1. Apply recommended changes
2. Review complete — no changes
