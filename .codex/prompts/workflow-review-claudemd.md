---
name: workflow-review-claudemd
description: "Analyze recent conversation history to improve CLAUDE.md files — find violated instructions, missing patterns, and outdated rules. Use when tuning Claude Code behavior or after a batch of sessions. Do NOT use for quick questions (edit files directly instead)."
argument-hint: "<scope: global|local|number|empty>"
---

Scope: $ARGUMENTS

If no arguments provided, analyzes both global and local CLAUDE.md against last 20 conversations.

**Do NOT use subagents or parallel agents. Process all analysis linearly.**

For implementation and review checklists, see `references/workflow/task-execution-checklists`.

## Review Process

### Step 1: Gather Conversation History
1. Read recent conversation logs (use `$ARGUMENTS` to determine scope — global, local, or N conversations)
2. Identify conversations where Claude Code behavior was suboptimal, instructions were violated, or the user had to correct course

### Step 2: Analyze Current CLAUDE.md
1. Read the current CLAUDE.md file(s) in scope
2. For each conversation issue found:
   - Was there an existing instruction that was violated? → Flag as enforcement gap
   - Was the behavior correct but not documented? → Flag as missing pattern
   - Is there an outdated rule that caused bad behavior? → Flag as stale rule

### Step 3: Multi-Perspective Review
Analyze sequentially from each perspective:

**3.1 Instruction Compliance**
- Which CLAUDE.md rules were violated in recent conversations?
- Are instructions clear enough to follow, or ambiguous?
- Are there conflicting instructions?

**3.2 Missing Patterns**
- What corrections did the user make that aren't captured in CLAUDE.md?
- What successful behaviors should be codified?
- What edge cases caused confusion?

**3.3 Staleness Audit**
- Which rules reference tools, files, or patterns that no longer exist?
- Which rules are redundant with default behavior?
- Which rules are too specific (one-time fix) vs durable?

### Step 4: Produce Recommendations
```markdown
## CLAUDE.md Review — {scope}

### Violations Found
- {rule} — violated in {N} conversations — {root cause}

### Missing Patterns
- {pattern} — observed in {context} — suggested rule

### Stale Rules
- {rule} — reason it's stale — suggested action (remove/update)

### Proposed Changes
1. {specific edit with file:line reference}
```

### Step 5: Implementation
- Present findings to user
- Ask which changes to implement
- Make approved edits to CLAUDE.md file(s)

---

### Cross-References

- **skill:code-agent-meta-patterns** — CLAUDE.md design patterns, context budget rules
- **reference:workflow/context-efficiency** — context management patterns
- **reference:workflow/task-execution-checklists** — implementation and review checklists
- **skill:verification-before-completion** — evidence-before-claims methodology
- **skill:writing-skills** — skill creation and testing methodology
