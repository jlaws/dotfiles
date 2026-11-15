---
name: session-handoff
description: "Use when preserving work for another session or agent."
---

# Session Handoff

Create a HANDOFF.md that lets the next session (or a different agent) resume work without losing context.

## When to Create

- Session ending with incomplete work
- Context window pressure (approaching compaction)
- Switching to a different task temporarily
- Handing off to a teammate or different agent
- Before a planned break in multi-session work

## HANDOFF.md Template

```markdown
# Handoff: {Task Title}

**Created:** {date}
**Branch:** {branch name}
**Status:** {In Progress | Blocked | Ready for Review}

## Summary
{2-3 sentences: what was being done and current state}

## Key Decisions
- {Decision 1}: {rationale}
- {Decision 2}: {rationale}

## Rejected Approaches
- {approach tried and abandoned}: {why it failed or was not pursued}

## Files Modified
| File | Change | Status |
|------|--------|--------|
| `path/to/file` | {what changed} | {done / partial / needs review} |

## Test Results
- Last run: `{test command}` → {pass/fail, count}
- Failing: {list specific failures if any}

## Open Issues
- [ ] {Unresolved problem or question}
- [ ] {Known bug introduced but not yet fixed}

## Next Steps
1. {First thing to do when resuming}
2. {Second priority}
3. {Third priority}

## Context (for resuming agent)
- {Any non-obvious state: environment setup, feature flags, config changes}
- {Dependencies or blockers from external systems}
```

## Location

- Default: project root (`./HANDOFF.md`)
- Multiple handoffs: `./handoffs/{date}-{task-slug}.md`
- Gitignore if sensitive: add to `.gitignore`

## Resume Protocol

When resuming from a handoff:

1. Read HANDOFF.md
2. Verify file state matches "Files Modified" table
3. Run tests listed in "Test Results" to confirm current state
4. Check "Open Issues" for unresolved blockers
5. Start from "Next Steps" item 1

## Examples

**Trigger:** "I'm done for the day, save my progress"
**Action:** Summarize current state, pending tasks, and decisions into HANDOFF.md
**Result:** Next session loads HANDOFF.md and continues seamlessly from where you left off

## Rules

- Write the handoff BEFORE context is lost — don't wait until the last message
- Include exact file paths (not relative descriptions)
- Include exact test commands (not "run the tests")
- List decisions with rationale — the "why" is harder to reconstruct than the "what"
- Keep it under 100 lines — this is a resume document, not documentation
- Write/update at context-fill milestones (e.g. 50%, 75%), not only at the limit
- After a compaction, emit a short re-orientation digest (task + current file + next step) before continuing
- Delete the handoff after successful resumption
