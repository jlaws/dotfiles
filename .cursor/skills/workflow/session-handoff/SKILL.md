---
name: session-handoff
description: "Create structured HANDOFF.md documents preserving decisions, file paths, test results, and next steps between sessions. Use when saving session progress, creating handoff notes, preparing context for next session, ending a work session, switching context, or hitting context limits. Do NOT use for general documentation (use technical-writing-for-devtools). Do NOT use for multi-agent task handoff (use multi-agent-development)."
---

# Session Handoff

Create a HANDOFF.md that lets the next session (or a different agent) resume work without losing context.

## When to Create

- Session ending with incomplete work
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
- Delete the handoff after successful resumption
