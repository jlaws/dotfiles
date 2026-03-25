---
name: team-investigate
description: "Systematic hypothesis debugging — investigate root causes one theory at a time. Use when debugging complex bugs where the root cause is unclear. Do NOT use for simple bugs (use /debug instead)."
argument-hint: "<bug-description>"
---

Bug description: $ARGUMENTS

**Do NOT use subagents or parallel agents. Process all hypotheses linearly.**

## Hypothesis-Driven Investigation

### Step 1: Reproduce & Characterize
1. Reproduce the bug — get the exact error message, stack trace, or unexpected behavior
2. Note the conditions: inputs, environment, timing, frequency
3. Identify the boundary: what works vs what doesn't

### Step 2: Generate Hypotheses
List 3-5 possible root causes, ranked by likelihood:
1. Most likely cause (based on error message, stack trace, recent changes)
2. Second most likely
3. ...

For each hypothesis, note:
- **What to check** — specific file, function, or state to inspect
- **Evidence that would confirm** — what you'd expect to see if this is the cause
- **Evidence that would refute** — what you'd expect to see if this is NOT the cause

### Step 3: Investigate Sequentially
For each hypothesis (most likely first):
1. Gather evidence (read code, check logs, add instrumentation)
2. **Confirm or refute** — be explicit about which
3. If confirmed: proceed to fix
4. If refuted: move to next hypothesis
5. If inconclusive: note what's missing and continue

**Stop as soon as you find the root cause.** Don't investigate remaining hypotheses.

### Step 4: Fix & Verify
1. Implement the fix
2. Verify the original bug is resolved
3. Check for regressions
4. Commit with a message explaining the root cause

### Step 5: Report
```markdown
## Investigation: {bug description}

### Root Cause
{explanation}

### Hypotheses Tested
1. {hypothesis} — {confirmed/refuted} — {evidence}
2. ...

### Fix
{what was changed and why}

### Verification
{how the fix was verified}
```
