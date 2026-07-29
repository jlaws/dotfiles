---
name: debugging-methodology
description: "Use when investigating bugs, failures, or odd behavior."
allowed-tools: Read, Grep, Glob, Bash
---

# Debugging Methodology

Find the cause before changing anything. A fix aimed at a symptom you have not explained either does
not hold or moves the bug somewhere else, and each speculative attempt makes the state harder to reason
about. Finish Phase 1 before proposing a fix.

Report what you found, where (`file:line`), and the fix — finding first, explanation after. Read the
relevant code before forming a theory about it, and when the cause is still unclear, say so rather than
offering a guess as an answer.

## Stop after two attempts

Two failed fixes at the same problem means the mental model is wrong, not that the next attempt needs to
be bolder:

| Attempts | Action |
|----------|--------|
| 1-2 | Normal Phase 3 (hypothesis + test) |
| 3 | STOP. Structured analysis before retrying |
| 3+ no progress | Ask user — root cause unclear or architectural change needed |

**Structured analysis before attempt 3:**
1. Write down what you tried and what happened
2. Articulate your assumptions — which one is wrong?
3. Identify information gaps
4. Re-read relevant code with fresh eyes
5. Return to Phase 1 with a new mental model

## The Four Phases

### Phase 1: Root Cause Investigation

**0. Classify the Failure Locus**

Before proposing any fix, decide where the fault actually lives:

| Locus | Meaning | Fix belongs in |
|-------|---------|----------------|
| Code-under-test (SUT) | The product code is wrong | Product code |
| Test harness | Setup, fixtures, or scaffolding is wrong | Test infrastructure |
| Checker / oracle | The assertion or expected value is wrong | The test's expectation |
| Environment | Config, deps, versions, network | Environment, not code |

A harness, checker, or environment fault is **never** fixed by changing product code. Misclassifying the locus is the most common cause of thrashing.

**1. Reproduce Consistently**
- Can you trigger it reliably? Exact steps? Minimal reproduction?
- If not reproducible, gather more data — don't guess
- For intermittent issues: add logging with timestamps, stress test, look for race conditions

**2. Check Recent Changes**
```bash
git log --oneline -20                    # Recent commits
git diff HEAD~5 -- src/                  # Recent code changes
git log --all --oneline -- <file>        # History of specific file
```

### Phase 2: Pattern Analysis

1. Find working examples of similar code
2. Compare against references — list every difference
3. Understand dependencies (settings, config, environment)

### Phase 3: Hypothesis and Testing

1. **Form Single Hypothesis**: "I think X is the root cause because Y"
2. **Test Minimally**: Smallest possible change, one variable at a time
3. **Verify**: Worked? → Phase 4. Didn't? → NEW hypothesis (don't add more fixes)

#### Structured Hypothesis Investigation

When the cause is unclear, enumerate 3-5 hypotheses up front (most likely first). Independent hypotheses that share no state may be investigated in parallel via subagents; otherwise investigate sequentially. Either way, keep the confirm/refute discipline for each hypothesis — do not blur evidence across them.

For each hypothesis, note:
- **What to check** — specific file, function, or state to inspect
- **Evidence that would confirm** — what you'd expect to see if this IS the cause
- **Evidence that would refute** — what you'd expect to see if this is NOT the cause

Investigate each in turn:
1. Gather evidence (read code, check logs, add instrumentation)
2. **Confirm or refute** — be explicit about which
3. If confirmed: proceed to Phase 4
4. If refuted: move to next hypothesis
5. If inconclusive: note what's missing and continue

**Stop as soon as you find the root cause.** Don't investigate remaining hypotheses.

#### Investigation Report Template

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

### Phase 4: Implementation

1. **Create Failing Test** — simplest possible reproduction, automated
2. **Implement Single Fix** — ONE change at a time
3. **Verify** — test passes? No other tests broken?

If the bug stems from invalid data flowing through multiple layers, read `references/defense-in-depth.md` to add validation at every layer and make it structurally impossible.

**If 3+ Fixes Failed**: see "Stop after two attempts" above — discuss fundamentals rather than retrying.

---

## Advanced Techniques

### Git Bisect
```bash
git bisect start
git bisect bad                    # Current is bad
git bisect good v1.0.0            # This was good
git bisect good   # or bad, repeat until found
git bisect reset
```

### Differential Debugging

| Aspect | Working | Broken |
|--------|---------|--------|
| Environment | Dev | Prod |
| Runtime version | 18.16.0 | 18.15.0 |
| Data | Empty DB | 1M records |
| Config | Default pool=5 | Custom pool=20 |

### Condition-Based Waiting & Test Pollution

If the failure is intermittent or involves async timing, read `references/condition-based-waiting.md` for wait strategies before writing polling loops or arbitrary sleeps.

If the root cause traces through multiple upstream callers and you cannot isolate it by reading call sites alone, read `references/root-cause-tracing.md` for a systematic tracing procedure.

---

## Never Mask Errors

| Masking Pattern | Do Instead |
|---|---|
| `catch (e) { /* ignore */ }` | Handle meaningfully or propagate |
| `if (x != null)` around internal logic | Fix why x is null |
| `@Disabled` / `skip()` on failing test | Fix the test or file tracked issue |
| Try-catch wrapping entire function | Catch specific exceptions at boundaries |
| Defensive null checks hiding broken contracts | Fix the broken contract upstream |

If unfixable now: log it, track it, surface it. Never silence it.
