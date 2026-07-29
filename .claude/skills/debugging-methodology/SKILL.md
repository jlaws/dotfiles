---
name: debugging-methodology
description: "Use when investigating bugs, failures, or odd behavior."
allowed-tools: Read, Grep, Glob, Bash
skills:
  - verification-before-completion
---

# Debugging Methodology

## The Iron Law

```
NO FIXES WITHOUT ROOT CAUSE INVESTIGATION FIRST
```

Random fixes waste time and create new bugs. Complete Phase 1 before proposing fixes.

## Debugging Directness

- Never speculate about a bug without reading the relevant code first.
- State what you found, where (file:line), and the fix. One pass.
- If the cause is unclear: say so explicitly. Do not guess.
- No preamble, no hedging. Finding first, explanation after.

## The Two-Attempt Rule

After 2 failed fix attempts at the same problem, **STOP**:

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

**1. Read Error Messages Carefully**
- Read stack traces completely; note line numbers, file paths, error codes
- The error message often tells you exactly what's wrong — don't skim

**2. Reproduce Consistently**
- Can you trigger it reliably? Exact steps? Minimal reproduction?
- If not reproducible, gather more data — don't guess
- For intermittent issues: add logging with timestamps, stress test, look for race conditions

**3. Check Recent Changes**
```bash
git log --oneline -20                    # Recent commits
git diff HEAD~5 -- src/                  # Recent code changes
git log --all --oneline -- <file>        # History of specific file
```

**4. Gather Evidence at Each Layer**

For multi-component systems, trace at every boundary:

```bash
# Trace data flow through each layer
echo "=== Layer: HTTP Request ==="    # What's coming in?
echo "=== Layer: Middleware ==="      # What transforms it?
echo "=== Layer: Business Logic ==="  # What processes it?
echo "=== Layer: Database ==="        # What gets persisted?
echo "=== Layer: Response ==="        # What goes out?
```

**5. Trace Data Flow** — Where does the bad value originate? Keep tracing upstream until you find the source.

**6. Ask "Why Does It Work Locally?"**

When bugs appear in production/CI but not locally:

| Factor | Local | Production |
|--------|-------|------------|
| Concurrency | Single user, sequential | Many users, concurrent |
| Data volume | Seed data, small | Real data, large |
| Configuration | Dev defaults | Production settings (pool sizes, timeouts, caches) |
| Network | localhost, fast | Real latency, DNS, proxies |
| Dependencies | Mocked or local | Real services, rate limits |
| Timing | Debugger pauses, slow | Full speed, race conditions |

Map every environmental difference. The bug lives in one of these gaps.

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

**If 3+ Fixes Failed**: See Two-Attempt Rule above. STOP and discuss fundamentals.

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

## Patterns by Issue Type

| Issue Type | Investigation Approach |
|-----------|----------------------|
| **Intermittent** | Add logging with timing, look for race conditions, stress test under load |
| **Performance** | Profile first — common culprits: N+1 queries, unnecessary re-renders, sync I/O in async paths |
| **Production-only** | Gather evidence (Sentry/logs/metrics), map local vs prod differences, reproduce under equivalent conditions |
| **Connection/resource exhaustion** | Monitor pool metrics, check for leaks in error paths, look for N+1 patterns, check if `finally`/`defer` cleanup runs |
| **Data-dependent** | Identify which data triggers it, find the minimum failing dataset, check for encoding/null/edge cases |

For language-specific debugging tools (breakpoints, profilers, stack traces), see the corresponding language reference files.

## Red Flags — STOP and Return to Phase 1

- "Quick fix for now, investigate later"
- "Just try changing X and see"
- "I don't fully understand but this might work"
- "One more fix attempt" (when already tried 2+)

| Excuse | Reality |
|--------|---------|
| "Issue is simple" | Simple issues have root causes too |
| "Emergency, no time" | Systematic is FASTER than thrashing |
| "Multiple fixes saves time" | Can't isolate what worked |
| "I see the problem" | Seeing symptoms ≠ understanding root cause |
| "Just increase the pool size" | Treating symptoms hides the leak |

## Never Mask Errors

| Masking Pattern | Do Instead |
|---|---|
| `catch (e) { /* ignore */ }` | Handle meaningfully or propagate |
| `if (x != null)` around internal logic | Fix why x is null |
| `@Disabled` / `skip()` on failing test | Fix the test or file tracked issue |
| Try-catch wrapping entire function | Catch specific exceptions at boundaries |
| Defensive null checks hiding broken contracts | Fix the broken contract upstream |

If unfixable now: log it, track it, surface it. Never silence it.

## Quick Debugging Checklist

- [ ] Spelling errors / typos
- [ ] Case sensitivity
- [ ] Null/undefined values
- [ ] Off-by-one errors
- [ ] Async timing / race conditions
- [ ] Scope issues / type mismatches
- [ ] Missing dependencies / env vars
- [ ] Cache / stale state
- [ ] Error path cleanup (connections, file handles, locks)
- [ ] Environment differences (local vs CI vs prod)
