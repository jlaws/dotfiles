---
name: cmd-j-debug
description: "Systematic bug investigation — root cause analysis, not random fixes. Use when a test fails, a bug appears, or behavior is unexpected. Do NOT use for asking how to fix something (code directly instead)."
disable-model-invocation: true
---
# Debug

Before investigating, gather diagnostic context:

1. **Check git status** for uncommitted changes (note them for context).
2. **Detect test runner** from project config (package.json scripts, pytest.ini, Makefile, etc.).
3. **Check recent commits** (`git log --oneline -10`) for potential culprits.
4. **Capture failure output**: If the user's provided input references a test name, run it first to get the current failure output.

Now investigate using the methodology below: the user's provided input

---

## Debugging Methodology

### The Iron Law

```
NO FIXES WITHOUT ROOT CAUSE INVESTIGATION FIRST
```

Random fixes waste time and create new bugs. Complete Phase 1 before proposing fixes.

### The Two-Attempt Rule

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

### The Four Phases

#### Phase 1: Root Cause Investigation

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

#### Phase 2: Pattern Analysis

1. Find working examples of similar code
2. Compare against references — list every difference
3. Understand dependencies (settings, config, environment)

#### Phase 3: Hypothesis and Testing

1. **Form Single Hypothesis**: "I think X is the root cause because Y"
2. **Test Minimally**: Smallest possible change, one variable at a time
3. **Verify**: Worked? → Phase 4. Didn't? → NEW hypothesis (don't add more fixes)

#### Phase 4: Implementation

1. **Create Failing Test** — simplest possible reproduction, automated
2. **Implement Single Fix** — ONE change at a time
3. **Verify** — test passes? No other tests broken?

**If 3+ Fixes Failed**: See Two-Attempt Rule above. STOP and discuss fundamentals.

---

### Advanced Techniques

#### Git Bisect
```bash
git bisect start
git bisect bad                    # Current is bad
git bisect good v1.0.0            # This was good
git bisect good   # or bad, repeat until found
git bisect reset
```

#### Differential Debugging

| Aspect | Working | Broken |
|--------|---------|--------|
| Environment | Dev | Prod |
| Runtime version | 18.16.0 | 18.15.0 |
| Data | Empty DB | 1M records |
| Config | Default pool=5 | Custom pool=20 |

---

### Patterns by Issue Type

| Issue Type | Investigation Approach |
|-----------|----------------------|
| **Intermittent** | Add logging with timing, look for race conditions, stress test under load |
| **Performance** | Profile first — common culprits: N+1 queries, unnecessary re-renders, sync I/O in async paths |
| **Production-only** | Gather evidence (Sentry/logs/metrics), map local vs prod differences, reproduce under equivalent conditions |
| **Connection/resource exhaustion** | Monitor pool metrics, check for leaks in error paths, look for N+1 patterns, check if `finally`/`defer` cleanup runs |
| **Data-dependent** | Identify which data triggers it, find the minimum failing dataset, check for encoding/null/edge cases |

For language-specific debugging tools (breakpoints, profilers, stack traces), see the corresponding language reference files.

### Red Flags — STOP and Return to Phase 1

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

### Never Mask Errors

| Masking Pattern | Do Instead |
|---|---|
| `catch (e) { /* ignore */ }` | Handle meaningfully or propagate |
| `if (x != null)` around internal logic | Fix why x is null |
| `@Disabled` / `skip()` on failing test | Fix the test or file tracked issue |
| Try-catch wrapping entire function | Catch specific exceptions at boundaries |
| Defensive null checks hiding broken contracts | Fix the broken contract upstream |

If unfixable now: log it, track it, surface it. Never silence it.

### Quick Debugging Checklist

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

### Condition-Based Waiting

Flaky tests often guess at timing with arbitrary delays. This creates race conditions where tests pass on fast machines but fail under load or in CI.

**Core principle:** Wait for the actual condition you care about, not a guess about how long it takes.

**Use when:**
- Tests have arbitrary delays (`setTimeout`, `sleep`, `time.sleep()`)
- Tests are flaky (pass sometimes, fail under load)
- Tests timeout when run in parallel
- Waiting for async operations to complete

**Don't use when:**
- Testing actual timing behavior (debounce, throttle intervals)
- Always document WHY if using arbitrary timeout

#### Core Pattern

```typescript
// BAD: Guessing at timing
await new Promise(r => setTimeout(r, 50));
const result = getResult();
expect(result).toBeDefined();

// GOOD: Waiting for condition
await waitFor(() => getResult() !== undefined);
const result = getResult();
expect(result).toBeDefined();
```

#### Quick Patterns

| Scenario | Pattern |
|----------|---------|
| Wait for event | `waitFor(() => events.find(e => e.type === 'DONE'))` |
| Wait for state | `waitFor(() => machine.state === 'ready')` |
| Wait for count | `waitFor(() => items.length >= 5)` |
| Wait for file | `waitFor(() => fs.existsSync(path))` |
| Complex condition | `waitFor(() => obj.ready && obj.value > 10)` |

#### Implementation

Generic polling function:
```typescript
async function waitFor<T>(
  condition: () => T | undefined | null | false,
  description: string,
  timeoutMs = 5000
): Promise<T> {
  const startTime = Date.now();

  while (true) {
    const result = condition();
    if (result) return result;

    if (Date.now() - startTime > timeoutMs) {
      throw new Error(`Timeout waiting for ${description} after ${timeoutMs}ms`);
    }

    await new Promise(r => setTimeout(r, 10)); // Poll every 10ms
  }
}
```

#### Common Mistakes

- **Polling too fast:** `setTimeout(check, 1)` - wastes CPU. Fix: Poll every 10ms
- **No timeout:** Loop forever if condition never met. Fix: Always include timeout with clear error
- **Stale data:** Cache state before loop. Fix: Call getter inside loop for fresh data

#### When Arbitrary Timeout IS Correct

```typescript
// Tool ticks every 100ms - need 2 ticks to verify partial output
await waitForEvent(manager, 'TOOL_STARTED'); // First: wait for condition
await new Promise(r => setTimeout(r, 200));   // Then: wait for timed behavior
// 200ms = 2 ticks at 100ms intervals - documented and justified
```

Requirements: First wait for triggering condition, based on known timing (not guessing), comment explaining WHY.

### Root Cause Tracing

Bugs often manifest deep in the call stack. Your instinct is to fix where the error appears, but that's treating a symptom.

**Core principle:** Trace backward through the call chain until you find the original trigger, then fix at the source.

**Use when:**
- Error happens deep in execution (not at entry point)
- Stack trace shows long call chain
- Unclear where invalid data originated
- Need to find which test/code triggers the problem

#### The Tracing Process

1. **Observe the Symptom** — e.g., `Error: git init failed in /Users/jesse/project/packages/core`
2. **Find Immediate Cause** — What code directly causes this?
3. **Ask: What Called This?** — Trace the call chain upward
4. **Keep Tracing Up** — What value was passed? Where did it come from?
5. **Find Original Trigger** — e.g., variable accessed before initialization

#### Adding Stack Traces

When you can't trace manually, add instrumentation:

```typescript
async function gitInit(directory: string) {
  const stack = new Error().stack;
  console.error('DEBUG git init:', {
    directory,
    cwd: process.cwd(),
    nodeEnv: process.env.NODE_ENV,
    stack,
  });
  await execFileAsync('git', ['init'], { cwd: directory });
}
```

**Critical:** Use `console.error()` in tests (not logger - may not show)

#### Finding Which Test Causes Pollution

If something appears during tests but you don't know which test, use bisection: run tests one-by-one, stop at first polluter.

#### Key Principle

**NEVER fix just where the error appears.** Trace back to find the original trigger.

**Stack Trace Tips:**
- In tests: Use `console.error()` not logger
- Before operation: Log before the dangerous operation, not after it fails
- Include context: Directory, cwd, environment variables, timestamps
- Capture stack: `new Error().stack` shows complete call chain

### Defense-in-Depth Validation

When you fix a bug caused by invalid data, adding validation at one place feels sufficient. But that single check can be bypassed by different code paths, refactoring, or mocks.

**Core principle:** Validate at EVERY layer data passes through. Make the bug structurally impossible.

#### The Four Layers

**Layer 1: Entry Point Validation** — Reject obviously invalid input at API boundary
```typescript
function createProject(name: string, workingDirectory: string) {
  if (!workingDirectory || workingDirectory.trim() === '') {
    throw new Error('workingDirectory cannot be empty');
  }
  // ... proceed
}
```

**Layer 2: Business Logic Validation** — Ensure data makes sense for this operation
```typescript
function initializeWorkspace(projectDir: string, sessionId: string) {
  if (!projectDir) {
    throw new Error('projectDir required for workspace initialization');
  }
  // ... proceed
}
```

**Layer 3: Environment Guards** — Prevent dangerous operations in specific contexts
```typescript
async function gitInit(directory: string) {
  if (process.env.NODE_ENV === 'test') {
    const normalized = normalize(resolve(directory));
    const tmpDir = normalize(resolve(tmpdir()));
    if (!normalized.startsWith(tmpDir)) {
      throw new Error(
        `Refusing git init outside temp dir during tests: ${directory}`
      );
    }
  }
  // ... proceed
}
```

**Layer 4: Debug Instrumentation** — Capture context for forensics
```typescript
async function gitInit(directory: string) {
  const stack = new Error().stack;
  logger.debug('About to git init', { directory, cwd: process.cwd(), stack });
  // ... proceed
}
```

#### Applying the Pattern

When you find a bug:
1. **Trace the data flow** - Where does bad value originate? Where used?
2. **Map all checkpoints** - List every point data passes through
3. **Add validation at each layer** - Entry, business, environment, debug
4. **Test each layer** - Try to bypass layer 1, verify layer 2 catches it

**Don't stop at one validation point.** Add checks at every layer.

---

## Verification Before Completion

**Core principle:** Evidence before claims, always.

**Violating the letter of this rule is violating the spirit of this rule.**

### The Iron Law

```
NO COMPLETION CLAIMS WITHOUT FRESH VERIFICATION EVIDENCE
```

If you haven't run the verification command in this message, you cannot claim it passes.

### The Gate Function

```
BEFORE claiming any status:
1. IDENTIFY: What command proves this claim?
2. RUN: Execute the FULL command (fresh, complete)
3. READ: Full output, check exit code, count failures
4. VERIFY: Does output confirm the claim?
   - If NO: State actual status with evidence
   - If YES: State claim WITH evidence
5. ONLY THEN: Make the claim
```

### Common Failures

| Claim | Requires | Not Sufficient |
|-------|----------|----------------|
| Tests pass | Test output: 0 failures | Previous run, "should pass" |
| Linter clean | Linter output: 0 errors | Partial check, extrapolation |
| Build succeeds | Build: exit 0 | Linter passing |
| Bug fixed | Original symptom: passes | Code changed, assumed fixed |

### Red Flags - STOP

- Using "should", "probably", "seems to"
- Expressing satisfaction before verification
- About to commit/push/PR without verification
- **ANY wording implying success without having run verification**

### Key Patterns

```
Tests:     Run -> See "34/34 pass" -> THEN claim "All tests pass"
Red-Green: Write -> Run (pass) -> Revert -> Run (MUST FAIL) -> Restore -> Run (pass)
Build:     Run build -> See exit 0 -> THEN claim "Build passes"
```

**No shortcuts. Run the command. Read the output. THEN claim the result.**
