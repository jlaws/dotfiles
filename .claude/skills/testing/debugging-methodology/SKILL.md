---
name: debugging-methodology
description: Master systematic debugging with proven techniques, profiling tools, and disciplined root cause analysis. Use when investigating bugs, test failures, performance issues, or unexpected behavior.
---

# Debugging Methodology

Transform debugging from frustrating guesswork into systematic problem-solving with proven strategies, powerful tools, and methodical approaches.

## The Iron Law

```
NO FIXES WITHOUT ROOT CAUSE INVESTIGATION FIRST
```

Random fixes waste time and create new bugs. If you haven't completed Phase 1, you cannot propose fixes.

## When to Use This Skill

**Use for ANY technical issue:**
- Test failures
- Bugs in production
- Unexpected behavior
- Performance problems
- Build failures
- Integration issues
- Memory leaks
- Distributed systems issues

**Use ESPECIALLY when:**
- Under time pressure (emergencies make guessing tempting)
- "Just one quick fix" seems obvious
- You've already tried multiple fixes
- You don't fully understand the issue

---

## The Four Phases

You MUST complete each phase before proceeding to the next.

### Phase 1: Root Cause Investigation

**BEFORE attempting ANY fix:**

**1. Read Error Messages Carefully**
- Don't skip past errors or warnings
- Read stack traces completely
- Note line numbers, file paths, error codes
- They often contain the exact solution

**2. Reproduce Consistently**
- Can you trigger it reliably?
- What are the exact steps?
- Create minimal reproduction
- If not reproducible → gather more data, don't guess

**3. Check Recent Changes**
- Git diff, recent commits
- New dependencies, config changes
- Environmental differences
- What changed that could cause this?

**4. Gather Evidence (Multi-Component Systems)**

For EACH component boundary:
```bash
# Layer 1: Workflow
echo "=== Secrets available in workflow: ==="
echo "IDENTITY: ${IDENTITY:+SET}${IDENTITY:-UNSET}"

# Layer 2: Build script
echo "=== Env vars in build script: ==="
env | grep IDENTITY || echo "IDENTITY not in environment"

# Layer 3: Actual operation
echo "=== State before operation: ==="
# Log what data enters and exits each component
```

**5. Trace Data Flow**
- Where does bad value originate?
- What called this with bad value?
- Keep tracing up until you find the source
- Fix at source, not at symptom

### Phase 2: Pattern Analysis

**Find the pattern before fixing:**

1. **Find Working Examples** - Locate similar working code
2. **Compare Against References** - Read reference implementations completely
3. **Identify Differences** - List every difference, however small
4. **Understand Dependencies** - What settings, config, environment needed?

### Phase 3: Hypothesis and Testing

**Scientific method:**

1. **Form Single Hypothesis**
   - State clearly: "I think X is the root cause because Y"
   - Write it down, be specific

2. **Test Minimally**
   - SMALLEST possible change to test hypothesis
   - One variable at a time
   - Don't fix multiple things at once

3. **Verify Before Continuing**
   - Did it work? Yes → Phase 4
   - Didn't work? Form NEW hypothesis
   - DON'T add more fixes on top

### Phase 4: Implementation

**Fix the root cause, not the symptom:**

1. **Create Failing Test Case** - Simplest possible reproduction, automated
2. **Implement Single Fix** - ONE change at a time, no bundled refactoring
3. **Verify Fix** - Test passes? No other tests broken?

**If 3+ Fixes Failed: Question Architecture**
- Each fix reveals new problems in different places = architectural problem
- STOP and question fundamentals
- Don't attempt Fix #4 without discussion

---

## Debugging Tools by Language

### JavaScript/TypeScript

```typescript
// Chrome DevTools Debugger
function processOrder(order: Order) {
    debugger;  // Execution pauses here

    // Conditional breakpoint
    if (order.items.length > 10) {
        debugger;  // Only breaks if condition true
    }
}

// Console debugging techniques
console.log('Value:', value);                    // Basic
console.table(arrayOfObjects);                   // Table format
console.time('op'); /* code */ console.timeEnd('op');  // Timing
console.trace();                                 // Stack trace
console.assert(value > 0, 'Must be positive');  // Assertion

// Performance profiling
performance.mark('start-operation');
// ... operation code
performance.mark('end-operation');
performance.measure('operation', 'start-operation', 'end-operation');
```

**VS Code Debugger:**
```json
{
    "type": "node",
    "request": "launch",
    "name": "Debug Program",
    "program": "${workspaceFolder}/src/index.ts",
    "outFiles": ["${workspaceFolder}/dist/**/*.js"],
    "skipFiles": ["<node_internals>/**"]
}
```

### Python

```python
# Built-in debugger (pdb)
import pdb
pdb.set_trace()  # Debugger starts here

# Breakpoint (Python 3.7+)
breakpoint()  # More convenient

# Post-mortem debugging
try:
    risky_operation()
except Exception:
    import pdb
    pdb.post_mortem()  # Debug at exception point

# Logging for debugging
import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
logger.debug(f'Fetching user: {user_id}')

# Profile performance
import cProfile
import pstats
cProfile.run('slow_function()', 'profile_stats')
stats = pstats.Stats('profile_stats')
stats.sort_stats('cumulative').print_stats(10)
```

### Go

```go
// Delve debugger: dlv debug main.go

import (
    "runtime/debug"
)

// Print stack trace
debug.PrintStack()

// Panic recovery with debugging
defer func() {
    if r := recover(); r != nil {
        fmt.Println("Panic:", r)
        debug.PrintStack()
    }
}()

// CPU profiling
import "runtime/pprof"
f, _ := os.Create("cpu.prof")
pprof.StartCPUProfile(f)
defer pprof.StopCPUProfile()
```

---

## Advanced Techniques

### Binary Search Debugging (Git Bisect)

```bash
git bisect start
git bisect bad                    # Current commit is bad
git bisect good v1.0.0            # v1.0.0 was good
# Git checks out middle commit, test it, then:
git bisect good   # if it works
git bisect bad    # if it's broken
git bisect reset  # when done
```

### Differential Debugging

| Aspect       | Working         | Broken          |
|--------------|-----------------|-----------------|
| Environment  | Development     | Production      |
| Node version | 18.16.0         | 18.15.0         |
| Data         | Empty DB        | 1M records      |
| Time         | During day      | After midnight  |

### Memory Leak Detection

```typescript
// Chrome DevTools: Take heap snapshots, compare

// Node.js
if (process.memoryUsage().heapUsed > 500 * 1024 * 1024) {
    require('v8').writeHeapSnapshot();
}
```

---

## Patterns by Issue Type

### Intermittent Bugs

1. Add extensive logging with timing
2. Look for race conditions
3. Check timing dependencies (setTimeout, Promise order)
4. Stress test - run many times, vary timing

### Performance Issues

1. Profile first - don't optimize blindly
2. Common culprits: N+1 queries, unnecessary re-renders, synchronous I/O
3. Tools: DevTools Performance, Lighthouse, cProfile, clinic.js

### Production Bugs

1. Gather evidence (Sentry, logs, metrics)
2. Reproduce locally with production data (anonymized)
3. Safe investigation - don't change production
4. Test fixes in staging first

---

## Red Flags - STOP and Follow Process

If you catch yourself thinking:
- "Quick fix for now, investigate later"
- "Just try changing X and see"
- "Skip the test, I'll manually verify"
- "I don't fully understand but this might work"
- "One more fix attempt" (when already tried 2+)

**ALL mean: STOP. Return to Phase 1.**

## Common Rationalizations

| Excuse | Reality |
|--------|---------|
| "Issue is simple" | Simple issues have root causes too |
| "Emergency, no time" | Systematic is FASTER than thrashing |
| "Multiple fixes saves time" | Can't isolate what worked |
| "I see the problem" | Seeing symptoms ≠ understanding root cause |

---

## Quick Debugging Checklist

When stuck, check:
- [ ] Spelling errors (typos in variable names)
- [ ] Case sensitivity
- [ ] Null/undefined values
- [ ] Array index off-by-one
- [ ] Async timing (race conditions)
- [ ] Scope issues
- [ ] Type mismatches
- [ ] Missing dependencies
- [ ] Environment variables
- [ ] Cache issues

## Best Practices

1. **Reproduce First** - Can't fix what you can't reproduce
2. **Isolate the Problem** - Remove complexity until minimal case
3. **Read Error Messages** - They're usually helpful
4. **Check Recent Changes** - Most bugs are recent
5. **Use Version Control** - Git bisect, blame, history
6. **Take Breaks** - Fresh eyes see better
7. **Document Findings** - Help future you
8. **Fix Root Cause** - Not just symptoms

## Real-World Impact

- Systematic approach: 15-30 minutes to fix
- Random fixes approach: 2-3 hours of thrashing
- First-time fix rate: 95% vs 40%
- New bugs introduced: Near zero vs common
