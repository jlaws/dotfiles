---
name: team-investigate
description: "Competing hypothesis debugging — multiple agents investigate different theories in parallel. Use when debugging complex bugs where the root cause is unclear. Do NOT use for simple bugs (use /debug instead)."
argument-hint: "<bug-description>"
---

Bug description: $ARGUMENTS

## Competing Hypothesis Investigation

### Step 1: Formulate Hypotheses

From the bug description, formulate 2-4 competing hypotheses about the root cause. Each hypothesis should be:
- **Specific** — names a component, module, or mechanism
- **Testable** — can be confirmed or ruled out by reading code/logs
- **Independent** — investigating one doesn't require results from another

### Step 2: Dispatch Parallel Explore Subagents

For each hypothesis, dispatch a parallel Explore subagent:

```
Agent(
  description="Investigate hypothesis N: [summary]",
  prompt="Investigate whether [hypothesis]. Search for [specific code/logs/patterns]. Return: evidence for/against, confidence level (high/medium/low), and suggested next steps if confirmed.",
  subagent_type="Explore"
)
```

Each subagent should:
- Search for evidence supporting or refuting its assigned hypothesis
- Report findings with file:line references
- Rate confidence: high / medium / low
- Suggest what to investigate next if this hypothesis is correct

### Step 3: Synthesize Results

After all subagents return:
1. Compare evidence across hypotheses
2. Identify the most likely root cause (highest confidence with strongest evidence)
3. If multiple hypotheses have strong evidence, they may be related — look for a common cause
4. If no hypothesis has strong evidence, formulate new hypotheses based on findings

### Step 4: Report

Present findings as:

```markdown
## Investigation Results

### Most Likely Root Cause
[hypothesis] — confidence: [high/medium/low]
Evidence: [summary with file:line references]

### Other Hypotheses Investigated
| Hypothesis | Confidence | Key Evidence |
|-----------|-----------|-------------|
| ... | ... | ... |

### Recommended Fix
[approach based on root cause analysis]
```
