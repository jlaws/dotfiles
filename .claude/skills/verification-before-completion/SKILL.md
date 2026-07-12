---
name: verification-before-completion
description: "Use when about to claim work is complete, fixed, or passing, before committing or creating PRs — requires running verification commands and confirming output before making success claims. Do NOT use for test-first development workflow (use test-driven-development)."
compatibility: claude-code
allowed-tools: Read, Grep, Glob, Bash
---

# Verification Before Completion

**Core principle:** Evidence before claims, always.

**Violating the letter of this rule is violating the spirit of this rule.**

## The Iron Law

```
NO COMPLETION CLAIMS WITHOUT FRESH VERIFICATION EVIDENCE
```

If you haven't run the verification command in this message, you cannot claim it passes.

## The Gate Function

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

## Common Failures

| Claim | Requires | Not Sufficient |
|-------|----------|----------------|
| Tests pass | Test output: 0 failures | Previous run, "should pass" |
| Linter clean | Linter output: 0 errors | Partial check, extrapolation |
| Build succeeds | Build: exit 0 | Linter passing |
| Bug fixed | Original symptom: passes | Code changed, assumed fixed |
| Agent completed | VCS diff shows changes | Agent reports "success" |
| Requirements met | Line-by-line checklist | Tests passing |
| UI/web renders | Fresh page evidence: screenshot or asserted URL/title/DOM/a11y snapshot | "The code should render it" |

For UI/web changes, prefer an accessibility-tree snapshot over raw HTML — semantic and far cheaper in tokens.

## Green Is Not Enough: Why Did It Pass?

A passing check counts only if it exercised the target. Judge *why* it is green:

| Verdict | Meaning |
|---------|---------|
| PASS-hardening | The assertion ran the changed path and held |
| INCONCLUSIVE | Green, but the assertion never exercised the target (dead test, wrong path, mocked away) |

- A green result must cite the oracle output **and** evidence the target condition actually ran (log line, coverage, instrumented print).
- Never judge on pre-action evidence. Capture a **fresh post-action observation**; if it disagrees with the claim, report a **DEVIATION** — do not retro-fit the claim to stale state.
- Judge iteration by a mechanical metric (count, exit code, measured value), never "seems better".

## Verdict Grammar

Report review/verification status with one standard verdict plus severity:

| Verdict | Meaning |
|---------|---------|
| PASS | Verified, no blocking issues |
| CONCERNS | Works, but non-blocking issues found (list them) |
| FAIL | A defect is proven with evidence |
| BLOCKED | Could not verify — a coverage or tooling limit |

- Findings carry a priority: **P0** (must fix now) through **P3** (nice to have).
- **BLOCKED is a limitation of the check, never a product defect.** Do not downgrade BLOCKED into FAIL or PASS — state what you could not verify and why.

## Evidence Hierarchy

Weight evidence by strength; prefer the strongest available:

1. **Reproduced** — a deterministic run or observation of the behavior (strongest)
2. **Static-traced** — followed the code path by reading, no run
3. **Pattern-match** — "looks like" a known issue (weakest; verify before claiming)

## Read-Only Reviewer Contract

A review or audit agent MUST NOT mutate the code it inspects. Declare the boundary up front and keep findings evidence-only. Fixing is a separate, later step by a different actor.

## Red Flags - STOP

- Using "should", "probably", "seems to"
- Expressing satisfaction before verification
- About to commit/push/PR without verification
- Trusting agent success reports
- Relying on partial verification
- **ANY wording implying success without having run verification**

## Rationalization Prevention

| Excuse | Reality |
|--------|---------|
| "Should work now" | RUN the verification |
| "I'm confident" | Confidence is not evidence |
| "Just this once" | No exceptions |
| "Linter passed" | Linter is not compiler |
| "Agent said success" | Verify independently |
| "Partial check is enough" | Partial proves nothing |

## Key Patterns

```
Tests:     Run -> See "34/34 pass" -> THEN claim "All tests pass"
Red-Green: Write -> Run (pass) -> Revert -> Run (MUST FAIL) -> Restore -> Run (pass)
Build:     Run build -> See exit 0 -> THEN claim "Build passes"
Requirements: Re-read plan -> Checklist -> Verify each -> Report
Agent:     Agent reports -> Check VCS diff -> Verify changes -> Report actual state
```

## When To Apply

**ALWAYS before:** Any success/completion claim, any positive statement about work state, committing, PR creation, task completion, moving to next task, delegating to agents.

**No shortcuts. Run the command. Read the output. THEN claim the result.**
