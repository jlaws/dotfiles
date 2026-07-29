---
name: verification-before-completion
description: "Use when verifying evidence before claiming work complete."
allowed-tools: Read, Grep, Glob, Bash
---

# Weighing and Reporting Evidence

Shared vocabulary for review and audit output. Use it so a verdict means the same thing everywhere.

This is about how to *weigh and report* what you found. It is not a checklist for re-running your own
work — Claude verifies its own work natively, and adding a separate verification pass on top wastes
tokens without improving the result.

## Green is not enough: ask why it passed

A passing check counts only if it exercised the target.

| Verdict | Meaning |
|---------|---------|
| PASS-hardening | The assertion ran the changed path and held |
| INCONCLUSIVE | Green, but the assertion never touched the target — dead test, wrong path, mocked away |

A green result is worth citing only alongside evidence the target condition actually ran: a log line,
coverage, an instrumented print. Judge iteration by a mechanical metric — a count, an exit code, a
measured value — not "seems better".

Observations made before the change do not support a claim about the state after it. If a fresh
observation disagrees with what you expected, report the deviation rather than fitting the claim to
stale state.

## Verdict grammar

| Verdict | Meaning |
|---------|---------|
| PASS | Verified, no blocking issues |
| CONCERNS | Works, but non-blocking issues found — list them |
| FAIL | A defect is proven with evidence |
| BLOCKED | Could not verify, because of a coverage or tooling limit |

Findings carry a priority from **P0** (must fix now) to **P3** (nice to have).

**BLOCKED describes the check, not the product.** It means you could not look, so it cannot be
downgraded into FAIL or PASS. Say what you could not verify and why.

## Evidence hierarchy

Prefer the strongest available, and say which one you have:

1. **Reproduced** — a deterministic run or direct observation of the behavior
2. **Static-traced** — followed the code path by reading, without running it
3. **Pattern-match** — resembles a known issue; the weakest, so confirm before asserting it

Cite a concrete `file:line` for every finding.

## Read-only reviewer contract

A review or audit agent does not mutate the code it inspects. Declare that boundary up front and keep
findings evidence-only. Fixing is a separate step by a different actor — mixing them means the review
loses its independence and the diff no longer shows what was wrong.

## What counts as proof of a claim

| Claim | What supports it |
|-------|------------------|
| Tests pass | Test output showing zero failures |
| Build succeeds | Exit 0 from the build, not from the linter |
| Bug fixed | The original symptom, re-observed |
| Delegated work done | The VCS diff, not the agent's summary |
| Requirements met | The requirements, checked one by one |
| UI renders | A fresh accessibility-tree snapshot — semantic and far cheaper than raw HTML |
| Docs current | Docs updated, or an explicit N/A with a reason (see `documentation-validation`) |
