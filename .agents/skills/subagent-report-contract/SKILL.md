---
name: subagent-report-contract
description: "Use when returning findings as a dispatched agent."
---

# Agent Report Contract

**Core principle:** your report is injected into the caller's context verbatim, and it is the only
artifact — nothing is saved, nothing can be recovered later. So compress prose, never substance.

Answer in 2,000 tokens of narration and you spend 2,000 tokens of the caller's budget; a six-lens
fan-out spends that six times before the caller has read a line. But a report that drops a finding to
look short has destroyed information no one can get back.

## What to cut, and what to keep

| Cut | Keep |
|-----|------|
| Restating the question you were asked | Every finding, including minor ones |
| Narrating your search ("I looked at X, then Y") | A `file:line` for each finding |
| Hedging, preamble, and sign-off | Exact error strings, exit codes, quoted output |
| Generic advice that would fit any codebase | Why a finding matters, where it is not obvious |
| Repeating one point once per file it touches | Alternatives you considered and rejected, and why |

**There is no length budget.** When you are unsure whether a detail is relevant, include it — the
caller cannot ask a follow-up of a file that does not exist.

## Shape

```
<VERDICT LINE — your mode, severity totals, or one-line recommendation>
- <finding> (`path:line`) — <why it matters, when not obvious>
- <finding> (`path:line`)
<totals, when there are more than two findings>
```

Zero findings → `no findings — surface not present`. Do not manufacture material to fill the shape.

## Evidence is never compressed

Failures, exit codes, error strings, and quoted output are reproduced byte-for-byte, never paraphrased
or trimmed. Brevity applies to your prose, not to what you found.

## Integration

**Pairs with:** `dispatching-parallel-agents` (the caller's side)
**Uses:** `output-completeness` — a report with a dropped section costs the reader more than it saves
**See also:** `reference:context-efficiency` — Command Output Shaping, the same rule for tool output
