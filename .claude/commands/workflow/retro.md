---
name: retro
description: "Engineering retrospective from git history — metrics, patterns, and insights for a time window. Use when reviewing what shipped, what blocked, and high-churn areas. Do NOT use for team retrospectives (single-person focused)."
argument-hint: "<time-window, e.g. '14d', '30d', 'since v1.2.0'>"
---

Time window: $ARGUMENTS (default: 14 days)

Load skill `analysis-output-patterns` for output structure rules.

---

## Engineering Retrospective

### Step 1 — Determine Time Window
- Parse arguments into `--since` range (support `14d`, `30d`, `since v1.2.0`)
- Calculate "previous period" of same length for comparison

### Step 2 — Gather Metrics (both periods)

| Metric | Command |
|--------|---------|
| Commits | `git log --oneline --since/--until` count |
| Files changed | `git log --name-only` unique files |
| Lines added/removed | `git log --numstat` sums |
| Merge commits | `git log --merges` count |
| Active hours | Commit timestamp spread |

### Step 3 — Identify Patterns

**What Shipped** — Group commits by theme (feature, fix, refactor, docs, chore) from message prefixes or conventional commits. List major changes with brief descriptions.

**What Blocked** — Identify:
- Reverted commits (`git log --grep="revert"`)
- High-churn files (changed >3 times in window) — potential instability
- Long-lived branches that didn't merge

**High-Churn Areas** — Top 10 most-modified files. Flag files with both significant additions and deletions (refactoring signals).

### Step 4 — Period Comparison

Compare current vs previous period: commits (up/down/flat), code churn (growing/shrinking), churn concentration (spreading or focusing).

### Step 5 — Present Report

```markdown
## Retro — {start_date} to {end_date}

### Metrics
| Metric | This Period | Previous Period | Trend |
|--------|------------|----------------|-------|
| Commits | N | N | ↑/↓/→ |
| Files touched | N | N | ↑/↓/→ |
| Lines added | N | N | ↑/↓/→ |
| Lines removed | N | N | ↑/↓/→ |

### What Shipped
- {grouped by theme}

### High-Churn Files
| File | Changes | Net Lines |
|------|---------|-----------|

### Observations
- {patterns, blockers, insights}
```

### Rules
- Single-person focused — no team breakdowns
- Facts from git, not speculation
- Flag high-churn files as **potential** instability, not definitive problems
- Keep report concise — summary, not exhaustive log
