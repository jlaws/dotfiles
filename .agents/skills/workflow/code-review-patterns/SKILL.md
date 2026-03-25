---
name: code-review-patterns
description: Use when reviewing PRs, responding to review feedback, or running a self-review. Do NOT use for general code smell detection outside PR context (use code-quality).
---

# Code Review Patterns

## Review Mindset

**Goals:** Catch bugs/edge cases, ensure maintainability, share knowledge, enforce standards, improve design.

**Not goals:** Show off knowledge, nitpick formatting (use linters), block progress unnecessarily, rewrite to preference.

---

## Giving Reviews

### Review Process (time-boxed)

#### Phase 1: Context (2-3 min)
1. Read PR description and linked issue
2. Check PR size (>400 lines? Ask to split)
3. CI/CD status passing?
4. Understand the business requirement

#### Phase 2: High-Level (5-10 min)
- Does solution fit the problem? Simpler approaches?
- Consistent with existing patterns? Will it scale?
- Are there tests? Do they cover edge cases?

#### Phase 3: Line-by-Line (10-20 min)
- **Logic**: Edge cases, off-by-one, null checks, race conditions
- **Security**: Input validation, SQL injection, XSS, data exposure
- **Performance**: N+1 queries, unnecessary loops, memory leaks, blocking ops
- **Maintainability**: Clear names, SRP functions, magic numbers extracted

#### Phase 4: Summary (2-3 min)
1. Summarize key concerns
2. Highlight what worked well
3. Clear decision: Approve / Comment / Request Changes
4. Offer to pair if complex

### Feedback Severity Labels

```
[blocking]    - Must fix before merge
[important]   - Should fix, discuss if disagree
[nit]         - Nice to have, not blocking
[suggestion]  - Alternative approach to consider
[learning]    - Educational, no action needed
```

### Feedback Techniques

- **Ask questions** instead of stating problems: "What happens if `items` is empty?"
- **Suggest, don't command**: "Would it make sense to extract this? It appears in 3 places."
- **Be specific and actionable**: "Race condition when concurrent access — consider a mutex here."

### Handling Disagreements

1. **Seek understanding**: "What led you to choose this pattern?"
2. **Acknowledge valid points**: "That's a fair consideration about X."
3. **Provide data**: "Can we add a benchmark to validate?"
4. **Escalate if needed**: Get architect/senior to weigh in
5. **Let go if non-critical**: Perfection is the enemy of progress

### Common Pitfalls

| Anti-Pattern | Problem | Fix |
|---|---|---|
| **Perfectionism** | Blocking for style preferences | Use linters; only block for correctness |
| **Scope creep** | "While you're at it..." | File separate issues for unrelated improvements |
| **Delayed reviews** | PRs stale for days | Review within 4 business hours |
| **Rubber stamping** | Approving without reading | Use the phased review process above |
| **Bike shedding** | Debating trivial choices at length | Time-box; default to author's preference |
| **Drip-feed comments** | Incomplete feedback across multiple rounds | Give complete feedback in one round; forces thorough upfront analysis |
| **Gatekeeper framing** | Reviews as compliance gates | Frame as knowledge transfer — every comment should teach something. Mentor, not gatekeeper |

---

## Pre-Submission Diff Review

Self-review workflow for current branch changes vs main. Catch issues before reviewers do.

**Do NOT use `gh` or any GitHub CLI commands. All information must come from local git.**

### Step 1 — Identify Changes

```bash
git diff main...HEAD
git log main..HEAD --oneline
```

FAIL FAST if current branch IS main or has no commits ahead of main.

### Step 2 — Gather Full Context

```bash
git diff main...HEAD --name-only
```

**Read every changed file in full** — not just diff hunks. Context beyond changed lines catches cross-cutting issues.

### Step 3 — Detect Review Scope

Inspect file extensions. Load matching language skills:

| Extension | Skill |
|-----------|-------|
| `.py` | `languages:python-patterns` |
| `.js`, `.ts`, `.tsx` | `languages:js-ts-patterns` |
| `.go` | `languages:go-concurrency-patterns` |
| `.sh` | `languages:bash-defensive-patterns` |
| `.swift` | `languages:swift-patterns` |
| `.rs` | `languages:rust-project-patterns` |

Flag missing tests if diff modifies source but includes no test changes.

### Step 4 — Multi-Perspective Analysis

Analyze the diff from each perspective independently, then merge findings.

**4.1 Code Review** — Edge cases, error handling, logic errors, missing validation (Phase 3 checklist above).

**4.2 Code Quality** — Cross-reference `workflow:code-quality`: smells, naming, DRY violations, unnecessary complexity, coupling.

**4.3 Security** — Cross-reference `security:security-analysis`: STRIDE threats, injection, XSS, SSRF, path traversal, auth gaps, secrets in code, insecure defaults.

**4.4 Testing** — Cross-reference `testing:language-testing-patterns`: coverage gaps, test quality (behavior vs implementation), missing integration tests, flaky indicators.

**4.5 Language-Specific Gotchas** — Apply auto-detected `languages:*-patterns` skills. See [references/language-gotchas.md](references/language-gotchas.md).

### Step 5 — Structured Findings Report

```markdown
## Diff Review — {BRANCH_NAME}

### Critical
- {finding} — {file:line} — {perspective}

### High
- {finding} — {file:line} — {perspective}

### Medium
- {finding} — {file:line} — {perspective}

### Test Gaps
- {description of missing coverage}

### What Looks Good
- {positive observation}
```

Omit empty severity sections. Always include "What Looks Good".

### Step 6 — Decision Gate

**Default: report only.** Do NOT automatically implement fixes.

After presenting findings, ask:
1. Implement fixes for findings above
2. Nothing — review complete

If implementing, follow `workflow:pr-comment-resolution` Step 4 (scope guard, atomic commits, verify before push).

---

## Receiving & Responding to Reviews

### Response Pattern

```
1. READ: Complete feedback without reacting
2. UNDERSTAND: Restate requirement in own words (or ask)
3. VERIFY: Check against codebase reality
4. EVALUATE: Technically sound for THIS codebase?
5. RESPOND: Technical acknowledgment or reasoned pushback
6. IMPLEMENT: One item at a time, test each
```

### Handling Unclear Feedback

If ANY item is unclear, stop. Do not implement partially.

```
Understand 1,2,3,6. Unclear on 4,5.
RIGHT: "Understand 1,2,3,6. Need clarification on 4 and 5 before proceeding."
WRONG: Implement 1,2,3,6 now, ask about 4,5 later
```

### From External Reviewers — Verify Before Implementing

Before implementing: technically correct for THIS codebase? Breaks existing functionality? Reason for current implementation? Works on all platforms? Does reviewer understand full context?

If suggestion seems wrong: push back with technical reasoning.
If conflicts with prior architectural decisions: stop and discuss with project owner.

### When to Push Back

- Suggestion breaks existing functionality
- Reviewer lacks full context
- Violates YAGNI (unused feature)
- Technically incorrect for this stack
- Legacy/compatibility reasons exist
- Conflicts with prior architectural decisions

**How:** Technical reasoning, specific questions, reference working tests/code.

### Implementation Order for Multi-Item Feedback

1. Clarify anything unclear FIRST
2. Blocking issues (breaks, security)
3. Simple fixes (typos, imports)
4. Complex fixes (refactoring, logic)
5. Test each fix individually
6. Verify no regressions

### GitHub Thread Replies

Reply inline in comment threads (`gh api repos/{owner}/{repo}/pulls/{pr}/comments/{id}/replies`), not as top-level PR comments.

---

## Cross-References

Load these skills if the review scope requires them:
- `workflow:code-quality` — if detecting code smells or anti-patterns not covered above
- `security:security-analysis` — if the diff touches auth, crypto, network, or trust boundaries
- `security:auth-implementation-patterns` — if reviewing auth/authz logic specifically
- `security:secrets-management` — if the diff contains credentials, tokens, or env var handling
- `testing:language-testing-patterns` — if assessing test coverage or test quality
- `testing:test-driven-development` — if evaluating test design or TDD compliance
- `workflow:verification-before-completion` — if implementing fixes discovered during review
- `workflow:pr-comment-resolution` — if responding to reviewer feedback on your own PR
- `languages:*-patterns` — load the language-specific skill matching the primary language in the diff
