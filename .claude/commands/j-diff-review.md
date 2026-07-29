---
name: j-diff-review
description: "Deep multi-perspective diff review — code quality, security, testing gaps, and language-specific gotchas. Use when reviewing a diff or PR before merge. Do NOT use for simple code questions (ask directly instead)."
argument-hint: "<diff-ref-or-branch>"
model: opus
effort: xhigh
---

Review: $ARGUMENTS

Load these skills before starting:

- `code-review-patterns` — review mindset, severity labels, giving and receiving feedback
- `language-testing-patterns` — test quality and coverage assessment
- `analysis-output-patterns` — output structure
- `verification-before-completion` — verdict grammar and evidence hierarchy for reporting findings

Read `.claude/references/workflow/existing-code-discipline.md` when the diff touches established code.

---

## Step 1: Identify changes

```bash
git diff main...HEAD
git log main..HEAD --oneline
```

Stop if the current branch is main or has no commits ahead of main.

**Use local git only — no `gh` or other GitHub CLI.** A review must reflect the code in front of you, not
PR metadata that can disagree with it.

## Step 2: Gather full context

```bash
git diff main...HEAD --name-only
```

**Read every changed file in full**, not just the diff hunks. Cross-cutting problems live in the code the
diff did not touch — a changed function's callers, an invariant asserted elsewhere.

## Step 3: Detect scope

Match file extensions to the language references under `.claude/references/languages/` and read the ones
that apply: `python-patterns`, `js-ts-patterns`, `go-concurrency-patterns`, `bash-defensive-patterns`,
`swift-patterns`, `rust-project-patterns`.

Flag missing tests when the diff changes source but no test files.

## Step 4: Analyze from each perspective

The four perspectives are independent, so analyze inline for a small diff, or delegate them in parallel
for a large one:

| Perspective | Looks for | Agent |
|---|---|---|
| Code quality | Edge cases, error handling, logic errors, missing validation, smells, coupling | `code-reviewer` |
| Security | Injection, XSS, SSRF, path traversal, auth gaps, secrets, insecure defaults | `security-reviewer` |
| Testing | Coverage gaps, tests asserting implementation rather than behavior, flakiness | `test-writer` |
| Language-specific | Idiom violations and per-language traps from the Step 3 references | inline |

Deduplicate across perspectives and resolve contradictions. Check each delegated finding against the
diff — a subagent's summary describes what it looked for, the diff shows what is there.

### Adversarial debate, for high-risk diffs

When the change is risky enough that a missed finding is expensive, escalate:

1. **Freeze a shared packet** — the diff plus context, identical for every reviewer.
2. **Fan out the perspective agents blind to each other**, scaling the count to the risk.
3. **Cross-critique for one or two rounds.** Broadcast round-one findings; a reviewer may revise, but a
   change of position has to carry a technical reason. "Good point" is not one.
4. **Report survivors and disputes.** What survives cross-critique is high-confidence. Leave genuine
   disputes for the human instead of forcing consensus.

## Step 5: Report

Every finding cites `file:line`. Omit empty severity sections, and always include what looks good.

```markdown
## Diff Review — {BRANCH_NAME}

### Critical
- {finding} — {file:line} — {perspective}

### High
### Medium
### Test Gaps
### What Looks Good
```

Close with a **PASS / CONCERNS / FAIL / BLOCKED** verdict. BLOCKED describes a limit of the review, not a
defect in the code — say what you could not cover and why.

## Step 6: Decision gate

**Report only by default.** Findings are not a mandate to change code: the author may have context you
do not, and mixing review with rewriting hides what was wrong.

Then ask whether to implement the fixes or finish. If implementing, follow `pr-comment-resolution` for
scope guard, atomic commits, and reply workflow.
