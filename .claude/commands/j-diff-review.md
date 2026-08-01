---
name: j-diff-review
description: "Deep multi-perspective diff review — code quality, security, testing gaps, documentation drift, observability gaps, and language-specific gotchas. Use when reviewing a diff or PR before merge. Do NOT use for simple code questions (ask directly instead)."
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
- `documentation-validation` — the per-change docs gate and its change-type matrix

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

Note which languages the diff touches so the language agent can load the matching references under
`.claude/references/languages/`.

Flag missing tests when the diff changes source but no test files.

## Step 4: Analyze from each perspective

Dispatch all six agents in parallel, in a single message, on every review. Give each the same frozen
packet: the diff, the changed-file list, and the branch name.

Every dispatched agent **reports only** — it returns findings and edits nothing. The report-only default
in `code-review-patterns` governs the agents that load it; disposition is your job (Step 6). An agent
whose lens does not apply to this diff returns "no findings — surface not present" rather than
manufacturing material.

| Perspective | Agent | Looks for | Loads |
|---|---|---|---|
| Code quality | `code-reviewer` | Edge cases, error handling, logic errors, missing validation, smells, coupling | `code-review-patterns`, `output-completeness` + `.claude/references/workflow/`; apply `code-quality` |
| Security | `security-reviewer` | Injection, XSS, SSRF, path traversal, auth gaps, secrets, insecure defaults | `code-review-patterns` + `.claude/references/security/` |
| Testing | `test-writer` | Coverage gaps, tests asserting implementation rather than behavior, flakiness | `test-driven-development`, `language-testing-patterns` + `.claude/references/testing/` |
| Documentation | `documentation-writer` | Stale README/API/CHANGELOG/config/CLI docs, new public surface left undocumented, drifted paths and counts | `documentation-validation`, `post-ship-doc-sync` + `.claude/references/documentation/` |
| Language-specific | `language-specialist` | Idiom violations and per-language traps for the languages the diff touches | `language-testing-patterns`, `test-driven-development` + `.claude/references/languages/` |
| Observability | `devops-engineer` | New code paths with no logging/metrics/tracing, silently swallowed errors, new surface with no SLO or alert, analytics events missing from the tracking plan | `.claude/references/devops/` (observability, sre-practices, incident-management) + `.claude/references/architecture/error-handling-patterns` |

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
### Doc Gaps
### Observability Gaps
### What Looks Good
```

Close with a **PASS / CONCERNS / FAIL / BLOCKED** verdict. BLOCKED describes a limit of the review, not a
defect in the code — say what you could not cover and why.

## Step 6: Disposition ladder

The agents reported; none of them changed anything. Every finding is now yours to dispose of. For each
one, take the first rung that applies — do not skip ahead:

1. **Fix it** — if the fix is reasonably scoped: clear defect, inside the diff's boundary, verifiable.
   Apply it, run the project's checks, commit atomically.
2. **Add it to the active plan** — if a plan file already exists in `~/.claude/plans/`, append the finding
   there as a future phase. **Never create a new plan file.**
3. **Add it to the repository's future-work mechanism** — if the repo has one, follow its convention
   (`TODO.md`, `docs/plans/`, GitHub issues, a tracker named in CLAUDE.md or CONTRIBUTING.md). Detect it;
   do not invent one.
4. **Ask** — nothing above applied. Ask the user, carrying a recommendation and the research behind it.

Read the code a finding touches before deciding. Reaching rung 4 without having researched is the failure
mode; so is skipping rung 1 for something you could simply have fixed.

**Handing back a findings list with no disposition is a failed run.** State the rung for every finding.

Scope guard: fixing a finding does not license unrelated refactors. A fix that grows past the diff's
boundary drops to rung 2. Follow `pr-comment-resolution` for atomic commits.
