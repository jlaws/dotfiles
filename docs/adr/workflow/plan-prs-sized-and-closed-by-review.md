---
status: accepted
topic: workflow
created: 2026-11-14
updated: 2026-11-14
---
# Plan PRs are sized for review and closed by a diff review

## Context

Plan execution ended when `finishing-branch` opened the PR. No execution surface ran j-diff-review, and
subagent mode ran its own single-reviewer whole-branch pass. j-diff-review's disposition ladder fixed
only findings inside the diff's boundary, appended others to the active plan as future phases, or used
a tracker it detected ("Detect it; do not invent one"). `writing-plans` defaulted to one PR per plan,
while `code-review-patterns` asked to split any PR over 400 lines.

The user asked for plan runs that end with j-diff-review, fix everything except large, unrelated items
(filed in the repository's own bug and task tracker), and report the PR link and whether another plan
PR remains; and for plans broken
into reviewable PRs of about 3-5k lines, not as a hard requirement.

## Decision Drivers

* **Must end every plan PR boundary with j-diff-review** (user request).
* **Must fix everything except findings that are both large and unrelated, and file every one of those, security findings included** (user request).
* **Must file through each repository's own tracking mechanism**, not a design the KB imposes: repositories use GitHub issues, Linear, or a `tasks/` folder (user request).
* **Should keep one PR-opening path per boundary**, so the stop point and the final report are unambiguous.
* **Should target about 3-5k changed lines per PR, not as a gate** (user request).

## Decision

We will close each plan PR boundary with **j-diff-review before the PR opens**. `executing-plans` Step 5
and `subagent-driven-development` Step 2 call it; its Step 7 opens the PR or pushes to the open one and
writes the run's single report, ending with the PR URL and whether another plan PR remains. Its ladder
fixes every finding except one that is both large and unrelated (defined in Step 6), which is filed in
the repository's own bug and task tracking mechanism, following its format (an in-repo entry is
committed on the branch); security findings follow the same rule, and
`writing-plans` no longer keeps them off public trackers. Reviews diff against `origin/main`. Each
reviewer runs in a fresh context on an artifact-only packet (base ref, branch, HEAD SHA, changed files,
languages), loads the diff itself, and decides what else to read; the orchestrating session, often the one that wrote the
code, lists the changed files, checks only the lines findings cite, and never passes the plan, the spec,
or its intent to a reviewer. Plans target about 3-5k changed lines per PR; `code-review-patterns` asks to
split only a PR well over that with no stated reason.

## Rationale

Reviewing before the PR opens leaves one opener (j-diff-review Step 7) and one stop point, so neither
`finishing-branch`'s "Then stop" nor the executor's PR stop rule needs a carve-out, and the PR shows its
fixes from the first push. One review per boundary avoids paying for two whole-branch reviews in subagent
mode. Fixing a small unrelated defect does not conflict with "lazy about new scope" in
`code-ladder-vs-completeness.md`: a defect fix is not new scope. The 400-line prompt would fire on every
PR sized to the new target and stop an automated run.

## Consequences

**Gained**: every plan PR gets the multi-perspective review and lands with its fixes; deferred work has one destination; the final report says what comes next.
**Accepted**: six review agents per boundary; PRs larger than the 400-line heuristic for human reviewers; filed entries accumulate until cleaned up; a repository with no tracking mechanism gets its findings listed as unfiled in the report rather than filed; a security finding filed in a public tracker is public.

## Ruled Out

| Idea | Why ruled out | When |
|------|---------------|------|
| Review after the PR opens | `finishing-branch` "Then stop" and the executor's PR stop rule would need carve-outs, and the PR gets a second push of fixes | 2026-11-14 |
| Append deferred findings to the active plan | Adds a phase after the validation gate and makes /j-next pick up a phantom part | 2026-11-14 |
| Mandate a `tasks/` folder in every repository | Repositories track work differently (GitHub issues, Linear, a `tasks/` folder); a KB-imposed design would compete with each repo's own | 2026-11-14 |
| Keep subagent mode's own whole-branch review | Doubles review cost per boundary; j-diff-review's lenses cover it | 2026-11-14 |
| Reviewers forked from the orchestrating session, or briefed with the plan and intent | They inherit a context that is often the one that wrote the code, and a reviewer handed the author's intent tends to confirm it | 2026-11-14 |
| Keep security findings out of public trackers | The user chose to file every large, unrelated item, security included | 2026-11-14 |
| Keep one PR per plan as the default | Replaced by the 3-5k changed-line target | 2026-11-14 |
| Keep the 400-line split prompt | Fires on every PR sized to the target and stops automated runs | 2026-11-14 |

## Enforcement

- Owner: `.claude/commands/j-diff-review.md` (Steps 6-7), `.claude/skills/executing-plans/SKILL.md` (Step 5), `.claude/skills/writing-plans/SKILL.md` (PR Boundaries), and their mirrors.
- Pinned by: `tests/test_agent_config.py::test_diff_review_fixes_all_but_large_unrelated_findings`, `::test_diff_review_diffs_against_origin_main`, `::test_diff_review_reviewers_start_from_a_clean_context`, `::test_plan_boundaries_close_with_the_diff_review`, `::test_plans_size_prs_for_review`.

## Reversal Conditions

Review cost per boundary becomes the dominant cost of plan runs, filed entries stay unresolved across
several later PRs, or reviewers report target-sized PRs as too large to review.

## Related

- [Thorough about quality dimensions, lazy about new scope](code-ladder-vs-completeness.md)
- [ADRs are living documents, and dead ones are deleted](adrs-are-living-documents.md)
- [Plans record decisions, not code](plans-record-decisions-not-code.md)
