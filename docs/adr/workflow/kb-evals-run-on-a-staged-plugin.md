---
status: accepted
topic: workflow
created: 2026-11-15
updated: 2026-11-25
---
# KB evals run on a staged plugin

## Context

Each skill should have evals, and a baseline should exist before the KB changes. `claude plugin
eval` scores cases against a plugin and isolates every run from `~/.claude`: no user CLAUDE.md,
settings, hooks, memory, or outside skills load. This KB is a plain `.claude/` tree mirrored to
`~`, not a plugin.

## Decision Drivers
* **Must compare the KB against plain Claude** on the same tasks, so a score means something
* **Must not leak KB content into the plain-Claude arm**
* **Should reuse the CLI's graders, votes, and cost caps** rather than reimplement them

## Decision

We will **stage the Claude tree into two temporary plugins and run each with `--ablation none`**:
- `kb`: skills, commands, and agents. `.claude/CLAUDE.md` plus two harness notes go in through each
  case's `append_system_prompt`; an agent case's task also carries the reference-path note, because
  a dispatched agent does not see that system prompt. References are copied into each run's
  workspace by a scaffold.
- `vanilla`: an empty plugin, which is plain Claude.

Delta is kb minus vanilla, computed on result graders only. There are three suites (triggers,
effects, workflows), each in its own eval dir; triggers run on kb only. Models are Opus 5.5 and
Sonnet 5.5, with Fable 5.2 as the sole judge. Each case runs three times. Spend is checked against
a machine-local, locked ledger and a committed per-period budget. Compact summaries are committed;
raw results stay machine-local. Runs bill an API key. One invocation holds one CLI version:
- The runner pins `claude` to its real path once and runs that binary throughout.
- It checks the version before each batch's paid call; a change stops the run with exit 4.
- A result written by another version is settled, then quarantined with a note.
- An interrupted reservation stays charged, because in-flight spend is unknown; the `settle`
  command closes it at a cost the owner states.
- `run --plugin` refuses a plugin staged before an edit to the tree: the stage marker holds a source
  digest, and a mismatch exits 1 before any spend.

## Rationale

Two stagings make each arm explicit: nothing in the vanilla tree can reach the KB. Running the
CLI keeps graders, judge votes, isolation, and cost ceilings in one maintained tool.

## Consequences
**Gained**: within-harness comparisons are valid, per case and per suite, across models.
**Accepted**: absolute scores carry these divergences from a real session:
- CLAUDE.md arrives as appended system text.
- References sit in the workspace, not under `~/.claude/references/`.
- Asset names carry the `kb:` namespace.
- No settings or hooks load.
- Bash is sandboxed.
- No network and no `gh`.
- Git cases name a working git in the prompt, because the sandbox's `/usr/bin/git` shim fails.

Every asset needs cases, and each run costs money. Capped runs are scored and censored runs are
counted apart. A model in the judge's family cannot be evaluated without first changing the judge.
A summary's KB digest check compares results with each other, not with the staged KB: results
recorded before the digest sidecar read `unrecorded` and are unverified, and a partial rerun over
older results is refused until that staging is rerun or the older results are moved aside.

## Ruled Out
| Idea | Why ruled out | When |
|------|---------------|------|
| The CLI's built-in with/without arms | Case-level `append_system_prompt` and the scaffold would leak the KB into the without-arm | 2026-11-15 |
| A custom `claude --bare -p` harness | Reimplements graders, judge votes, and cost caps | 2026-11-15 |
| Converting the KB into a plugin | Breaks the `~` mirror that every harness reads | 2026-11-15 |
| Evaluating in the real `~/.claude` | Personal settings, memory, and hooks would confound every score | 2026-11-15 |
| Cross-judging with each model under test | Confounds the cross-model comparison | 2026-11-15 |
| Drop capped runs from the score | 41 of 46 capped trigger runs had already fired, so dropping them lowers should-fire recall by 3.6 points on Opus and 4.4 on Sonnet | 2026-11-23 |
| Stage per invocation into a new directory | Re-keys every cached batch: `seed.sh` holds the staged path | 2026-11-25 |
| Read the CLI version once per step | A symlink repointed mid-step lets later batches run on another version | 2026-11-25 |
| Release an interrupted reservation automatically | The cost of runs in flight is unknown, so any release can undercount | 2026-11-25 |
| Compare any two summaries | A judge change reads as a KB change; compare refuses summaries judged by different models | 2026-11-25 |

## Enforcement
- Owner: `evals/`
- Pinned by: `tests/test_evals.py`
- Judge-family guard in `runner.run` and `summary.check`, pinned by `tests/test_evals.py`
- Every case names the asset rule it tests, and every reference-indexing agent has a
  reference-grounded case, both checked by `evals/kbeval/cases.py` lint

## Reversal Conditions
- Claude Code adds a supported way to evaluate a user config directory.
- The pilot shows references or agents unreachable from the staged plugin.

## Related
- [Skills do not pre-approve tools](skills-do-not-pre-approve-tools.md)
