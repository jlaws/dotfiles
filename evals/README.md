# KB evals

Behavioral evals for the Claude tree (`.claude/`): does each skill, command, and agent trigger when
it should, and does it improve the result over plain Claude? The harness stages the tree into a
temporary plugin and runs `claude plugin eval` on it. The decision record is
`docs/adr/workflow/kb-evals-run-on-a-staged-plugin.md`. This directory is repo-only; setup does not
sync it to `~`.

## Run

Every target below except `eval-stage`, `eval-lint`, `eval-spend`, and `eval-compare` bills the API
key. Export `ANTHROPIC_API_KEY` first. Only these targets set `KBEVAL_ALLOW_PAID=1`; without it the
runner refuses to call `claude`.

```bash
make eval-stage                      # free: build $TMPDIR/kbeval-kb and $TMPDIR/kbeval-vanilla
make eval-lint                       # free: lint every case; ends "0 problems"
make eval-pilot                      # paid: the five pilot cases, both models, one run each
make eval-smoke ASSETS="a b"         # paid: kb on Sonnet, one run per case, then a vanilla sample
make eval-baseline MODEL=claude-opus-5-5   # paid: the full baseline for one model
make eval-compare OLD=<summary.json> NEW=<summary.json>   # free: regressions between two summaries
make eval-spend                      # free: snapshot ledger totals into evals/spend.json; exits 1 over the ceiling
python3 -m evals.kbeval summarize    # free: check every result, then write evals/baselines/ (nothing if a check fails)
```

`python3 -m evals.kbeval run --suite ... --staging ... --plugin ...` runs one suite for one staging
and model; the make targets call the plans built on it.

`J ?= 8` sets `-j` (concurrent runs). `eval-baseline` also takes `VANILLA_RUNS` (default 3),
`VANILLA_MODELS` (default both models), and `EXCLUDE_TAGS` (default empty); the kb staging always
runs three times per case.

## Budget

`evals/budget.json` holds the period, its ceiling, the smoke cap, and the per-run p95 cost the
reserve uses (measured by the pilot below):

```json
{"period": "initial-baseline-2026-11", "ceilingUsd": 2500, "smokeCapUsd": 40, "p95RunUsd": 1.0}
```

A new period needs a committed bump that the user approves. The spend ledger is machine-local, at
`${XDG_STATE_HOME:-$HOME/.local/state}/dotfiles-kbeval/ledger.jsonl`, and every append holds an
exclusive lock. `make eval-spend` snapshots its per-period totals into the committed
`evals/spend.json`.

Each batch's `--max-cost-usd` cap is the smallest of three limits: the remaining budget minus a
reserve of `jobs x p95RunUsd` for in-flight runs, a batch allowance of `cases x runs x max(10 x
p95RunUsd, $5)`, and the plan's own cap ($10 for the pilot, `smokeCapUsd` per branch for smoke).
The CLI checks the cap before each run starts, so in-flight runs can pass it; the reserve covers
them.

Before the paid call the runner reserves the cap in the ledger, checking the budget and appending
under one lock, so concurrent runners cannot spend the same headroom. After the call it settles
the reservation to the actual cost. A crash, an interrupt, or an unreadable result leaves the full
cap charged; a run that never loaded (`No eval cases found`, an untrusted plugin) settles to $0. A
result whose cost is missing, negative, or not a number aborts the run with the cap charged. A
ledger line that cannot be read stops every run until it is fixed by hand.

Each batch writes `cli.log` (exit code, stdout, stderr) beside its result. Every result, log, and
kept trace is scanned for `sk-ant-`; a hit quarantines the result and aborts the run.

## Layout

```text
evals/
├── budget.json, spend.json, harness.json
├── kbeval/                     # stage, lint, run, summarize, compare; determination checks
├── fixtures/<name>/            # frozen fixture trees (specs below)
├── cases/
│   ├── triggers/<asset>--<kind>-<slug>/
│   ├── effects/<asset>--<kind>-<slug>/
│   └── workflows/wf--<slug>/
└── baselines/<model>/<suite>.json   # compact committed summaries
```

- `kind` is one of `trigger-should`, `trigger-near`, `trigger-noauto`, `effect`.
- The directory name is the case name, so names are unique by construction.
- Tags: `[<suite>, <kind>]`, plus `heavy` for multi-agent commands and `extended` on an asset's
  third effect case.
- A case holds `prompt.md`, `graders/*.md`, and optionally `case.yaml` (`context` keys only) and
  `fixture.sh`.

`harness.json` selects two staging modes the pilot can switch: `commandForm` (`slash`, or `skill`)
and `refsMode` (`workspace`, or `add_dirs`).

## Case-authoring rules

**Prompts**
- Trigger and skill-effect prompts read like user requests and never name the skill.
- Command effect and workflow prompts start with `/kb:<command> `, followed by the task in plain
  language.
- Agent effect prompts start with `Use the kb:<agent> agent to `.
- For the vanilla staging, the builder strips that leading directive, so plain Claude gets the bare
  task.
- The builder appends one fixed line to every effect and workflow prompt, in both stagings: "Run
  non-interactively: do not ask questions; take the recommended option for every decision and state
  it."
- For a case whose `fixture.sh` runs `git init`, the builder adds one line before that, in both
  stagings: "In this sandbox `/usr/bin/git` fails; run git as `<git>`." `<git>` is the first `git`
  on the staging machine's `PATH` outside `/usr/bin`; with none, the line is left out.

**Grading contract.** The prompt states every graded criterion and every graded artifact path, such
as "save the plan to ./plans/plan.md". A `regex` or `llm` result grader that reads a file names one
path through `target`/`focus: { source: file, path: <path> }`.

**Triggers**
- Exactly one `tool_used` grader: `tool: Skill`, `input_match: '"skill"\s*:\s*"(?:kb:)?<name>"'`.
- `trigger-should`: `min: 1`. `trigger-near` and `trigger-noauto`: `min: 0`, `max: 0`.
- `max_turns: 6`, `timeout_seconds: 120`, `allowed_tools: [Skill, Read, Glob, Grep]`. Six, not
  three: at three, runs that read a file first hit the turn cap before reaching the skill.

**Effects and workflows**
- At least one result grader: `regex` on a produced file, `file_exists`, or `llm` with 2-5
  concrete `PASS if` / `FAIL if` claims, all required. Result graders never mention `kb:`.
- At least one process grader: `tool_used` on `Skill`, or on `Agent` with
  `"subagent_type"\s*:\s*"kb:<agent>"`, or `tool_order`, or `tool_used` on `Read` with
  `input_match: '(?:\.kb-references|_refs)/<dir>/'` (both `refsMode` layouts). Process graders
  are indicators; the summary leaves them out of delta. They also match tool calls made inside a subagent, so a `gh` call
  the `create-pr` agent makes counts.
- Effects: `max_turns: 40`, `timeout_seconds: 900`,
  `allowed_tools: [Skill, Read, Glob, Grep, Agent, TodoWrite]`.
- Workflows: `max_turns: 150`, `timeout_seconds: 3600`, same tools.

**Fixtures**
- Optional `fixture.sh` gets the staged fixtures dir as `$1` and starts with `set -euo pipefail`. It
  copies from `$1` and builds local state in the working dir with `git`, `mkdir`, and `printf`. The
  seed runs it with the user's global and system git config switched off.
- Lint rejects network and exec commands as whole words (`curl`, `wget`/`wget2`, `ssh` and
  `ssh-*` tools, `scp`, `nc`/`ncat`, `sudo`, `brew`, `eval`, `git clone`, `npm install`/`i`/`ci`,
  `pip install`, recursive `rm`, `sh`/`bash`/`zsh`/`dash -c`, `python`/`node`/`perl`/`ruby -c`/`-e`,
  `git config --global`/`--system`, force pushes, `git remote add` to a URL), absolute paths
  outside `$1`, and `..` paths. Heredoc bodies are seeded file content and are not checked as
  commands.
- Never write `context.scaffold_script`; the builder owns it. `case.yaml` holds only `context:` with
  a flow-list `add_dirs: [..]` or a `history_file:`.
- `fixture.sh` may also seed files with quoted heredocs.
- Lint checks these rules line by line. It is a guard rail for reviewed, in-repo cases, not a
  security boundary.

**What lint enforces beyond the rules above.**
- Frontmatter keys: `prompt.md` and each grader type accept only their listed keys, and each grader
  type has its required keys (`regex`: `pattern` and `target`; `file_exists`: `path`; `tool_used`:
  `tool`; `tool_order`: `before` and `after`).
- YAML: quoted scalars, including values inside flow maps, must parse (no unescaped inner quote, no
  invalid escape such as `"\d"` inside double quotes).
- Regex: JavaScript syntax only. No inline flags (`(?i)`), `(?P<name>`, POSIX classes, or
  `\A`/`\Z`/`\z`; use `flags:` (letters from `dgimsuvy`) and `^`/`$`. A pattern that matches
  everything is rejected.
- A pattern that starts with a `^##` heading anchor needs `flags: m`; without it `^` matches only
  at file start, where a level-2 heading never sits.
- Seed checks see only files a fixture copies from `evals/fixtures/`, not files its heredocs write.
- `llm` graders hold 2-5 `PASS if` / `FAIL if` claims.
- Grounding: every graded path is named in the prompt or seeded by the fixture. A result grader
  that already passes on the seeded files, or a `file_exists` with `exists: false`, is rejected,
  because a run that does nothing would score it. A `file_exists` or `match: not_contains` grader
  needs a positive regex or llm grader on the same path, so an empty or missing file cannot pass.
- No graded path under `.claude/`, no grader file name ending `.md.md`, no non-`.md` file in
  `graders/`, and at most one `extended` case per asset.

**Limits.** `runs` 1-50, `max_turns` at most 200, `timeout_seconds` at most 3600. Only `EVAL_*` env
keys. No WebFetch or WebSearch. No `sk-ant-` strings. No network, secrets, or private project names:
the repo is public, and every input comes from the prompt or a fixture.

## Offline limits

What an offline run can show, per asset where it differs from a real session:

| Asset | Offline limit |
|---|---|
| j-create-pr | Branch, commit, and push to the local remote; the `gh pr create` attempt is graded as a call |
| j-rebase | The prompt supplies the PR branch |
| j-pr-fix | Comments arrive inline in the prompt |
| j-next | The prompt states that the previous PR merged |
| j-paper, j-explain | Local sources only |
| j-email-analysis | No downloads |
| github-issue-resolution | The issue text arrives inline in the prompt; no `gh` |
| pr-comment-resolution | Comments arrive inline in the prompt; replies are written to a file, not posted |
| finishing-branch | Push goes to the local bare remote; the `gh pr create` attempt is graded as a call |
| post-ship-doc-sync | The prompt supplies the shipped change list in place of `git log` against a real remote |
| dependency-upgrade | No package downloads; the case grades the upgrade plan and edits, not an install |
| j-retro | History is synthetic: the fixture writes dated commits, a tag, a revert, and an unmerged branch on a local repo |
| j-config-audit | The audited `.claude/` and `.codex/` trees are seeded into the workspace as test input (they do not load as config); no user-level `~/.claude` |
| j-email-consolidate | Per-email reports are pre-written by the fixture; no source emails or downloads |
| create-pr | Same as j-create-pr: push to the local remote; the `gh pr create` attempt is graded as a call |
| research-analyst | Local sources only; no WebFetch or WebSearch |
| j-cloud, j-devops, j-observe, j-data, j-frontend | Artifacts are written, not deployed, run, or built |

## Divergences

Absolute scores differ from a real session in these ways; comparisons within the harness hold:
- CLAUDE.md arrives as appended system text.
- References sit in the workspace at `./.kb-references/`, not under `~/.claude/references/`.
- Asset names carry the `kb:` namespace.
- No settings or hooks load.
- Bash runs under the OS sandbox.
- No network and no `gh`.
- Files under `.claude/` cannot be written (the run's permission mode denies it), so cases
  write skill, agent, and settings drafts to workspace paths.
- Inside the Bash sandbox `git` resolves to the macOS `/usr/bin/git` shim, which fails (`xcrun`
  cannot write its cache; often `xcode-select: Failed to locate 'git'`). The CLI builds the run's
  `PATH` itself, and a case's `env:` accepts only `EVAL_*` keys, so the caller cannot fix it. Without
  help, most git-fixture runs in the `smoke-feat-kb-evals-workflow-commands` smoke hit it, and kb
  runs stopped on it more often than they switched binaries (the KB's stop-after-two-failures
  rule), which biases against kb. The builder's git line (Prompts) names a working git in both
  arms.
- The seed adds `__pycache__/` to `.git/info/exclude`, so running Python leaves `git status` clean.
- The child session inherits `ANTHROPIC_API_KEY`, so a run granted Bash could print it. The
  sandbox blocks network, and the key scan covers results, logs, and kept traces; traces of runs
  without `--keep-temp` are deleted by the CLI unscanned.

## Fixture specs

The specs are frozen; cases depend on these exact contents. Each fixture ships with the case PR
that first uses it; `py-calc`, `py-calc-red`, `ts-todo`, `docs-drift`, `kb-mini`, `email-digest`,
`paper-stub`, and `py-calc-planned` ship so far. A case's `fixture.sh` may layer state on top of a frozen fixture
(appended commits, extra files); the fixture itself does not change. Tests pin `py-calc`'s seeded
bug, `py-calc-red`'s one failing test, `docs-drift` rejecting `--out`, and `py-calc-planned` being
`py-calc` plus `plans/plan.md`.

| Fixture | Contents |
|---|---|
| `py-calc` | `calc/__init__.py`; `calc/ops.py` with `add(a, b)` and `mean(xs)`. The docstring says the mean of an empty list is `0.0`, but the body divides by `len(xs)` (the seeded bug). `tests/test_ops.py` (unittest) covers `add` and `mean([1, 2, 3])` only. README says `python3 -m unittest`. |
| `py-calc-red` | `py-calc` plus `tests/test_empty.py` asserting `mean([]) == 0.0`, which fails |
| `py-calc-planned` | `py-calc` plus `plans/plan.md`: a two-task plan to fix `mean([])` test-first and document it in README |
| `ts-todo` | `src/todo.js` (ESM `addTodo`, `complete`) and `test/todo.test.js` (`node --test`, covering `addTodo` only) |
| `docs-drift` | `tool.py` with an `--output` flag and `allow_abbrev=False`, so `--out` fails with "unrecognized arguments"; README documents `--out` |
| `email-digest` | `inbox/2026-11-01-newsletter.txt`: plain text with five links and two paper titles |
| `paper-stub` | `paper/main.tex` skeleton with an empty Related Work section, plus `refs.bib` with three entries |
| `kb-mini` | `.claude/skills/broken-skill/SKILL.md` (missing `description`) and `.claude/commands/j-demo.md` |

**Git fixtures.** `fixture.sh` copies `$1/<name>/.`, runs `git init -q -b main`, sets repo-local
`user.name kbeval` and `user.email kbeval@example.invalid`, runs `git add -A`, and makes the
initial commit. It then creates the remote with `git init -q --bare .remote.git`, `git remote add
origin ./.remote.git`, and `git push -q origin main`, and lists `.remote.git` in
`.git/info/exclude` (the seed adds `.kb-references`). A case may add branches, commits, or a
worktree after that to set up its scenario.

## CLI surface

Checked on `2.1.325 (Claude Code)`: every flag the runner passes exists in
`claude plugin eval --help`.

## Summaries and compare

`python3 -m evals.kbeval summarize [--check]` writes `evals/baselines/<model>/<suite>.json`: the run
settings, Claude Code version, judge, date, cost, and one line per case with `kbScore`,
`vanillaScore`, `delta`, `firedRate`, run counts, `errorRuns`, and the distinct error strings.
Scores use result graders only; `firedRate` is the process-grader pass rate in kb (for a
near-miss or noauto trigger, 1.0 means the skill did not fire). A case `passed` when `kbScore`, its mean run score,
is at least 2/3. Suite aggregates carry a 95% bootstrap interval over cases (seed 0).
It reads only results the runner accepted (those with a `.key`), and it checks before writing: if
any case is missing runs or still holds an API error, it writes nothing and exits 1 (`--check` is
accepted and changes nothing). It also refuses to write text holding an `sk-ant-` key, a local
path (`/Users/`, `/home/`, `/var/folders/`), or an API request or message ID, because summaries
are committed to a public repo.

`make eval-compare OLD=... NEW=...` compares two summaries: the suite means with both intervals,
cases whose `kbScore` dropped by at least 1/3, and pass/fail flips. Cases in only one file are
listed as "not run here", since a partial rerun covers a subset.

## Pilot measurements

Run 2026-11-15 on Claude Code 2.1.325, one run per case, judge `claude-fable-5-2`. Cost per run is
the CLI's `costUsd` at list price, judge calls included.

| Type | Model | Runs | Mean $/run | Max $/run |
|---|---|---|---|---|
| trigger | Opus 5.5 | 2 | 0.12 | 0.14 |
| trigger | Sonnet 5.5 | 2 | 0.06 | 0.07 |
| effect-skill | Opus 5.5 | 2 | 0.12 | 0.15 |
| effect-skill | Sonnet 5.5 | 2 | 0.05 | 0.07 |
| effect-command | Opus 5.5 | 5 | 0.42 | 0.98 |
| effect-command | Sonnet 5.5 | 4 | 0.13 | 0.21 |
| effect-agent | both | 0 | not measured; extrapolated at the effect-command rate | |

Per-run p95 for the reserve: $1.00 (`p95RunUsd` in `evals/budget.json`).

Harness checks (kb staging):

| Check | Result | Evidence |
|---|---|---|
| References readable | yes | A probe told only "references/workflow/hook-patterns.md" read `./.kb-references/workflow/hook-patterns.md` on its first tool call |
| `kb:` agents resolve | yes | The run's init lists `kb:architecture-specialist` and every other KB agent; the Opus j-arch run dispatched it |
| `/kb:` commands expand | yes | A probe given `/kb:j-arch` quoted the command body verbatim with no tool call |
| Model and judge IDs | yes | Runs report `claude-opus-5-5` and `claude-sonnet-5-5`; judge votes ran on `claude-fable-5-2` |
| `sk-ant-` in results or traces | none | Scanned after every batch and in the kept probe traces |

Branch taken: none. `commandForm` stays `slash` and `refsMode` stays `workspace`.

The CLI deletes each run's trace unless `--keep-temp` is passed, so smoke runs pass it and their
traces stay under `/private/tmp/e-*` (sealed, read-only) for transcript review; the key scan covers
them.

**Extrapolation** (the pilot-era estimate of 2026-11-15, superseded by `## Baseline` below; kept for
the record). Planned cases, from the coverage rules: about 124 trigger cases, about 187
effect cases (about 71 skill, 76 command, 39 agent, counting third cases), and 4 workflow cases.
Three runs per case; effects and workflows run in both stagings. Workflows are priced at five times
the effect-command rate.

| Item | Opus 5.5 | Sonnet 5.5 |
|---|---|---|
| Triggers (kb only) | $45 | $24 |
| Effects, skill | $50 | $23 |
| Effects, command | $189 | $61 |
| Effects, agent | $97 | $31 |
| Workflows | $48 | $17 |
| Baseline total | $429 | $156 |

Projected total: $585 for both baselines, plus up to $160 of smoke (four case PRs at the $40 cap),
plus the $8 reserve (8 jobs at the $1.00 p95), about $753 against the $2,500 budget. The pilot cases
were small; heavy multi-agent commands will cost more per run, so the baseline step re-extrapolates from
smoke data.

## Baseline

**Settings** (re-extrapolated 2026-11-15 from `make eval-lint` counts and smoke costs): 124 trigger,
162 effect, and 4 workflow cases; three kb runs per case, and for effects and workflows three
vanilla runs on both models (`VANILLA_RUNS=3`); no tags excluded. Measured Sonnet means per run:
triggers $0.07 (no judge), effects $0.29 kb and $0.19 vanilla including the judge; workflows priced
at $1.00. Opus was priced at twice Sonnet. Estimate: about $274 Sonnet and $548 Opus, $822 in all,
against $2,383 left of the $2,500 budget, so no trim applied. Actual: $245.26 Sonnet and $416.18
Opus, $661.44 in all (Opus ran about 1.7 times Sonnet; workflows about $0.31 per Opus run). Period
total after the baseline, pilot and smoke included: $778.83 of $2,500 (`evals/spend.json` at that
time; the post-fix rerun below moved it).

Results (2026-11-16, Claude Code 2.1.325, judge Fable 5.2). Means are per-case scores on result
graders (triggers: the Skill grader), each column rounded on its own, with case-level bootstrap 95%
CIs; delta is kb minus vanilla per case.

| Model | Suite | Cases | kb | Vanilla | Delta | Spend |
|---|---|---|---|---|---|---|
| Opus 5.5 | Triggers, should fire | 60 | 0.72 [0.61, 0.83] | n/a (kb only) | n/a | |
| Opus 5.5 | Triggers, near and noauto | 64 | 1.00 | n/a (kb only) | n/a | |
| Opus 5.5 | Triggers, all | 124 | 0.87 [0.80, 0.92] | n/a (kb only) | n/a | $48.37 |
| Opus 5.5 | Effects | 162 | 0.95 [0.92, 0.97] | 0.89 [0.85, 0.93] | +0.05 [+0.03, +0.08] | $360.48 |
| Opus 5.5 | Workflows | 4 | 1.00 | 1.00 | 0.00 | $7.33 |
| Sonnet 5.5 | Triggers, should fire | 60 | 0.60 [0.48, 0.71] | n/a (kb only) | n/a | |
| Sonnet 5.5 | Triggers, near and noauto | 64 | 1.00 | n/a (kb only) | n/a | |
| Sonnet 5.5 | Triggers, all | 124 | 0.81 [0.73, 0.87] | n/a (kb only) | n/a | $25.03 |
| Sonnet 5.5 | Effects | 162 | 0.93 [0.90, 0.96] | 0.88 [0.84, 0.92] | +0.05 [+0.02, +0.08] | $216.20 |
| Sonnet 5.5 | Workflows | 4 | 1.00 | 1.00 | 0.00 | $4.03 |

Per-case rows are in `evals/baselines/<model>/<suite>.json`. Cross-model, effects:

```bash
make eval-compare OLD=evals/baselines/claude-sonnet-5-5/effects.json NEW=evals/baselines/claude-opus-5-5/effects.json
```

**Reading the numbers**
- Triggers: near-miss and noauto cases sit at the ceiling, so the all-triggers mean overstates
  recall. The should-fire rows are the trigger signal: descriptions miss about 28% (Opus) and 40%
  (Sonnet) of requests they should catch.
- Effects: the gain is small, and it comes from cases where the asset fired. Where the kb arm's
  process grader fired at least once, delta is +0.075 (Opus, 124 cases) and +0.073 (Sonnet, 114);
  where it never fired, about -0.01 (38 and 48 cases), which is noise. The kb arm is the whole
  plugin, not only the named asset. `firedRate` averages every process
  grader, behavioral ones included, so this split is not by whether the asset loaded; see
  `tasks/kbeval-summarize-stale-results.md`.
- Effects headroom is thin: 130 of 162 cases have delta 0 on both models, and about 120 score 1.0 in
  both arms. The gain rests on about 25 cases. Grouping each asset's cases in the bootstrap leaves
  the delta CI above zero: [+0.02, +0.09] Opus, [+0.02, +0.08] Sonnet.
- Workflows show no headroom: plain Claude also completes all four, and n=4 gives no usable CI.
- Errors: 26 Opus and 20 Sonnet trigger runs hit the 6-turn cap or the 120 s timeout, and one Opus
  effect run hit 900 s; each case's `errorRuns` counts them. Capped runs are still scored: a run
  that fired the skill before the cap counts as fired.
- Five effect cases change pass state from Sonnet to Opus (four to pass, one to fail). With three
  runs and a 2/3 pass line, flips this few are within run-to-run noise.

## Asset determination

Which assets the baseline says to delete or fix (2026-11-16 baseline, KB 479cc63, 3 kb runs per
case per model). Method: `python3 -m evals.kbeval.determination` (`check`, `figures`, `spend`)
reads the raw per-run results under `~/.local/state/dotfiles-kbeval/results/<purpose>/kb/`
(machine-local, so it reruns only where those results exist), keeping only batches
with a `.key` and `partial` false, as `summary._load` does. Its checks are `html-lead`,
`writing-plans-risks`, `research-claim-c`, `j-explain-self-contained`, `arch-design`,
`no-regression`, and `library-use`; `figures` produced the counts below. Each llm-grader record
stores the graded file in `evidence`, so a failure can be read, not only counted.

**Verdict**: delete none; fix four assets and three graders; consult commands offer both their
references and delegation.

**Deletion.** No asset meets the skill-audit bar, "both unreferenced and unmentioned": every asset
is referenced. 130 of 162 effect cases have delta 0 on both models and about 120 score 1.0 in both
arms (see Reading the numbers): those cases have no headroom, which says nothing about the asset.
Every negative asset delta is a one- or two-run flip at 3 runs per arm.

**Fixes.**

| Asset | Defect | Baseline evidence (kb runs) | Change |
|---|---|---|---|
| html-report-authoring | Template puts `<h1>` then `nav.toc`, against its own BLUF rule | latency-readout `html` 3/6 (Opus 1/3, Sonnet 2/3); 0 of 6 graded files put a lead before the TOC | A `<p class="lead">` headline line between `<h1>` and the TOC |
| writing-plans | Header template `**Risks:**` overrides a prompt's "under a Risks heading" | Opus id-collision: `risks-section` 0/2 among runs that loaded the skill | `## Risks` |
| research-analyst | `confidence-scoring.md` flags "any contradiction" in one place and only unresolved ones in step 5 | claim-confidence `claims` 4/6 (Opus 2/3, Sonnet 2/3); both failing runs called the Claim C conflict resolved and did not flag it | Step 5 flags every contradiction, resolved ones included |
| skill-lookup-discipline | Claims a session-start hook triggers it; none exists (`.claude/settings.json` registers only `UserPromptSubmit`) | n/a (false statement) | Claim removed |

**Triggers.** Own Skill grader passes over trigger-should and effect kb runs, both models: 0/18 for
debugging-methodology, dispatching-parallel-agents, documentation-validation, executing-plans,
output-completeness, refactoring-and-debt, and subagent-report-contract; 0/6 for the j-arch,
j-debug, and j-devops commands. Each has another reach path: debugging-methodology through
`/j-debug`; dispatching-parallel-agents through `/j-plan` and `subagent-driven-development`;
documentation-validation through the `create-pr` and `documentation-writer` agents' `skills:`
preloads and the workflow commands; executing-plans through `/j-execute-plan` and `/j-next`;
output-completeness and subagent-report-contract through agent `skills:` preloads;
refactoring-and-debt through `/j-defrag`; the commands are typed. Descriptions stay unchanged:
within skills, description length does not predict firing (all 47-86 characters; 47-character
skills fire 6 of 6).

**Consult commands.** `delegated` passed in 0 of 12 kb runs for each of j-arch, j-cloud, j-devops,
j-docs, j-frontend, j-lang, and j-data (0 of 84), and in 0 of 12 for j-system-design. Stored graded
files cite `.kb-references/` in 1 of 12 j-cloud runs, 4 of 12 j-system-design runs, and none for the
others with evidence (j-lang has no llm grader, so no stored file); this is a lower bound on library
reads. The commands named their library only inside a conditional delegation sentence, a path the
model never took ("the file is 25 lines, so a direct review costs less than a handoff"). Each consult
command now lists its agent's reference index beside optional delegation, and j-system-design lists
one per perspective: [consult commands offer references and delegation](../docs/adr/workflow/consult-commands-offer-references-and-delegation.md).
Their cases gain a `read-refs` process grader (`tool_used` on `Read` of `<dir>` under `.kb-references/`, or `_refs/` under
`refsMode: add_dirs`).

**Measurement defects.**
- j-explain `self-contained` had no `match: not_contains`, so it passed only on external resources.
  Re-scored on stored files with the fix: 12/12 pass, against 0/12 stored.
- j-data `no-serial` matched SQL comments. Re-scored on stored files with the fix: Opus 3/3 (0/3
  stored), Sonnet 3/3 (3/3 stored).
- architecture-specialist `not-retried` rejected correct non-retry rows (Sonnet runs 0 and 1); its
  claim moves into the `design` llm grader.
- Case defects and harness gaps that need more than a grader fix:
  [kbeval-case-defects](../tasks/kbeval-case-defects.md),
  [kbeval-summarize-stale-results](../tasks/kbeval-summarize-stale-results.md).

### Post-fix rerun

Each change is committed, then re-measured on the kb arm under ledger purpose `fix-asset-defects`
and compared run by run with the stored baseline. `evals/baselines/` is not rewritten, so no summary
mixes KB versions. Criteria:
- `html-lead`: latency-readout `html` passes in at least 5/6 of runs (6 runs per model).
- `writing-plans-risks`: every Opus id-collision run that loaded the skill passes `risks-section`;
  fewer than 2 such runs is BLOCKED.
- `research-claim-c`: claim-confidence `claims` passes in at least 11/12 of runs.
- `j-explain-self-contained`: `self-contained` passes in at least 11/12 of runs across both cases.
- `arch-design`: retry-policy-adr `design` passes in at least 5/6 of runs.
- `no-regression`: per case, a kb score drop of at least 1/3 on both models, or at least 2/3 on one,
  fails; graders changed in this work are skipped.
- `library-use` (report only): consult and j-system-design runs that delegated, and runs that read
  references inline.

Results (2026-11-16, Claude Code 2.1.325 at every rerun, kb arm only, ledger purpose
`fix-asset-defects`). Baseline is KB 479cc63 at 3 runs per model; "KB" is the commit each rerun
staged on a clean tree. Counts are passing runs over graded runs.

| Asset | KB | Runs per model | Check | Baseline | Post-fix | Result |
|---|---|---|---|---|---|---|
| architecture-specialist | 123dbee | 3 | `design` (now carries the non-retry claim) | 6/6 (old claims) | 6/6 (Opus 3/3, Sonnet 3/3) | PASS |
| html-report-authoring | 43a4a49 | 6 | latency-readout `html` | 3/6 (Opus 1/3, Sonnet 2/3) | 10/12 (Opus 4/6, Sonnet 6/6) | PASS |
| html-report-authoring | 43a4a49 | 6 | lead before the TOC (report only) | 0/6 | 12/12 | |
| j-explain | 43a4a49 | 3 | `self-contained`, both cases | 0/12 | 12/12 | PASS |
| writing-plans | 56fe485 | 6, Opus only | id-collision `risks-section`, runs that loaded the skill | 0/2 | 5/5 | PASS |
| research-analyst | 617b357 | 6 | claim-confidence `claims` | 4/6 (Opus 2/3, Sonnet 2/3) | 11/12 (Opus 6/6, Sonnet 5/6) | PASS |

`self-contained` catches `src`/`href` loads, `@import`, and `url()`; it misses `srcset`,
`<object data>`, SVG `<image href>`, and script-issued requests, so 12/12 is a lower bar than the
skill's zero-network-requests rule (`tasks/kbeval-case-defects.md`).

`no-regression` passed for every rerun asset. Per-case kb scores that moved (baseline -> post-fix):
architecture-specialist catalog-cache-traps Opus 1.00 -> 0.67 (one model, under the 2/3 line);
html-report-authoring latency-readout Opus 0.83 -> 0.92, Sonnet 0.92 -> 1.00; writing-plans
id-collision-risk 0.67 -> 1.00; research-analyst claim-confidence Opus 0.67 -> 1.00, Sonnet
0.67 -> 0.83, seed-claim-review Opus 1.00 -> 0.83, Sonnet 0.67 -> 1.00. Every other case held its
score. Nothing was reverted or BLOCKED. The research-analyst rerun staged 617b357, which also holds
the skill-lookup-discipline sentence removal (no rerun; it adds no behavior).

Consult commands (ed0c8c5) and j-system-design (a2d8077), 3 runs per model: every case scored 1.00
before and after, on both models. `library-use` (post-fix only; `read-refs` is new):

| Command | Delegated | Read references inline, Opus | Read references inline, Sonnet |
|---|---|---|---|
| j-arch | 0/12 | 6/6 | 6/6 |
| j-cloud | 0/12 | 3/6 | 6/6 |
| j-devops | 0/12 | 0/6 | 5/6 |
| j-docs | 0/12 | 2/6 | 3/6 |
| j-frontend | 0/12 | 0/6 | 3/6 |
| j-lang | 0/12 | 2/6 | 3/6 |
| j-data | 0/12 | 0/6 | 4/6 |
| j-system-design | 0/12 | 3/6 | 3/6 |

The model still never delegates; with the list beside it, it reads the library inline in 49 of 96
runs (Opus 16 of 48, Sonnet 33 of 48). The change also softened "delegate to" to "you may instead
delegate to", so the reads are not credited to the list alone. No case score moved: these cases sit
at the ceiling, so the reads show reach, not a quality gain.

A review follow-up labeled j-system-design's lists by perspective and moved them below "Synthesize
their findings ... For small systems, analyze the perspectives inline instead." Rerun at 6faff20
(3 runs per model): scores held at 1.00, delegated 0/12, inline reads Opus 3/6, Sonnet 0/6 (from
3/6). Not significant at n=6, but the move was reverted; the table above is a2d8077's result, and
the shipped text differs from it only by naming `devops-engineer` in the opening delegation line.

Purpose spend: $89.71 of the $100 cap (estimate $84.60, plus $7.24 for the 6faff20 rerun). Period
total after the reruns: $868.54 of $2,500 (`evals/spend.json`). `evals/baselines/` is unchanged.
