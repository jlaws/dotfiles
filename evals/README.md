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
├── kbeval/                     # stage, lint, run, summarize, compare
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
  `"subagent_type"\s*:\s*"kb:<agent>"`, or `tool_order`. Process graders are indicators; the summary
  leaves them out of delta. They reference main-loop tool calls only, never a subagent's own calls.
- Effects: `max_turns: 40`, `timeout_seconds: 900`,
  `allowed_tools: [Skill, Read, Glob, Grep, Agent, TodoWrite]`.
- Workflows: `max_turns: 150`, `timeout_seconds: 3600`, same tools.

**Fixtures**
- Optional `fixture.sh` gets the staged fixtures dir as `$1` and starts with `set -euo pipefail`. It
  copies from `$1` and builds local state in the working dir with `git`, `mkdir`, and `printf`. The
  seed runs it with the user's global and system git config switched off.
- Lint rejects network and exec commands (`curl`, `wget`, `ssh`, `scp`, `nc`, `sudo`, `brew`,
  `eval`, `git clone`, `npm install`/`i`/`ci`, `pip install`, `rm -r`, `sh -c`, `python -c`,
  `git remote add` to a URL), absolute paths outside `$1`, and `..` paths.
- Never write `context.scaffold_script`; the builder owns it. `case.yaml` holds only `context:` with
  a flow-list `add_dirs: [..]` or a `history_file:`.
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
- `llm` graders hold 2-5 `PASS if` / `FAIL if` claims.
- Grounding: every graded path is named in the prompt or seeded by the fixture. A result grader
  that already passes on the seeded files, or a `file_exists` with `exists: false`, is rejected,
  because a run that does nothing would score it.
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
- Inside the Bash sandbox the `/usr/bin/git` shim fails; Homebrew's `git` on `PATH` works.
- The child session inherits `ANTHROPIC_API_KEY`, so a run granted Bash could print it. The
  sandbox blocks network, and the key scan covers results, logs, and kept traces; traces of runs
  without `--keep-temp` are deleted by the CLI unscanned.

## Fixture specs

The specs are frozen; cases depend on these exact contents. Each fixture ships with the case PR
that first uses it; `py-calc`, `py-calc-red`, `ts-todo`, `docs-drift`, and `kb-mini` ship so far.
A test pins `py-calc`'s seeded bug.

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
`vanillaScore`, `delta`, `firedRate`, run counts, and errors. Scores use result graders only;
`firedRate` is the process-grader pass rate in kb. A case `passed` when `kbScore`, its mean run score,
is at least 2/3. Suite aggregates carry a 95% bootstrap interval over cases (seed 0).
It reads only results the runner accepted (those with a `.key`), and it checks before writing: if
any case is missing runs or still holds an API error, it writes nothing and exits 1 (`--check` is
accepted and changes nothing).

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

**Extrapolation** (a dated estimate, 2026-11-15; recompute it from `make eval-lint` case counts
before the baseline). Planned cases, from the coverage rules: about 124 trigger cases, about 187
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

Empty until the baseline runs.
