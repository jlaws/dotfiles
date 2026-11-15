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
make eval-spend                      # free: snapshot ledger totals into evals/spend.json
```

`J ?= 8` sets `-j` (concurrent runs). `eval-baseline` also takes `VANILLA_RUNS` (default 3),
`VANILLA_MODELS` (default both models), and `EXCLUDE_TAGS` (default empty); the kb staging always
runs three times per case.

## Budget

`evals/budget.json` holds the period, its ceiling, and the smoke cap:

```json
{"period": "initial-baseline-2026-11", "ceilingUsd": 2500, "smokeCapUsd": 40}
```

A new period needs a committed bump that the user approves. The spend ledger is machine-local, at
`${XDG_STATE_HOME:-$HOME/.local/state}/dotfiles-kbeval/ledger.jsonl`, and every append holds an
exclusive lock. `make eval-spend` snapshots its per-period totals into the committed
`evals/spend.json`.

Each batch's `--max-cost-usd` cap is the remaining budget minus a reserve of `jobs x p95`, where
`p95` is the per-run cost on the `Reserve p95: $<usd>` line of the pilot table below ($5 before the
pilot). The pilot and smoke plans also cap their own spend: $10 for the pilot, `smokeCapUsd` per
branch for smoke. The CLI checks the cap
before each run starts, so in-flight runs can pass it; the reserve covers them. A batch with no
JSON result is charged its full cap. A batch whose result reports runs but a null or zero cost
aborts the run.

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
- `max_turns: 3`, `timeout_seconds: 120`, `allowed_tools: [Skill, Read, Glob, Grep]`.

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
- Optional `fixture.sh` gets the staged fixtures dir as `$1`. It may only copy from `$1` and run
  `git` or `mkdir` inside the working dir.
- It may not run `curl`, `wget`, `git clone`, `npm install`, `pip install`, or `ssh`, or touch
  absolute paths outside `$1`.
- Never write `context.scaffold_script`; the builder owns it.

**Limits.** `runs` 1-50, `max_turns` at most 200, `timeout_seconds` at most 3600. Only `EVAL_*` env
keys. No WebFetch or WebSearch. No `sk-ant-` strings. No network, secrets, or private project names:
the repo is public, and every input comes from the prompt or a fixture.

## Offline limits

What an offline run can show, per asset where it differs from a real session:

| Asset | Offline limit |
|---|---|
| j-create-pr | Branch, commit, and push to the local remote; the `gh pr create` attempt is graded as a call |
| j-rebase | The prompt supplies the PR branch |
| j-pr-fix | Comments arrive in `review-comments.json` |
| j-next | The prompt states that the previous PR merged |
| j-paper, j-explain | Local sources only |
| j-email-analysis | No downloads |

## Divergences

Absolute scores differ from a real session in these ways; comparisons within the harness hold:
- CLAUDE.md arrives as appended system text.
- References sit in the workspace at `./.kb-references/`, not under `~/.claude/references/`.
- Asset names carry the `kb:` namespace.
- No settings or hooks load.
- Bash runs under the OS sandbox.
- No network and no `gh`.

## Fixture specs

Frozen; cases depend on these exact contents.

| Fixture | Contents |
|---|---|
| `py-calc` | `calc/__init__.py`; `calc/ops.py` with `add(a, b)` and `mean(xs)`. The docstring says the mean of an empty list is `0.0`, but the body divides by `len(xs)` (the seeded bug). `tests/test_ops.py` (unittest) covers `add` and `mean([1, 2, 3])` only. README says `python3 -m unittest`. |
| `py-calc-red` | `py-calc` plus `tests/test_empty.py` asserting `mean([]) == 0.0`, which fails |
| `py-calc-planned` | `py-calc` plus `plans/plan.md`: a two-task plan to fix `mean([])` test-first and document it in README |
| `ts-todo` | `src/todo.js` (ESM `addTodo`, `complete`) and `test/todo.test.js` (`node --test`, covering `addTodo` only) |
| `docs-drift` | `tool.py` with an `--output` flag; README documents `--out` |
| `email-digest` | `inbox/2026-11-01-newsletter.txt`: plain text with five links and two paper titles |
| `paper-stub` | `paper/main.tex` skeleton with an empty Related Work section, plus `refs.bib` with three entries |
| `kb-mini` | `.claude/skills/broken-skill/SKILL.md` (missing `description`) and `.claude/commands/j-demo.md` |

**Git fixtures.** `fixture.sh` copies `$1/<name>/.`, runs `git init -q -b main`, sets repo-local
`user.name kbeval` and `user.email kbeval@example.invalid`, and makes the initial commit. It then
creates the remote with `git init -q --bare .remote.git`, `git remote add origin ./.remote.git`,
and `git push -q origin main`, and lists `.remote.git` and `.kb-references` in
`.git/info/exclude`.

## CLI surface

Checked on `2.1.325 (Claude Code)`: every flag the runner passes exists in
`claude plugin eval --help`.

## Pilot measurements

Empty until the pilot runs.

## Baseline

Empty until the baseline runs.
