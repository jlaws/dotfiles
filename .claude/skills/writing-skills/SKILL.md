---
name: writing-skills
description: "Use when creating, editing, or validating agent skills."
---

# Writing Skills

A skill exists to teach what a strong model does not already know. Everything else crowds out the
task it was loaded to help with.

**Codex reads `~/.agents/skills`, Gemini/Antigravity reads `~/.gemini/antigravity-cli/skills`, and Claude reads `~/.claude/skills`.** The trees have
diverged on purpose: `.claude/` is written for the Claude 5 generation, `.agents/skills/` serves Codex (GPT-6.2), and `.gemini/antigravity-cli/skills/`
serves Google Antigravity (Gemini 4). Parity tests pin which skills and supporting files exist in each tree, not identical prose.

## What earns a place in a skill

Keep a section only if it passes one of these:

- The model would get it wrong or vague from memory: version-gated behavior, a non-obvious default
- It is a number or threshold with provenance, not a guess
- It is an opinionated choice that closes a decision the model would otherwise re-litigate
- It is specific to this repo: a real path, a real gotcha, a real convention

Cut what fails all four. Generic best practice, framework tours, and derivable taxonomies are the
usual offenders. A skill that could apply to any repo teaches little about this one.

Facts that defy a reasonable assumption go in a `## Gotchas` section of SKILL.md, not a supporting
file: the agent needs them before it knows to look. Add one each time a correction exposes a wrong
assumption.

## Claims and evidence

A number in a KB asset is a claim the reader will act on. Every quantitative claim carries how it was
measured, or is marked unmeasured.

| Instead of | Write |
|---|---|
| "Tables are ~40% more efficient than prose" | "Tables are denser than prose (unmeasured here; the ratio depends on the content)" |
| "This cuts review time in half" | "Measured on <what>, <date>: <before> -> <after>" |

Distinguish a claim from a target. "min 80% coverage" is a gate and needs no provenance; "cuts tokens
2-3x" is a claim and does.

Say when a practice loses, not only when it wins. A rule with no stated cost reads as free, and the
reader stops looking for the cost. Where a technique has fixed overhead: a skill body preloaded on
every dispatch, a subagent's dispatch cost: say at what size the overhead exceeds the benefit.

Numbers from a third party are that party's claim, not yours. Leave them out of the asset: state the
rule, mark it "reported elsewhere, unmeasured here" where the reader needs to know, and credit the
source in `REFERENCES.md`.

**A counterfactual is not a measurement.** "This saved N tokens" about work you did is unknowable;
the unwritten version was never written, so there is no baseline to subtract from. Either run a
holdout (leave a fraction of the work unshaped and compare) or label the figure an estimate and give
its range. Naming a saving you cannot have observed is the most common way a real technique acquires
a fake number.

**Link the counter-evidence.** Where someone has published a measurement against a claim you are
making, state the objection beside the claim, say which part survives your fix and which part
stands, and credit its source in `REFERENCES.md`. A claim with the strongest objection stated beside
it is more trustworthy than one without, and a reader who finds the objection elsewhere first will
discount everything around it.

**Retract in place.** A figure that stops reproducing gets struck where it was published, with what
replaced it: not quietly deleted. Deleting it hides that it was ever believed, which is the thing a
later reader most needs to know.

**A feature is a claim too.** Behavior shipped on a hunch carries the same burden as a number:
either evidence that it fires, or the label. Say it is speculative and say what would retire the
label. The opposite failure is worse and quieter: a rule the repo advertises but nothing enforces.

**Cherry-picking is only dishonest when it is silent.** A best-case example is a fair answer to
"what does this look like when it works", provided the general number sits next to it.

**If you do build a measurement, two things make it defensible.** Include cases where the technique
should *not* win, and fail loudly if it starts claiming a win there; a suite of only favourable
inputs proves the fixtures were chosen well, not that the technique works. And compare against a
*length-matched naive control*: for a compressor, plain truncation to the same output size. Beating
nothing is easy; beating the dumbest thing that produces the same volume is the real question.

## The four shifts

| Instead of | Write |
|---|---|
| A rule | The goal and the reason, so judgment covers cases you did not foresee |
| Examples enumerating usage | An interface whose shape implies correct use |
| Everything upfront | A pointer, with the detail in a file that loads when needed |
| The same rule on three surfaces | One statement, in the place that owns it |

Rigid prohibitions are usually wrong for some real situation, and the model cannot notice that if the
skill also forbids noticing. Say what you want and why it matters.

## When to create a skill

**Create when** the technique was not obvious, it is reusable, and someone else would benefit.

**Do not create for** one-off solutions, well-documented standard practice, project conventions
(those go in CLAUDE.md), or anything a script can enforce: automate those instead.

## Structure

```
skills/
  skill-name/
    SKILL.md              # required
    scripts/              # mechanical steps; call these instead of describing them
    references/           # heavy detail, loaded on demand
```

Split a file out when it is heavy reference or a reusable tool. Keep the rest inline, one level deep.
A pointer says when to load the file: "Read `references/CLAUDE_MD_TESTING.md` when an agent read the
skill and still chose wrong", not "see references/".

**For mechanical steps (validation, formatting, deterministic checks) ship a script and call it.**
Reserve the model for judgment. A prose checklist of things a script could verify is both longer and
less reliable. List a script's dependencies; do not assume they are installed.

Where a choice exists, give one default with an escape hatch rather than a menu of equal options.

Keep binding caps and stop conditions in the first screen of SKILL.md. After auto-compaction Claude Code
re-attaches only the first 5,000 tokens of each invoked skill, within a 25,000-token budget shared by all
re-attached skills, and `audit.py` SK-C6 warns when a SKILL.md passes 20,000 bytes.

Word guidance to apply to the whole task, not one moment: "Run the tests after every edit", not "Run
the tests". A rule Claude must never skip belongs in a hook (`${CLAUDE_SKILL_DIR}/../../references/workflow/hook-patterns.md`),
not skill prose.

## Frontmatter

`name` (letters, numbers, hyphens) and `description` are the essentials. Others worth knowing:

| Field | Use |
|---|---|
| `allowed-tools` | Pre-approves tools for the invoking turn only and restricts nothing; grant only a command scoped to one named script, such as `Bash(python3 ${CLAUDE_SKILL_DIR}/scripts/<name>.py *)`, never bare `Bash`, `Write`, `Edit`, `NotebookEdit`, `WebFetch`, or `WebSearch`, nor a wildcard-only scope such as `Bash(*)` |
| `disallowed-tools` | Removes tools while the skill is active |
| `model`, `effort` | Override tier or reasoning depth; tier aliases float across model generations. Set `model` only with `context: fork`: inline, a different model re-reads the conversation uncached |
| `paths` | Glob-gate the skill so it loads only for matching files |
| `disable-model-invocation` | Keep it out of the always-loaded listing; user-invocable only. Also blocks subagent preloading and scheduled runs |
| `context: fork`, `agent` | Run in an isolated subagent context |
| `argument-hint`, `arguments` | Autocomplete and `$name` substitution |

`license` and `compatibility` are agentskills.io fields Claude Code accepts but does not act on.

`paths` and `disable-model-invocation` are the progressive-disclosure levers: they decide whether a
skill costs context on every turn or only when it is relevant.

**Description = triggering conditions only.** Never summarize the workflow there; testing showed
Claude follows a description shortcut instead of reading the body. Codex skills under `.agents/`
carry a hard 64-character budget and must start with "Use when", enforced by `tests/`.
A `.claude` skill description is one quoted line of at most 200 characters, and a `.gemini/antigravity-cli` skill description is one quoted line of at most 1,024 characters (`tests/`). A description
that misses its should-fire cases is tuned against `tune-` trigger cases and adopted only on `held-`
cases (`docs/adr/workflow/skill-descriptions-are-tuned-against-held-out-triggers.md`).

```yaml
# BAD: summarizes the workflow
description: Use when executing plans - executes tasks sequentially with review between tasks

# GOOD: just the trigger
description: Use when executing implementation plans with independent tasks
```

## Discoverability

- Name by what you do: `condition-based-waiting`, not `async-test-helpers`. Gerunds read well.
- Put searchable words in the description: symptoms, error text, tool names.
- Prefer concrete triggers over language-specific ones.

**Cross-references:** name the skill in prose (`use the code-review-patterns skill`). Never use `@`
links; they force-load and burn context.

## Flowcharts

Use one only for a non-obvious decision, a process loop, or "when to use A vs B". Never for reference
material, code examples, or linear instructions. Conventions and rendering: `graphviz-conventions.dot`
and `render-graphs.js` in this directory.

## Test it against a real agent

A skill you have not watched an agent use is a guess. Run the task without the skill and note where
the agent actually goes wrong; write the skill to address those specific failures; re-run.

If the agent read the skill and still chose wrong, ask it why. "The skill was clear, I ignored it"
means the principle is not doing any work. "It should have said X" is a direct edit. "I missed
section Y" is a structure problem. Detailed methodology: `references/CLAUDE_MD_TESTING.md`.

When wording shapes behavior, read it back cold a few times. If it supports several readings, rewrite
until they converge.

Resist answering every failure with a stronger prohibition. When an agent skips a step, the usual
cause is that the skill never said why the step mattered.

Classify the failure before choosing the fix. A skipped rule gets a prohibition with its reason. A wrong output shape gets a positive recipe that states the target, because a "don't" names the unwanted shape and invites it. An omitted element gets a checklist item. Every step ends in a completion criterion that is checkable and exhaustive; a vague bound invites stopping early.

When the agent writes a superseded form although the skill states the current one, add a table near the top mapping the remembered form to the current form.

## Eval triggering and effect

Test two things separately. **Triggering**: write should-trigger and near-miss prompts in users'
words, never naming the skill; fix misses through `description`, or `disable-model-invocation` for a
skill only the user should start. **Effect**: run the same tasks with and without the skill. Grade
each task with one result grader (the produced file or answer) and one process grader (the skill
loaded, or the expected tool ran). Change one thing per round
(`${CLAUDE_SKILL_DIR}/../../references/ai-ml/eval-and-benchmarking.md`, Hillclimbing).

Process graders are indicators and stay out of the score. A should-not-fire check is a Skill `tool_used` grader with `min: 0` and `max: 0`. When the skill loaded but the delta is negative, suspect the judge before the skill. Pin the model and the judge for any series you compare.

Tooling: skill-creator evaluates a single skill. `claude plugin eval` needs a plugin manifest or a skills-directory plugin,
isolates each run, and bills each run. To evaluate a plain skills tree, stage it into a temporary
plugin: copy skills, commands, and agents; add `.claude-plugin/plugin.json`; add a scaffold script
that copies references into each run's workspace.

## Before you ship

- [ ] Every section passes one of the four tests above
- [ ] Mechanical steps live in a script, not in prose
- [ ] `description` is a trigger, and any `.agents/` copy fits the 64-character budget
- [ ] Each supporting file has a pointer in SKILL.md that says when to load it
- [ ] KB self-docs current: the MEMORY.md index if the asset set changed
- [ ] Commands have a canonical `.agents/skills/cmd-j-*` skill for Codex, a Claude `/j-*` command, and a Gemini `j-*` skill; no deprecated Codex prompt copy
- [ ] The repo's lint and test commands pass
