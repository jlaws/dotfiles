---
name: writing-skills
description: "Use when creating, editing, or validating agent skills."
---

# Writing Skills

A skill exists to teach what a strong model does not already know. Everything else crowds out the
task it was loaded to help with.

**Codex reads `~/.agents/skills`, Gemini/Antigravity reads `~/.gemini/antigravity-cli/skills`, and Claude reads `~/.claude/skills`.** Each tree adapts its skill bodies and descriptions to its own harness while keeping workflow skill-set parity (and supporting-file parity between `.agents/skills/` and `.gemini/antigravity-cli/skills/`) in `tests/test_agent_config.py`.

## What Earns a Place in a Skill

Keep a section only if it passes one of these:

- The model would get it wrong or vague from memory: version-gated behavior, a non-obvious default
- It is a number or threshold with provenance, not a guess
- It is an opinionated choice that closes a decision the model would otherwise re-litigate
- It is specific to this repo: a real path, a real gotcha, a real convention

Cut what fails all four. Generic best practice, framework tours, and derivable taxonomies are the
usual offenders. A skill that could apply to any repo teaches little about this one.

Facts that defy a reasonable assumption go in a `## Gotchas` section of `SKILL.md`, not a supporting
file: the agent needs them before it knows to look. Add one each time a correction exposes a wrong
assumption.

## Claims and Evidence

A number in a KB asset is a claim the reader will act on. Every quantitative claim carries how it was
measured, or is marked unmeasured.

| Instead of | Write |
|---|---|
| "Tables are ~40% more efficient than prose" | "Tables are denser than prose (unmeasured here; the ratio depends on the content)" |
| "This cuts review time in half" | "Measured on <what>, <date>: <before> -> <after>" |

Distinguish a claim from a target. "min 80% coverage" is a gate and needs no provenance; "cuts tokens
2-3x" is a claim and does.

Say when a practice loses, not only when it wins. A rule with no stated cost reads as free, and the
reader stops looking for the cost. Where a technique has fixed overhead (a skill body preloaded on
every dispatch, a subagent's dispatch cost), say at what size the overhead exceeds the benefit.

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
stands, and credit its source in `REFERENCES.md`.

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
*length-matched naive control*: for a compressor, plain truncation to the same output size.

## The Four Shifts

| Instead of | Write |
|---|---|
| A rule | The goal and the reason, so judgment covers cases you did not foresee |
| Examples enumerating usage | An interface whose shape implies correct use |
| Everything upfront | A pointer, with the detail in a file that loads when needed |
| The same rule on three surfaces | One statement, in the place that owns it |

Rigid prohibitions, excuse tables, and all-caps "STOP" banners narrow judgment and fail on edge cases. Say what you want and why it matters.

## When to Create a Skill

**Create when** the technique was not obvious, it is reusable, and someone else would benefit.

**Do not create for** one-off solutions, well-documented standard practice, project conventions
(those go in `CLAUDE.md` / `AGENTS.md` / `GEMINI.md`), or anything a script can enforce: automate those instead.

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
Reserve the model for judgment. List a script's dependencies; do not assume they are installed.

Where a choice exists, give one default with an escape hatch rather than a menu of equal options.
Keep binding caps and stop conditions in the first screen of `SKILL.md` (`SKILL.md` under 20,000 bytes so compaction preserves it).

Word guidance to apply to the whole task, not one moment: "Run the tests after every edit", not "Run
the tests". A rule the agent must never skip belongs in a hook (`~/.agents/references/workflow/hook-patterns.md`),
not skill prose.

## Frontmatter

Skills under `.gemini/antigravity-cli/skills/` and `.agents/skills/` carry **only** `name` (kebab-case) and `description`:
- `.gemini/antigravity-cli/skills/` (Antigravity) descriptions use one quoted line (`<= 1,024` chars) without Codex's 64-character cap.
- `.agents/skills/` (Codex) descriptions must be **at most 64 characters** and start with `"Use when"` (`tests/test_agent_config.py`), with user-only command policy in `.agents/skills/cmd-j-*/agents/openai.yaml` (`policy.allow_implicit_invocation: false`).
- Claude-native skills under `.claude/skills/` use a single quoted description (`<= 200` chars) and may add scoped `allowed-tools`, `paths`, `disable-model-invocation`, or `context: fork`. No skill in any tree declares `skills:` in frontmatter; name dependencies in the body.

**Description = triggering conditions only.** Never summarize the workflow there; models follow a description shortcut instead of reading the body.

```yaml
# BAD: summarizes the workflow
description: "Use when executing plans - runs tasks sequentially with review"

# GOOD: just the trigger (<= 64 chars in .agents/)
description: "Use when executing a written plan in one working context."
```

## Discoverability & Flowcharts

- Name by what you do: `condition-based-waiting`, not `async-test-helpers`. Gerunds read well.
- Put searchable words in the description: symptoms, error text, tool names.
- **Cross-references:** name the skill in prose (`use the code-review-patterns skill`). Never use `@` links; they force-load and burn context.
- Use a flowchart only for a non-obvious decision, a process loop, or "when to use A vs B". Conventions and rendering: `graphviz-conventions.dot` and `render-graphs.js` in this directory.

## Test and Evaluate

Run the task without the skill and note where the agent actually goes wrong; write the skill to address those specific failures; re-run. If the agent read the skill and still chose wrong, inspect why (`references/CLAUDE_MD_TESTING.md`).

Classify the failure before choosing the fix: a skipped rule gets a prohibition with its reason; a wrong output shape gets a positive recipe that states the target; an omitted element gets a checklist item; a superseded form gets a mapping table near the top.

Change one thing per round when hillclimbing evals (`~/.agents/references/ai-ml/eval-and-benchmarking.md`, Hillclimbing). Process graders are indicators and stay out of the score; pin the model and judge across compared runs.

## Before You Ship

- [ ] Every section passes one of the four tests in "What Earns a Place in a Skill"
- [ ] Frontmatter in `.agents/` and `.gemini/` has only `name` and `description`, and `.agents/` fits `<= 64` chars starting with `"Use when"`
- [ ] Workflow skill sets and supporting files match between `.agents/skills/` and `.gemini/antigravity-cli/skills/`; reference twins expose the same ordered headings
- [ ] Commands have a `.agents/skills/cmd-j-*` skill, a Claude `.claude/commands/j-*.md` command, and a Gemini `.gemini/antigravity-cli/skills/j-*/SKILL.md` skill
- [ ] Each supporting file has a pointer in `SKILL.md` saying when to load it, and KB self-docs match (`documentation-validation`)
- [ ] The repo's lint and test commands pass
