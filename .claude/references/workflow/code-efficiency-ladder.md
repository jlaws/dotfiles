# Code Efficiency Ladder

A stopping rule for deciding whether code should exist before writing it. Where
`completeness-principle` governs how thoroughly to build what is in scope, this governs whether a
thing belongs in scope at all.

Adapted from [ponytail](https://github.com/DietrichGebert/ponytail) (`skills/ponytail/SKILL.md`),
which is the origin of the seven-rung form. Its rung 2 was added on 2026-06-22; anything carrying
seven rungs with reuse at position 2 derives from that version or later.

## The Ladder

Stop at the first rung that holds:

1. **Does this need to exist at all?** Speculative need = skip it, and say so in one line. (YAGNI)
2. **Already in this codebase?** A helper, util, type, or pattern that already lives here — reuse it. Look before you write; re-implementing what is a few files over is the most common slop.
3. **Stdlib does it?** Use it.
4. **Native platform feature covers it?** `<input type="date">` over a picker lib, CSS over JS, a DB constraint over app code.
5. **Already-installed dependency solves it?** Use it. Never add a new one for what a few lines can do.
6. **Can it be one line?** One line.
7. **Only then:** the minimum code that works.

Two rungs work? Take the higher one and move on.

## It Runs After Understanding, Never Instead Of It

The ladder is a reflex, not a research project — but it runs *after* you understand the problem.
Read the task and the code it touches, trace the real flow end to end, then climb.

**This is the carve-out that matters most.** Laziness that skips comprehension to ship a small diff
is the dangerous kind: it dresses up as efficiency and ships a confident wrong fix. The ladder
shortens the solution, never the reading. The smallest change in the wrong place is not lazy, it is
a second bug.

**Bug fix = root cause, not symptom.** A report names a symptom. Before editing, grep every caller
of the function you are about to touch. The root-cause fix *is* the smaller diff — one guard in the
shared function beats one guard in every caller, and patching only the path the ticket names leaves
every sibling caller broken. That framing is deliberate: it points the laziness reflex at the root
cause rather than away from it.

## Correctness Beats Brevity At Equal Size

**Two options the same size? Take the one that is correct on edge cases.** Lazy means writing less
code, not picking the flimsier algorithm.

This rule is load-bearing, not a platitude. It exists because the ladder has a measured failure
mode: reaching for the laziest stdlib helper can pick one with an edge-case hole. ponytail's own
audit found `parseaddr` accepted `"@missing-local.com"` because it does not require a local part,
and reported email-validation accuracy dropping from 100% to 79% on one non-Claude model. Its
diagnosis: "a genuine (if minor) cost of pushing toward one-liners."

## Never Simplify Away

- Input validation at trust boundaries
- Error handling that prevents data loss
- Security measures
- Accessibility basics
- Anything explicitly requested

If the user insists on the full version, build it — no re-arguing. These five are why the ladder is
safe to run; ponytail measured a paraphrase that dropped them scoring 95% safe against the full
ruleset's 100%.

## No Unrequested Abstractions

| Do not add | Threshold |
|---|---|
| An interface | One implementation |
| A factory | One product |
| A config value | It never varies |
| Scaffolding "for later" | No current caller |

Deletion over addition. Boring over clever — clever is what someone decodes at 3am. Fewest files
possible.

## Thinking Is Billed Too

Reasoning tokens cost the same as written ones. Every rule above trims what you write; this trims
what you spend getting there.

The ladder is a stopping rule, not a checklist to walk aloud. Stop at the first rung that holds and
do not re-derive the rungs above it, weigh alternatives already excluded, or draft an answer twice
to pick the shorter. Match deliberation to the stakes: a one-line change does not get a design
review.

This has receipts. ponytail's first version lost to a prose-only competitor on wall-clock time
(228s vs 136s) because it "deliberated about what not to build"; an anti-deliberation clause cut
that to 158s, and trimming the ruleset itself to 127s.

The exception is the same as the ladder's: never think less about *understanding* the problem.
Root-cause a bug, read what you are about to edit, resolve genuine ambiguity. Depth where the
problem is actually hard, nowhere else.

## Measured Scope

ponytail's own agentic benchmark, which is the strongest evidence available for this ruleset:
12 feature tasks plus 6 safety tasks against a real repo
(`tiangolo/full-stack-fastapi-template @ cd83fc1`), Haiku 4.5, n=4 per cell, LOC counted as
`git diff` added lines, safety checked deterministically rather than by an LLM judge.

| Arm | LOC | Tokens | Cost | Time | Safe (20 runs) |
|---|--:|--:|--:|--:|--:|
| Prose compression only | -20% | +7% | +3% | +2% | 100% |
| The full ladder | **-54%** | **-22%** | **-20%** | **-27%** | **100%** |
| A "YAGNI + one-liners" paraphrase | -33% | -14% | -21% | -30% | 95% |

These are ponytail's numbers, not measured here. Three limits it states itself, which decide what
they mean:

- **The win is on the code axis.** A separate benchmark measured this ruleset at 44% of a bare
  model on coding prompts but 87% on explanation-only prompts. The ladder needs an abstraction to
  skip or a stdlib call to reach for; it has nothing to bite on in prose.
- **It can cost more than it saves on short work, and on reasoning models.** ponytail measured
  +26.2% and +38.7% cost on two reasoning models — "the ruleset is re-sent as input every call and
  the baseline output is already terse, so the input and reasoning-token overhead outweighs the
  lines saved." On a small local model the aggregate flipped sign depending on the sample.
- **One model, deterministic safety checks on six tasks.** A floor that shows whether a known guard
  gets dropped, not proof that the code is secure.

## Cross-References

- **reference:completeness-principle** — the other half: thorough about quality dimensions, lazy about new scope
- **skill:code-quality** — the smells this ladder prevents, and the `// SIMPLIFIED:` marker for shortcuts it sanctions
- **skill:refactoring-and-debt** — removing what was already built
- **agent:scope-reviewer** — rung 1 at the feature level rather than the code level
- **reference:context-efficiency** — the same economy applied to what enters context rather than what gets built
