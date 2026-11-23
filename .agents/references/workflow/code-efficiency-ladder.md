# Code Efficiency Ladder

A stopping rule for deciding whether code should exist before writing it. Where
`completeness-principle` governs how thoroughly to build what is in scope, this governs whether a
thing belongs in scope at all.

## The Ladder

Stop at the first rung that holds:

1. **Does this need to exist at all?** Speculative need = skip it, and say so in one line. (YAGNI)
2. **Already in this codebase?** A helper, util, type, or pattern that already lives here: reuse it. Look before you write; re-implementing what is a few files over is the most common slop.
3. **Stdlib does it?** Use it.
4. **Native platform feature covers it?** `<input type="date">` over a picker lib, CSS over JS, a DB constraint over app code.
5. **Already-installed dependency solves it?** Use it. Never add a new one for what a few lines can do.
6. **Can it be one line?** One line.
7. **Only then:** the minimum code that works.

Two rungs work? Take the higher one and move on.

## It Runs After Understanding, Never Instead Of It

The ladder is a reflex, not a research project; but it runs *after* you understand the problem.
Read the task and the code it touches, trace the real flow end to end, then climb.

**This is the carve-out that matters most.** Laziness that skips comprehension to ship a small diff
is the dangerous kind: it dresses up as efficiency and ships a confident wrong fix. The ladder
shortens the solution, never the reading. The smallest change in the wrong place is not lazy, it is
a second bug.

**Bug fix = root cause, not symptom.** A report names a symptom. Before editing, grep every caller
of the function you are about to touch. The root-cause fix *is* the smaller diff: one guard in the
shared function beats one guard in every caller, and patching only the path the ticket names leaves
every sibling caller broken. That framing is deliberate: it points the laziness reflex at the root
cause rather than away from it.

## Correctness Beats Brevity

**Two options the same size? Take the one that is correct on edge cases.** Lazy means writing less
code, not picking the flimsier algorithm.

**Where they are not the same size, a trust boundary still takes the correct one.** The rungs stop
at the first option that works, and the cheaper option is usually the weaker one, so without this
the ladder selects the flimsier algorithm in exactly the case below. Validation, parsing of
untrusted input, and auth are decided on correctness first and length second.

This rule is load-bearing, not a platitude. Reaching for the laziest stdlib helper can pick one
with an edge-case hole: a stdlib one-liner can accept input the longer check rejected
(`email.utils.parseaddr("@missing-local.com")` returns an address with no local part).

## Never Simplify Away

- Input validation at trust boundaries
- Error handling that prevents data loss
- Security measures
- Accessibility basics
- Anything explicitly requested

**Weakening counts as removing.** Swapping a control for a laxer implementation of the same control
simplifies it away even though the line count barely moves.

If the user insists on the full version, build it: no re-arguing. These five are why the ladder is
safe to run.

## No Unrequested Abstractions

| Do not add | Threshold |
|---|---|
| An interface | One implementation |
| A factory | One product |
| A config value | It never varies |
| Scaffolding "for later" | No current caller |

Deletion over addition. Boring over clever: clever is what someone decodes at 3am. Fewest files
possible.

## Thinking Is Billed Too

Reasoning tokens cost the same as written ones. Every rule above trims what you write; this trims
what you spend getting there.

The ladder is a stopping rule, not a checklist to walk aloud. Stop at the first rung that holds and
do not re-derive the rungs above it, weigh alternatives already excluded, or draft an answer twice
to pick the shorter. Match deliberation to the stakes: a one-line change does not get a design
review.

The exception is the same as the ladder's: never think less about *understanding* the problem.
Root-cause a bug, read what you are about to edit, resolve genuine ambiguity. Depth where the
problem is actually hard, nowhere else.

## Where It Loses

- **The win is on the code axis.** The ladder needs an abstraction to skip or a stdlib call to
  reach for, so it bites hardest on code and least on explanation-only prompts, where the prose
  half of the pairing does the work instead.
- **It can cost more than it saves on short work and on reasoning models.** A ruleset re-sent on
  every call costs input tokens, and reasoning models add reasoning-token overhead (reported
  elsewhere, unmeasured here), so it can lose on short tasks (`references/workflow/context-efficiency.md`, Output Rules
  Earn Their Place Or Cost You).

## Cross-References

- **reference:completeness-principle**: the other half; thorough about quality dimensions, lazy about new scope
- **skill:code-quality**: the smells this ladder prevents, and the `// SIMPLIFIED:` marker for shortcuts it sanctions
- **skill:refactoring-and-debt**: removing what was already built
- **agent:scope-reviewer**: rung 1 at the feature level rather than the code level
- **reference:context-efficiency**: the same economy applied to what enters context rather than what gets built
