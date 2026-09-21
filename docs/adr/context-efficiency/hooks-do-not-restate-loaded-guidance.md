---
status: accepted
topic: context-efficiency
created: 2026-09-21
updated: 2026-09-21
deciders: ["@jlaws"]
---
# Hooks do not restate always-loaded guidance

## Context

Two hooks existed to repeat, at runtime, guidance the harness already sends on every request.

`.claude/hooks/guard-bash-output.sh` (and its byte-identical `.codex/` twin) ran on every Bash call,
tokenized the command, and emitted an advisory when it judged the output unbounded -- "Unbounded
'git log' -- prefer 'git log --oneline -20'", "'cat' of a whole file -- read a bounded range with
'sed -n "START,ENDp"' instead", and five more. Claude Code renders that as a `PreToolUse:Bash says`
line; Codex renders it through `hookSpecificOutput.additionalContext`.

A `SessionStart` hook in `.claude/settings.json` printed a `SKILL-LOOKUP DISCIPLINE:` banner at the
top of every session, naming three process skills and telling the reader to announce a skill before
acting.

Both were advisory and neither could block. Both restated text already in `.claude/CLAUDE.md`, which
is re-sent on every request in every repo: the Bash section says to cap output and make each failure
attributable, and the Knowledge base section says to check `skills/` before implementing.

## Decision Drivers

* **A second delivery channel for the same rule is pure cost** -- the first one already fired.
* **Precision is what makes an advisory worth reading.** A guard that nags about a bounded command
  trains the reader to skip it, and once skipped it protects nothing.
* **The always-loaded configs are the single owner of standing guidance.** That is the pattern
  `structure-is-conditional` established for prose rules; it should hold for hooks too.

## Decision

Hook output is for information the model does not already have. Guidance that already lives in an
always-loaded config does not get a second delivery channel.

Concretely: the repo registers no `PreToolUse` hook and no `SessionStart` hook, and ships no
`guard-bash-output.sh`. `UserPromptSubmit` / `log-prompt.sh` stays -- it writes a log to disk and
emits nothing into the conversation, so it is not subject to this rule.

## Rationale

The guard's own header called precision "the whole game" and argued a false positive costs more than
a miss. That is the right standard, and the ladder did not clear it: `cat` fired on every whole-file
read regardless of file size, `grep`/`rg` fired unless the invocation carried `-l`, `-c`, or `-m`,
and `pytest` fired on any unscoped run. All three are ordinary, deliberate commands. Tightening
those rules would have bought a quieter guard that still told the reader something `CLAUDE.md`
already told them.

The banner has the same shape at lower volume: correct content, zero new information, fired once per
session forever.

Note what this does not say. A hook carrying state the model genuinely lost -- the
`PreCompact`-writes / `SessionStart`-reads snapshot pattern in
`references/workflow/hook-patterns.md` -- is exactly the case the Decision leaves open.

## Consequences

**Gained**: No unsolicited hook output in the conversation. About 600 lines of bash plus a 544-line
test module leave the maintained surface. One owner per rule.

**Accepted**: Neither nudge is enforced at call time any more. The bounded-command guidance lives
only as prose in `references/workflow/context-efficiency.md` and the Bash section of `CLAUDE.md`;
the skill-lookup reminder lives only as the Knowledge base line in `.claude/CLAUDE.md`, which is
terser than the bullet `.gemini/GEMINI.md` carries. If the model ignores prose, nothing catches it.

## Ruled Out

| Idea | Why ruled out | When |
|------|---------------|------|
| Narrow the guard to `git log` and `git diff` | Keeps a script, a `--format` flag, and a `make verify` surface alive to restate one line of `CLAUDE.md`. The maintenance is the cost, not the rule count. | 2026-09-21 |
| Unwire the guard but keep the script | Leaves ~600 lines of dead bash and inverts its tests from "is registered" to "is not registered" -- a dormant feature that reads as a live one. | 2026-09-21 |
| Keep the `SessionStart` banner because it is short | Length is not the objection. Duplication is, and a short duplicate is still a duplicate. | 2026-09-21 |

## Enforcement
- `tests/test_agent_config.py::test_no_hook_restates_always_loaded_guidance` pins that
  `.claude/settings.json` declares `UserPromptSubmit` and nothing else, that `.codex/config.toml`
  declares no `[[hooks.PreToolUse]]`, and that neither hook tree ships `guard-bash-output.sh`.
- `macos_setup/dotfiles.py`'s `AGENT_REMOVALS` deletes the installed copies under `~` on the next
  `setup.sh -c`, archived first so `--uninstall` can restore them.

## Reversal Conditions

Reverse if either nudge turns out to have been load-bearing:

- The bounded-command half, if this repo's own transcripts show unbounded output measurably eating
  context again. `.claude/skills/skill-audit/scripts/adoption.py` already has the corpus access.
- The skill-lookup half, if the adoption report shows process-skill mentions dropping after this
  lands.

`git log -- .claude/hooks/guard-bash-output.sh` has the guard implementation;
`git log -- .claude/settings.json` has the banner text.

## Related
- [Structure preference is conditional on what the question asks for](structure-is-conditional.md)
- [ADRs are living documents](../workflow/adrs-are-living-documents.md)
