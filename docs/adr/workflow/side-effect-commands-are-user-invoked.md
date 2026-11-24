---
status: accepted
topic: workflow
created: 2026-11-14
updated: 2026-11-24
---
# Side-effect commands are user-invoked

## Context

Task skills with side effects should be user-invoked, so the model does not start them on its own.
In observed sessions Claude invokes j-plan, j-next, j-rebase, j-create-pr, and j-brainstorm through
the Skill tool, and skills chain to j-diff-review and j-execute-plan. Nothing invokes j-pr-fix,
j-tdd, j-defrag, or j-review-claudemd. Codex ignores `disable-model-invocation` and reads
`agents/openai.yaml` instead. Antigravity documents no invocation control.

## Decision Drivers
* **Must not break observed flows**: commands the model or a skill invokes stay invocable
* **Should match across harnesses** where a harness offers a control

## Decision

We will make **exactly j-pr-fix, j-tdd, j-defrag, and j-review-claudemd** user-invoked:
- Their `.claude/commands/` files set `disable-model-invocation: true`.
- Their `.agents/skills/cmd-j-*` twins carry `agents/openai.yaml` with implicit invocation off.
- No `.agents` skill keeps the inert `disable-model-invocation` key.

## Rationale

Only commands nothing invokes are flagged, so no chain breaks. Codex gets the same set through its
own control.

## Consequences
**Gained**: the model cannot start these four side-effect commands unprompted.
**Accepted**: Claude reaches the same logic through `pr-comment-resolution`,
`test-driven-development`, and `code-quality`. The j-defrag logic has no measured path: at the
2026-11-16 baseline `refactoring-and-debt` fired in none of its 18 kb runs (6 trigger-should, 12
effect, both models), so no run has shown Claude reaching it unprompted. j-review-claudemd runs
only when typed. Other Codex command skills stay implicitly invocable, as they already were in practice.

## Ruled Out
| Idea | Why ruled out | When |
|------|---------------|------|
| Flag every command except the chain targets | Breaks the observed Skill-tool flows | 2026-11-14 |
| Flag none | Leaves unused side-effect commands model-invocable | 2026-11-14 |
| Keep the inert `.agents` key | No harness reads it | 2026-11-14 |

## Enforcement
- Owner: `.claude/commands/*.md` frontmatter and `.agents/skills/cmd-j-*/agents/openai.yaml`
- Pinned by: `tests/test_agent_config.py::AgentConfigArchitectureTests::test_side_effect_commands_nothing_invokes_are_user_only` and `::test_codex_user_only_commands_disable_implicit_invocation`

## Reversal Conditions
- A skill or coordinator starts invoking one of the four.
- A harness changes its invocation semantics.

## Related
- [Skills do not pre-approve tools](skills-do-not-pre-approve-tools.md)
