---
status: accepted
topic: workflow
created: 2026-11-14
updated: 2026-11-14
---
# Skills do not pre-approve tools

## Context

`allowed-tools` in a skill or command pre-approves the listed tools for the invoking turn. It
restricts nothing: every other tool stays available under the normal permission rules. Skills in
this KB set it believing it scoped their tools. Most granted bare `Bash`, some granted `Write` or
`Edit`, and one granted `WebFetch` and `WebSearch` beside `Bash`. Each grant silently widened what
ran without a prompt while the skill was active.

## Decision Drivers
* **Must not widen permissions by accident**: a skill load should never auto-approve arbitrary shell commands or writes
* **Should keep one owner for pre-approval**: `.claude/settings.json` already holds the allowlist

## Decision

We will keep **pre-approval out of skills and commands**:
- No Claude skill or command grants an unscoped `Bash`, `Write`, `Edit`, `NotebookEdit`, `WebFetch`, or `WebSearch`.
- Skills drop `allowed-tools`, except skill-audit, which keeps `Bash(python3 ${CLAUDE_SKILL_DIR}/scripts/*)`.
- `.claude/settings.json` owns all other pre-approval. Restriction uses `disallowed-tools`.

## Rationale

The field does the opposite of what the skills meant. Dropping it removes the widening; the one
skill that runs a bundled script keeps a grant scoped to that script.

## Consequences
**Gained**: skill turns stop auto-approving Bash and file writes. One file owns pre-approval.
**Accepted**: commands outside the settings allowlist prompt in default mode. That is rare in `auto`
and `plan` mode, where most sessions run. `tasks/audit-default-agent-permissions.md` tracks the
breadth of the settings allowlist.

## Ruled Out
| Idea | Why ruled out | When |
|------|---------------|------|
| Narrow every grant to scoped commands | Only skill-audit runs a bundled script; the rest have nothing to scope | 2026-11-14 |
| A docs-only fix | The broad pre-approval would remain in every skill | 2026-11-14 |

## Enforcement
- Owner: `.claude/skills/*/SKILL.md` and `.claude/commands/*.md` frontmatter
- Pinned by: audit `SK-F11` and `CM-F8`; `tests/test_reference_tree.py::SkillPreApprovalTests`

## Reversal Conditions
- `allowed-tools` starts restricting the tool set instead of only pre-approving.
- Another skill needs a prompt-free bundled script (a scoped grant already passes the audit).

## Related
- [Commands inherit the session model](commands-inherit-the-session-model.md)
