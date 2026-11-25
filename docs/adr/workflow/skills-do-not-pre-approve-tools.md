---
status: accepted
topic: workflow
created: 2026-11-14
updated: 2026-11-25
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
- Skills drop `allowed-tools`, except skill-audit, which keeps one `Bash(python3 ${CLAUDE_SKILL_DIR}/scripts/<script>.py ...)` grant per bundled script. Naming each script, rather than `scripts/*`, keeps a `../` path out of the grant.
- A wildcard-only scope (`Bash(*)`, `Bash()`, `Bash(:*)`) counts as unscoped.
- `.claude/settings.json` owns all other pre-approval. Restriction uses `disallowed-tools`.
- No agent config pre-approves a command that runs an arbitrary program or repo-controlled code,
  launches an app, or sends data to a remote. An entry stays only if it can do none of the three.
  A family that mixes safe and unsafe subcommands is narrowed to the subcommands the KB runs.
  Antigravity permissions equal the Claude allow list, translated.

| Entry (Claude form) | Finding | Disposition |
|---|---|---|
| `npx *`, `xargs *`, `awk *`, `env` | run any program (`awk 'BEGIN{system()}'`, `xargs sh -c`, `env sh -c` in the prefix-matching harnesses) | removed |
| `chmod *`, `open *` | make a cloned file executable, then launch it | removed |
| `tee *` | writes any file; used only as a pipe sink | removed |
| `npm test *`, `eslint *`, `prettier *` | run package scripts or load repo JS config | removed |
| `git clone *`, `git clean *` | `--upload-pack` runs a command; `-fdx` deletes untracked work; no KB workflow runs either | removed |
| `gh *` | `gh api` POSTs, `gh gist create`, `gh extension install`, `gh alias set --shell` | narrowed to `gh pr view/list/diff/checks/create/edit *`, `gh issue view/list *`, `gh run list/view *`, `gh repo view *` |
| `python3`, `python`, `uv`, `cargo`, `make`, `ruff`, `rg`, `find` (Antigravity only) | `-c`, `run`, `build.rs`, the repo Makefile (which un-prompts the paid eval targets), `--pre`, `-exec` | removed |
| `.codex/rules/default.rules` trailing one-line rules (`just ...`, `cargo ...`, benchmarks, a private `cd` path, `ps`, `ulimit`, `ci ...`) | another project's approvals, one naming a private path | deleted |
| `tsc *`, `swiftlint *`, `swiftformat *`, `npm list/outdated/audit/info *`, `agent-browser *`, read utilities, `git push/reset/branch *`, `cp/mv/mkdir/touch/sed *` | no code execution; or workflow-critical with the limit recorded below | kept |

## Rationale

The field does the opposite of what the skills meant. Dropping it removes the widening; the one
skill that runs a bundled script keeps a grant scoped to that script.

## Consequences
**Gained**: skill turns stop auto-approving Bash and file writes. One file owns pre-approval.
**Accepted**: commands outside the settings allowlist prompt in default mode. That is rare in `auto`
and `plan` mode, where most sessions run.
**Now prompts**: on Claude and Codex, `npx`, `npm test`, `eslint`, `prettier`, `chmod`, `open`, `tee`,
`awk`, `xargs`, `env`, `git clone`, `git clean`, `gh api`, `gh pr merge`, `gh issue comment`, and
`gh issue close`. On Antigravity also `make`, `python3`, `python`, `uv`, `cargo`, `ruff`, `rg`, and
`find`.
**Accepted limits** (recorded, not filed): a prefix rule cannot bound the target of `cp`, `mv`,
`mkdir`, `touch`, or `sed -i`, nor a later `--force` on `git push`. The sandbox setting is the
control for that class and stays off.

## Ruled Out
| Idea | Why ruled out | When |
|------|---------------|------|
| Narrow every grant to scoped commands | Only skill-audit runs a bundled script; the rest have nothing to scope | 2026-11-14 |
| A docs-only fix | The broad pre-approval would remain in every skill | 2026-11-14 |
| Narrow `npx`, `make`, or `python` to named targets | In an untrusted clone the package, Makefile, and script are the attacker's | 2026-11-25 |
| Deny-prefix rules for dangerous flags | A prefix cannot match a flag in a later position | 2026-11-25 |

## Enforcement
- Owner: `.claude/skills/*/SKILL.md` and `.claude/commands/*.md` frontmatter
- Pinned by: audit `SK-F11` and `CM-F8`; `tests/test_reference_tree.py::SkillPreApprovalTests`

## Reversal Conditions
- `allowed-tools` starts restricting the tool set instead of only pre-approving.
- Another skill needs a prompt-free bundled script (a scoped grant already passes the audit).

## Related
- [Commands inherit the session model](commands-inherit-the-session-model.md)
