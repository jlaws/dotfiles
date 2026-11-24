# Hook Patterns

Reference for Claude Code hook configuration patterns. Hooks run shell commands at specific lifecycle points, enabling automated validation, formatting, and guardrails.

> **Note:** Hook `"command"` values are shell commands executed outside the Bash tool; they run as regular shell scripts. The "no compound commands" rule applies to Bash tool calls only, not to hook shell commands. However, prefer simple, focused hook commands where possible.

## Hook Lifecycle Points

| Hook | Fires When | Common Use |
|------|-----------|------------|
| `UserPromptSubmit` | A prompt is submitted, before the model sees it | Log prompts, add context |
| `PreToolUse` | Before a tool executes | Block dangerous commands, validate inputs |
| `PermissionRequest` | A tool call needs a permission decision | Allow or deny by policy |
| `PostToolUse` | After a tool call succeeds | Lint/format written files, verify output |
| `PostToolUseFailure` | After a tool call fails | Log the failure, add context for the retry |
| `Notification` | Claude sends a notification | Custom alerting, logging |
| `SubagentStop` | A subagent finishes | Check a subagent's result |
| `Stop` | Claude stops responding | Post-completion validation, cleanup |
| `PreCompact` | Before context is compacted | Snapshot task/files/next-step to a scratch file |
| `PostCompact` | After compaction completes | Log or check what survived |
| `SessionStart` | A session begins | Restore snapshot, load context-preservation digest |
| `SessionEnd` | A session ends | Cleanup, final logging |

This table is a subset; the harness's hooks reference lists every event.

## Configuration

Hooks live in `.claude/settings.json` (project) or `~/.claude/settings.json` (global).

```json
{
  "hooks": {
    "<lifecycle>": [
      {
        "matcher": "<tool-pattern>",
        "hooks": [
          { "type": "command", "command": "<shell-command>", "timeout": 5 }
        ]
      }
    ]
  }
}
```

`type` names the handler: `command`, `http`, `mcp_tool`, `prompt`, or `agent`. The examples in this
file use `command`. `timeout` is in seconds.

### Matcher Syntax

`matcher` is tested against **tool names**, not against the command or the file path. A matcher
made only of letters, digits, `_`, `-`, spaces, `,`, and `|` is an exact tool name, or a list of
exact names separated by `|` or `,`. Any other character makes it an unanchored regular expression.
It is the one thing in this file that is easy to get wrong, because `permissions.allow` in the same
`settings.json` *does* use the `Bash(npm *)` form. The two surfaces do not share a syntax.

| Pattern | Matches |
|---------|---------|
| `Bash` | the Bash tool, as an exact name |
| `Write\|Edit` | Write or Edit, each as an exact name |
| `Edit.*` | every tool whose name contains `Edit` (`NotebookEdit` too): the `.` makes it a regex, and a regex is unanchored |
| `^Edit$` | Edit only, as an anchored regex |
| `*`, `.*`, or (empty) | every tool |
| `Bash(git commit)` | **nothing.** Parses as `Bash` + a capture group, i.e. the tool name `Bashgit commit` |

To act on a specific command or path, match the tool name and add the `if` field to the handler.
`if` takes permission-rule syntax, such as `"Bash(git *)"` or `"Edit(*.ts)"`, and is read on tool
events only. Inspect `tool_input` inside the script for what `if` cannot express.

### Skill and Agent Scoped Hooks

Hooks can also live in skill or subagent frontmatter:

- A subagent-frontmatter hook ends when the subagent ends. In a project agent it needs workspace trust, and a `-p` run does not count as trust.
- A project skill-frontmatter hook persists for the rest of the session once the skill loads, and registers even in an untrusted `-p` run.
- On PreToolUse, a non-2 exit or a timeout does not block unless the hook sets `onFailure: "block"` (v2.1.295 or later). Set it on every safety gate.

## Per-Harness Support

Not every harness has a hook surface, and two that do disagree on the output shape. The first two
rows were checked against that harness's own documentation. The third is weaker evidence and is
marked as such: it records that this configuration ships no Gemini hook, which is not the same as
proving the harness has none.

| Tree | Tier | Config | Advisory output field |
|------|------|--------|-----------------------|
| `.claude/` | **hook** | `hooks.<Event>[]` in `settings.json` | `hookSpecificOutput.additionalContext` |
| `.codex/` | **hook** | `[[hooks.<Event>]]` in `config.toml` or `hooks.json` | `hookSpecificOutput.additionalContext` |
| `.gemini/` | **absent** | none | none |

The first two rows state what the harness supports. This configuration registers only
`UserPromptSubmit` in each.

The Gemini row is a property of this configuration: a parity test forbids a `hooks/` directory in
that tree, the sync step deletes one if it appears, and its settings file carries only
`permissions`. No claim is made about the harness itself. State a harness as absent rather than
claiming a hook it cannot honor, and say which kind of evidence you have.

### Exit codes and output

| Signal | Effect |
|--------|--------|
| exit 0, no `permissionDecision` | Advisory only. Normal permission flow applies, the command runs unchanged |
| exit 0 + `additionalContext` | The model sees the message; the command still runs |
| exit 0 + `systemMessage` | A warning shown to the user, not to the model; the command still runs |
| exit 0 + plain stdout | Becomes model context only on `UserPromptSubmit`, `UserPromptExpansion`, `SessionStart`, and `PostModelSwitch` |
| `permissionDecision: "deny"` + `permissionDecisionReason` | Blocks the call, reason shown to the model; a block message names the reason and the route to approval |
| exit 2 | Blocks the call regardless of JSON; this is why the blocking examples below `exit 2`, not `exit 1` |
| `updatedInput` | Rewrites the tool input. Avoid: it puts a lossy layer between the agent and its evidence |

A hook that only ever emits a message and exits 0 is advisory by construction; see `### Failure
Direction by Component` below.

## Common Patterns

### Pre-Commit Validation

```json
{
  "hooks": {
    "PreToolUse": [
      {
        "matcher": "Bash",
        "hooks": [
          {
            "type": "command",
            "if": "Bash(git commit *)",
            "command": "verify-before-commit.sh",
            "timeout": 60,
            "onFailure": "block"
          }
        ]
      }
    ]
  }
}
```

The script runs the repo's one verification command and, when it fails, exits 2 with the failure on
stderr.

### Auto-Format on Write

```json
{
  "hooks": {
    "PostToolUse": [
      {
        "matcher": "Write|Edit",
        "hooks": [
          { "type": "command", "command": "format-file.sh", "timeout": 30 }
        ]
      }
    ]
  }
}
```

The hook's input is JSON on stdin, and the command string has no file substitution. The script
reads the path itself:

```bash
#!/bin/bash
file=$(jq -r '.tool_input.file_path')
eslint --fix "$file" && prettier --write "$file"
```

### Block Dangerous Commands

```json
{
  "hooks": {
    "PreToolUse": [
      {
        "matcher": "^Bash$",
        "hooks": [
          { "type": "command", "command": "guard-destructive.sh", "timeout": 5 }
        ]
      },
      {
        "matcher": "^Bash$",
        "hooks": [
          { "type": "command", "command": "guard-force-push.sh", "timeout": 5 }
        ]
      }
    ]
  }
}
```

### Type-Check After Edits

```json
{
  "hooks": {
    "PostToolUse": [
      {
        "matcher": "^(Write|Edit)$",
        "hooks": [
          { "type": "command", "command": "typecheck-file.sh", "timeout": 30 }
        ]
      }
    ]
  }
}
```

The script type-checks only the file named in `tool_input.file_path` and exits 2 with the errors on
stderr. Do not pipe the checker into a pager or a line filter: a pipeline returns its last
command's status, so the hook could never report a failure.

### Post-Stop Verification

```json
{
  "hooks": {
    "Stop": [
      {
        "hooks": [
          { "type": "command", "command": "npm test -- --bail", "async": true }
        ]
      }
    ]
  }
}
```

`async: true` runs the suite in the background and delivers its output on the next turn. An async
hook cannot block.

## Design Principles

- **Fast**: Hooks should complete in <5s; slow hooks degrade the workflow
- **Loud failures**: Exit 2 with a clear error message on stderr to block the action; exit 1 is a non-blocking error
- **Narrow scope**: Use matchers to avoid running on every tool call
- **Idempotent**: Hooks may fire multiple times; ensure safe re-runs
- **Test manually first**: Run the hook command by hand before configuring
- **Hooks enforce; prose advises**: skills and CLAUDE.md are advisory. A rule that must hold needs a hook, or a test or review check where no hook exists

## Troubleshooting

| Problem | Fix |
|---------|-----|
| Hook not firing | Check matcher syntax matches tool name exactly |
| Hook blocks everything | A regex matcher is unanchored: anchor it (`^Edit$`), or use an exact name |
| Hook output not visible | On exit 2 the model reads **stderr**; on exit 0 it reads `additionalContext`. `systemMessage` is a warning shown to the user |
| Hook too slow | Move heavy work to a `Stop` hook, or set `async: true`; an async hook's `timeout` is not enforced |

## Advanced Patterns

### Runtime Cost Profiles

Gate hook enforcement behind an env var so users tune cost without editing hooks:

| Profile (`HOOK_PROFILE`) | Runs |
|--------------------------|------|
| `minimal` | Blocking safety gates only (dangerous commands) |
| `standard` | Safety + format/lint on write (default) |
| `strict` | Standard + type-check + tests on stop |

Also honor a disable list (e.g. `HOOK_DISABLE=typecheck,tests`) so one slow hook can be turned off without removing config.

Per-edit hooks stay fast, scope themselves to the changed file, and never prompt. The full test suite runs at commit or PR time, not on every edit.

### Tiered Authorization Gate

A PreToolUse gate can classify an action into three tiers instead of a binary allow/deny:

| Tier | Action | Examples |
|------|--------|----------|
| Allow | Proceed silently | reads, formatting, local test runs |
| Confirm | Require explicit human confirmation | deletes, `git push --force`, network writes, anything that spends money |
| Block | Refuse | `rm -rf /`, curl-piped-to-shell, writing secrets |

### Failure Direction by Component

A hook that filters or transforms tool content (not a safety gate) MUST pass content through unchanged if it errors: never block or corrupt the workflow because a formatter crashed. Safety gates are the opposite: fail closed (block on error). In Claude Code a gate fails closed only with `onFailure: "block"`; without it a non-2 exit or a timeout lets the action through.

**Detectors fail neutral, not open.** The rule above is for transforms, which have a correct
inert behavior: emit the input. A detector ("is this an error?", "is this content important?")
has no such fallback, and failing open means fabricating a low-confidence positive that downstream
logic then trusts. A detector with no information MUST return neutral and say so.

**Classify by function, not by shape.** Two components with the same plumbing fail in opposite
directions, so the row is chosen by what the component is *for*:

| Component | On error / no information |
|-----------|---------------------------|
| Transform that is not itself a control | Fail **open** (pass the content through unchanged) |
| Transform that *is* a control: redaction, scrubbing, sanitizing | Fail **closed**. A redactor that fails open emits the secret it exists to remove |
| Safety gate | Fail **closed** (block). In Claude Code that needs `onFailure: "block"`; without it a non-2 exit or a timeout lets the action through |
| Detector whose output feeds a safety gate | Fail **closed** (it inherits the gate's direction, because there the expensive error is the false negative) |
| Detector feeding anything else | Fail **neutral** (report no signal, never a weak positive) |

The middle two rows are the ones that get misread. A secret-redacting `PostToolUse` hook is a
transform by shape and a control by function, and the tiered gate above is a detector feeding a
gate: "neutral" there means the dangerous command is allowed.

### PreCompact Snapshot

A `PreCompact` hook can write current task, open files, and next step to a scratch file, and a `SessionStart` hook can read it back: enforcing the handoff rule under Context and artifacts in the always-loaded instruction file mechanically instead of relying on the model to remember.

## Cross-References

- **reference:permission-management**: settings hierarchy and permission patterns
- **skill:code-agent-meta-patterns**: broader CLAUDE.md and agent configuration
