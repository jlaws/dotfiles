# Hook Patterns

Reference for Claude Code hook configuration patterns. Hooks run shell commands at specific lifecycle points, enabling automated validation, formatting, and guardrails.

> **Note:** Hook `"command"` values are shell commands executed outside the Bash tool — they run as regular shell scripts. The "no compound commands" rule applies to Bash tool calls only, not to hook shell commands. However, prefer simple, focused hook commands where possible.

## Hook Lifecycle Points

| Hook | Fires When | Common Use |
|------|-----------|------------|
| `PreToolUse` | Before a tool executes | Block dangerous commands, validate inputs |
| `PostToolUse` | After a tool executes | Lint/format written files, verify output |
| `Notification` | Claude sends a notification | Custom alerting, logging |
| `Stop` | Claude stops responding | Post-completion validation, cleanup |
| `PreCompact` | Before context is compacted | Snapshot task/files/next-step to a scratch file |
| `SessionStart` | A session begins | Restore snapshot, load context-preservation digest |

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

### Matcher Syntax

`matcher` is a regex over **tool names**, not over the command or the file path. It is the one
thing in this file that is easy to get wrong, because `permissions.allow` in the same
`settings.json` *does* use the `Bash(npm *)` form. The two surfaces do not share a syntax.

| Pattern | Matches |
|---------|---------|
| `Bash` | every Bash call (also `BashOutput`, since the regex is unanchored) |
| `^Bash$` | Bash calls only |
| `Write\|Edit` | Write or Edit tool calls |
| `.*` or (empty) | every tool |
| `Bash(git commit)` | **nothing.** Parses as `Bash` + a capture group, i.e. the tool name `Bashgit commit` |

To act on a specific command or path, match the tool name and inspect `tool_input` inside the
script -- which is what `guard-bash-output.sh` does.

## Per-Harness Support

Not every harness has a hook surface, and two that do disagree on the output shape. The first two
rows were checked against that harness's own documentation. The third is weaker evidence and is
marked as such: it records that this configuration ships no Gemini hook, which is not the same as
proving the harness has none.

| Tree | Tier | Config | Advisory output field |
|------|------|--------|-----------------------|
| `.claude/` | **hook** | `hooks.PreToolUse[]` in `settings.json`, `matcher: "Bash"` | top-level `systemMessage` |
| `.codex/` | **hook** | `[[hooks.PreToolUse]]` in `config.toml` or `hooks.json`, `matcher = "^Bash$"` | `hookSpecificOutput.additionalContext` |
| `.gemini/` | **absent** | none | none |

The Gemini row is a property of this configuration: a parity test forbids a `hooks/` directory in
that tree, the sync step deletes one if it appears, and its settings file carries only
`permissions`. No claim is made about the harness itself. State a harness as absent rather than
claiming a hook it cannot honor, and say which kind of evidence you have.

### Exit codes and output

| Signal | Effect |
|--------|--------|
| exit 0, no `permissionDecision` | Advisory only. Normal permission flow applies, the command runs unchanged |
| exit 0 + `systemMessage` / `additionalContext` | The model sees the message; the command still runs |
| `permissionDecision: "deny"` + `permissionDecisionReason` | Blocks the call, reason shown to the model |
| exit 2 | Blocks the call regardless of JSON — this is why the blocking examples below `exit 2`, not `exit 1` |
| `updatedInput` | Rewrites the tool input. Avoid: it puts a lossy layer between the agent and its evidence |

A hook that only ever emits a message and exits 0 is advisory by construction — see the Fail-Open
Principle below.

## Common Patterns

### Pre-Commit Validation

```json
{
  "hooks": {
    "PreToolUse": [
      {
        "matcher": "^Bash$",
        "hooks": [
          { "type": "command", "command": "lint-staged && npm test", "timeout": 60 }
        ]
      }
    ]
  }
}
```

### Auto-Format on Write

```json
{
  "hooks": {
    "PostToolUse": [
      {
        "matcher": "Write|Edit",
        "hooks": [
          { "type": "command", "command": "eslint --fix ${file} && prettier --write ${file}", "timeout": 30 }
        ]
      }
    ]
  }
}
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
          { "type": "command", "command": "npx tsc --noEmit --pretty 2>&1 | head -20", "timeout": 120 }
        ]
      }
    ]
  }
}
```

### Post-Stop Verification

```json
{
  "hooks": {
    "Stop": [
      {
        "hooks": [
          { "type": "command", "command": "npm test -- --bail 2>&1 | tail -5", "timeout": 300 }
        ]
      }
    ]
  }
}
```

## Design Principles

- **Fast** — Hooks should complete in <5s; slow hooks degrade the workflow
- **Loud failures** — Exit non-zero with a clear error message to block the action
- **Narrow scope** — Use matchers to avoid running on every tool call
- **Idempotent** — Hooks may fire multiple times; ensure safe re-runs
- **Test manually first** — Run the hook command by hand before configuring

## Troubleshooting

| Problem | Fix |
|---------|-----|
| Hook not firing | Check matcher syntax matches tool name exactly |
| Hook blocks everything | Anchor the matcher (`^Bash$`, not `Bash`) and narrow inside the script |
| Hook output not visible | On exit 2 the model reads **stderr**; on exit 0 it reads `systemMessage` |
| Hook too slow | Move heavy work to `Stop` hook or run async with `&` |

## Advanced Patterns

### Runtime Cost Profiles

Gate hook enforcement behind an env var so users tune cost without editing hooks:

| Profile (`HOOK_PROFILE`) | Runs |
|--------------------------|------|
| `minimal` | Blocking safety gates only (dangerous commands) |
| `standard` | Safety + format/lint on write (default) |
| `strict` | Standard + type-check + tests on stop |

Also honor a disable list (e.g. `HOOK_DISABLE=typecheck,tests`) so one slow hook can be turned off without removing config.

### Tiered Authorization Gate

A PreToolUse gate can classify an action into three tiers instead of a binary allow/deny:

| Tier | Action | Examples |
|------|--------|----------|
| Allow | Proceed silently | reads, formatting, local test runs |
| Confirm | Require explicit human confirmation | deletes, `git push --force`, network writes, anything that spends money |
| Block | Refuse | `rm -rf /`, curl-piped-to-shell, writing secrets |

### Fail-Open Principle

A hook that filters or transforms tool content (not a safety gate) MUST pass content through unchanged if it errors — never block or corrupt the workflow because a formatter crashed. Safety gates are the opposite: fail closed (block on error).

### PreCompact Snapshot

A `PreCompact` hook can write current task, open files, and next step to a scratch file, and a `SessionStart` hook can read it back — enforcing the Context Preservation rule mechanically instead of relying on the model to remember.

## Cross-References

- **reference:permission-management** — settings hierarchy and permission patterns
- **skill:code-agent-meta-patterns** — broader CLAUDE.md and agent configuration
