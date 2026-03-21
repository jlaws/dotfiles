# Hook Patterns

Reference for Claude Code hook configuration patterns. Hooks run shell commands at specific lifecycle points, enabling automated validation, formatting, and guardrails.

> **Note:** Hook `"hook"` values are shell commands executed outside the Bash tool — they run as regular shell scripts. The "no compound commands" rule applies to Bash tool calls only, not to hook shell commands. However, prefer simple, focused hook commands where possible.

## Hook Lifecycle Points

| Hook | Fires When | Common Use |
|------|-----------|------------|
| `PreToolUse` | Before a tool executes | Block dangerous commands, validate inputs |
| `PostToolUse` | After a tool executes | Lint/format written files, verify output |
| `Notification` | Claude sends a notification | Custom alerting, logging |
| `Stop` | Claude stops responding | Post-completion validation, cleanup |

## Configuration

Hooks live in `.cursor/settings.json` (project) or `~/.cursor/settings.json` (global).

```json
{
  "hooks": {
    "<lifecycle>": [
      {
        "matcher": "<tool-pattern>",
        "hook": "<shell-command>"
      }
    ]
  }
}
```

### Matcher Syntax

| Pattern | Matches |
|---------|---------|
| `Bash(git commit)` | Bash calls containing "git commit" |
| `Write\|Edit` | Write or Edit tool calls |
| `Write(src/**)` | Write calls targeting `src/` paths |
| `Bash(npm *)` | Any Bash call starting with "npm" |
| (empty) | All calls of that tool |

## Common Patterns

### Pre-Commit Validation

```json
{
  "hooks": {
    "PreToolUse": [
      {
        "matcher": "Bash(git commit)",
        "hook": "lint-staged && npm test"
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
        "hook": "eslint --fix ${file} && prettier --write ${file}"
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
        "matcher": "Bash(rm -rf)",
        "hook": "echo 'Blocked: rm -rf is denied by project hooks' && exit 1"
      },
      {
        "matcher": "Bash(git push --force)",
        "hook": "echo 'Blocked: force push denied' && exit 1"
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
        "matcher": "Write(src/**/*.ts)|Edit(src/**/*.ts)",
        "hook": "npx tsc --noEmit --pretty 2>&1 | head -20"
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
        "hook": "npm test -- --bail 2>&1 | tail -5"
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
| Hook blocks everything | Narrow the matcher pattern (e.g., `Bash(git commit)` not `Bash(git)`) |
| Hook output not visible | Ensure command writes to stdout; stderr may be swallowed |
| Hook too slow | Move heavy work to `Stop` hook or run async with `&` |

## Cross-References

- **reference:permission-management** — settings hierarchy and permission patterns
- **skill:code-agent-meta-patterns** — broader CLAUDE.md and agent configuration
