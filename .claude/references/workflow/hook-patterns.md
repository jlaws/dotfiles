# Hook Patterns

Reference for Claude Code hook configuration patterns. Extracted from `code-agent-meta-patterns` skill.

## Pre-Commit Hook

Validate before committing:

```json
// .claude/settings.json
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

## Post-Tool Validation

Check results after tool execution:

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

## Design Principles
- Hooks should be fast (<5s); slow hooks degrade the workflow
- Fail hooks loudly with clear error messages
- Use matchers to scope hooks narrowly (don't lint on every Bash call)
- Test hooks manually before relying on them
