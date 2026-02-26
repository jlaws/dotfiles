# Permission Management

Reference for Claude Code settings hierarchy and permission patterns. Extracted from `code-agent-meta-patterns` skill.

## Settings Hierarchy

```
/Library/.../managed-settings.json   # Enterprise (highest priority)
~/.claude/settings.json              # User global
project/.claude/settings.json        # Project
project/.claude/settings.local.json  # Local (gitignored)
```

## Common Permission Patterns

```json
// .claude/settings.json (project)
{
  "permissions": {
    "allow": [
      "Bash(npm run *)",
      "Bash(git *)",
      "Read",
      "Write(src/**)",
      "Edit(src/**)"
    ],
    "deny": [
      "Bash(rm -rf *)",
      "Write(.env*)",
      "Bash(git push --force*)"
    ]
  }
}
```

## Rules
- Default to minimal permissions, expand as needed
- Deny rules override allow rules
- Use glob patterns for path-based permissions
- `settings.local.json` for personal overrides (gitignored)
