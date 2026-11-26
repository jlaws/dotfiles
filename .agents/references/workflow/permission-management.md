# Permission Management

Reference for settings hierarchy and permission configuration across Claude Code, Codex, and Google Antigravity (`agy`). Controls which tools, commands, and paths an agent can access without prompting.

## Settings Hierarchy

Files are merged top-down; higher-priority files override lower ones.

```
# Claude Code
/Library/.../managed-settings.json                # Enterprise (highest priority, admin-managed)
~/.claude/settings.json                           # User global
project/.claude/settings.json                     # Project (committed to repo)
project/.claude/settings.local.json               # Local (gitignored, personal overrides)

# Google Antigravity (agy)
~/.gemini/antigravity-cli/settings.json           # User global
project/.gemini/antigravity-cli/settings.json     # Project

# Codex
~/.codex/config.toml + ~/.codex/rules/*.rules     # User global posture + command prefix rules
project/.codex/config.toml                        # Project posture (approval_policy, sandbox_mode)
```

**Resolution rule:** deny > allow > default. If the same pattern appears in both allow and deny, deny wins.

## Permission Syntax

**Claude Code** (`.claude/settings.json`, glob wildcards):

```json
{
  "permissions": {
    "allow": ["<Tool>(<pattern>)", ...],
    "deny": ["<Tool>(<pattern>)", ...]
  }
}
```

| Syntax | Meaning |
|--------|---------|
| `"Read"` | Allow/deny all Read calls |
| `"Write(src/**)"` | Allow/deny Write to files under `src/` |
| `"Bash(npm run *)"` | Allow/deny Bash calls matching `npm run *` |
| `"Edit(src/**/*.ts)"` | Allow/deny Edit for TypeScript files in `src/` |
| `"Bash(git push --force*)"` | Deny force-push specifically |

**Google Antigravity** (`.gemini/antigravity-cli/settings.json`, literal word-prefix matching without trailing ` *` or `/**`, and absolute or `~/` file paths):

| Syntax | Meaning |
|--------|---------|
| `"command(git diff)"` | Allow/deny `run_command` calls starting with words `git diff` |
| `"command(sudo)"` | Deny `sudo` invocations |
| `"read_file(~/.ssh)"` | Deny `read_file` under `~/.ssh` |
| `"write_file(/abs/path)"` / `"edit_file(/abs/path)"` | Scope writes/edits to an absolute path prefix |

**Codex** (`.codex/rules/default.rules`, Starlark `prefix_rule` token lists):

```python
prefix_rule(pattern=["git", "status"], decision="allow")
prefix_rule(pattern=["sudo"], decision="forbidden")
```

## Common Permission Sets

### Minimal (Read-Only Exploration)

```json
{
  "permissions": {
    "allow": ["Read", "Glob", "Grep"],
    "deny": ["Write", "Edit", "Bash"]
  }
}
```

### Standard Development

```json
{
  "permissions": {
    "allow": [
      "Read",
      "Glob",
      "Grep",
      "Write(src/**)",
      "Edit(src/**)",
      "Bash(npm run *)",
      "Bash(git *)"
    ],
    "deny": [
      "Bash(rm -rf *)",
      "Write(.env*)",
      "Bash(git push --force*)",
      "Write(dist/**)"
    ]
  }
}
```

### CI/Agent Mode (Full Access)

```json
{
  "permissions": {
    "allow": [
      "Read",
      "Write",
      "Edit",
      "Bash"
    ],
    "deny": [
      "Bash(git push --force*)",
      "Bash(rm -rf /)",
      "Write(.env*)"
    ]
  }
}
```

## Best Practices

| Practice | Why |
|----------|-----|
| Start minimal, expand as needed | Prevents accidental damage to files outside scope |
| Always deny `.env*` writes | Secrets should never be written by Claude |
| Deny force push | Protects shared branch history |
| Use `settings.local.json` for personal prefs | Keeps project settings clean for the team |
| Deny `dist/`, `build/`, `node_modules/` writes | Generated directories shouldn't be hand-edited |

## Debugging Permissions

When Claude says "permission denied":

1. Check which settings file has the relevant rule: `cat .claude/settings.json`
2. Verify deny doesn't shadow your allow (deny always wins)
3. Check glob pattern matches the actual file path
4. Look for enterprise-level overrides in managed settings

## Cross-References

- **reference:hook-patterns**: pre/post tool hooks that work alongside permissions
- **skill:code-agent-meta-patterns**: broader CLAUDE.md and agent configuration
