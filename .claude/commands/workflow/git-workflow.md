---
description: Git branch, commit, and PR conventions
---

# Git Workflow

Follow these conventions when working with git.

## Branch Naming

| Type | Pattern | Example |
|------|---------|---------|
| Main | `main` | Always deployable |
| Feature | `feature/short-description` | `feature/user-auth` |
| Fix | `fix/issue-description` | `fix/null-pointer-login` |
| Cleanup | `cleanup/what-changed` | `cleanup/remove-dead-code` |
| Docs | `docs/what-documented` | `docs/api-endpoints` |

```bash
# Create feature branch
git checkout -b feature/user-auth

# Create fix branch
git checkout -b fix/null-pointer-login
```

## Commit Messages

### Format
```
type: description
```

- Lowercase
- Imperative mood ("add" not "added")
- No period at end
- Describe what AND why, not how

### Types
| Type | Use For |
|------|---------|
| `feat` | New feature |
| `fix` | Bug fix |
| `docs` | Documentation only |
| `refactor` | Code change without behavior change |
| `test` | Adding/updating tests |
| `chore` | Build, deps, config changes |

### Examples
```bash
# Good
git commit -m "feat: add password reset flow"
git commit -m "fix: prevent crash on empty user list"
git commit -m "refactor: extract validation into separate module"

# Bad
git commit -m "Fixed stuff"
git commit -m "WIP"
git commit -m "Updates."
```

## Pull Requests

### Title
Match the primary commit message.

### Body Template
```markdown
## Summary
- Brief description of changes
- Why this change was made

## Test Plan
- [ ] Unit tests pass
- [ ] Manual testing done
- [ ] Edge cases covered

## Related
Closes #123
```

### Guidelines
- Keep PRs small and focused
- One logical change per PR
- Reference related issues
- Include test plan
