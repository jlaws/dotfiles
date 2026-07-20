---
name: github-issue-resolution
description: "Use when resolving a GitHub issue through implementation."
compatibility: claude-code
allowed-tools: Read, Grep, Glob, Bash, Edit, Write
skills:
  - verification-before-completion
  - test-driven-development
---

# GitHub Issue Resolution

## Issue Analysis and Triage

### Initial Investigation

```bash
gh issue view $ISSUE_NUMBER --comments
gh issue view $ISSUE_NUMBER --json title,body,labels,assignees,milestone,state
gh issue list --search "similar keywords" --state closed --limit 10
git log --oneline --grep="component_name" -20
```

### Priority Classification

| Priority | Criteria |
|----------|----------|
| P0/Critical | Production breaking, security vulnerability, data loss |
| P1/High | Major feature broken, significant user impact |
| P2/Medium | Minor feature affected, workaround available |
| P3/Low | Cosmetic issue, enhancement request |

## Root Cause Analysis

```bash
git bisect start
git bisect bad HEAD
git bisect good <last_known_good_commit>
git bisect run ./test_issue.sh

git blame -L <start>,<end> path/to/file.js
rg "functionName" --type js -A 3 -B 3
```

## Branch Strategy

```bash
git checkout -b feature/issue-${ISSUE_NUMBER}-short-description
git checkout -b fix/issue-${ISSUE_NUMBER}-component-bug
git checkout -b hotfix/issue-${ISSUE_NUMBER}-critical-fix
```

## Implementation

### Incremental Commits
```bash
git commit -m "feat(auth): add user validation schema (#${ISSUE_NUMBER})"
git commit -m "test(auth): add unit tests for validation (#${ISSUE_NUMBER})"
```

## PR Creation

```bash
gh pr create \
  --title "Fix #${ISSUE_NUMBER}: Clear description of the fix" \
  --body "$(cat <<EOF
## Summary
Fixes #${ISSUE_NUMBER} by implementing proper error handling.

## Changes Made
- Added validation for expired tokens
- Updated unit and integration tests

## Testing
- [x] All existing tests pass
- [x] Added new unit tests
- [x] Manual testing completed

## Checklist
- [x] Code follows project style guidelines
- [x] Self-review completed
- [x] No new warnings introduced
- [x] Documentation updated (or N/A)
EOF
)" \
  --base main \
  --assignee @me \
  --label "bug,needs-review"
```

## Post-Implementation

```bash
gh run list --workflow=deploy
gh issue comment ${ISSUE_NUMBER} \
  --body "Fixed in PR #${PR_NUMBER}. Root cause: improper token validation."
gh issue close ${ISSUE_NUMBER} --comment "Resolved via #${PR_NUMBER}"
```

## Examples

**Trigger:** "Fix bug from GitHub issue #42"
**Action:** Fetch issue details, reproduce the bug, implement fix, add tests, open PR referencing #42
**Result:** PR created with `Fixes #42` — issue auto-closes on merge

## Delivery Checklist

1. Resolution summary with root cause
2. Code changes with explanations
3. Test results and coverage
4. Pull request with issue linking
5. Verification steps for reviewers
6. Rollback plan
7. Documentation validated — product docs and any KB self-docs current, or N/A with reason (see `documentation-validation`)
