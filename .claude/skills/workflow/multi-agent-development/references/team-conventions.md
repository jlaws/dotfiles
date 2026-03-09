# Conventions for Team-Enabled Skills

### Convention 1: Team Configuration Block

Every team-enabled skill should include:

```yaml
## Agent Team Mode
team:
  recommended_size: 3-5
  agent_roles:
    - name: role-name
      type: Explore  # or general-purpose
      focus: "What this agent does"
      skills_loaded: ["category:skill-name"]
  file_ownership: "by-module" | "by-perspective" | "shared-read-only"
  lead_mode: "delegate" | "hands-on"
```

### Convention 2: Single-Agent Fallback

Every team-enabled skill MUST work as a single agent too. Team mode is an optional enhancement, not a requirement. The skill's core workflow remains the same — team mode parallelizes it.

### Convention 3: Synthesis Protocol

1. Collect all teammate findings
2. Deduplicate across perspectives
3. Resolve contradictions (flag for user if unresolvable)
4. Merge into the skill's standard output template
5. Attribute findings to the perspective that caught them

### Convention 4: File Ownership Declaration

For implementation teams, each task declares owned files:

```
Files: src/auth/**
Constraint: Do NOT modify files outside this path
```

Lead enforces boundaries during task creation. If a task needs files owned by another agent, add a dependency (blockedBy) so they run sequentially.

### Convention 5: Worktree Lifecycle

Subagents and teammates that edit code MUST run in worktree isolation. The lifecycle is:

1. **Parent creates** — sets `isolation: "worktree"` when spawning
2. **Agent works** — commits to the worktree branch, then squashes before returning:
   ```bash
   git reset --soft $(git merge-base HEAD main 2>/dev/null || git merge-base HEAD master) && git commit -m "<summary>"
   ```
3. **Parent receives** — worktree path + branch name in agent result
4. **Parent merges** — integrates using `git merge <agent-branch> --no-edit` (or `git cherry-pick <commit>` for single commits)
5. **Parent cleans up** — `git worktree remove <path>` after successful merge

Agents must NEVER:
- Delete their own worktree
- Merge their branch into main/master
- Run the `finishing-branch` skill
- Call `git worktree remove`
- Copy files to the parent worktree via `cp`, `rsync`, or any file-copy mechanism
