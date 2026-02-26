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
