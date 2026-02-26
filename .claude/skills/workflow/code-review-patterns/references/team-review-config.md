# Team Review Configuration

Agent team configuration for parallel multi-perspective diff review.

## Team Configuration

```yaml
team:
  recommended_size: 4
  agent_roles:
    - name: security-reviewer
      type: Explore
      focus: "STRIDE analysis, vulnerability patterns, secrets detection"
      skills_loaded: ["security:security-analysis", "security:auth-implementation-patterns"]
      steps: ["Step 4.3"]
    - name: quality-reviewer
      type: Explore
      focus: "Code smells, edge cases, error handling, naming, DRY"
      skills_loaded: ["workflow:code-quality", "workflow:code-review-patterns"]
      steps: ["Step 4.1", "Step 4.2"]
    - name: test-reviewer
      type: Explore
      focus: "Coverage gaps, test quality, missing integration tests"
      skills_loaded: ["testing:language-testing-patterns", "testing:test-driven-development"]
      steps: ["Step 4.4"]
    - name: language-reviewer
      type: Explore
      focus: "Language-specific gotchas, idiom violations"
      skills_loaded: ["Auto-detected languages:*-patterns"]
      steps: ["Step 4.5"]
  file_ownership: "shared-read-only"
  lead_mode: "hands-on"
```

## Team Workflow

1. **Lead** executes Steps 1-3 (identify changes, gather context, detect scope)
2. **Lead** distributes the full diff + changed files to all reviewers
3. **Reviewers** work in parallel, each covering their assigned steps
4. **Lead** collects findings, deduplicates, resolves contradictions
5. **Lead** produces Step 5 structured report + Step 6 decision gate

## Single-Agent Fallback

Without team mode, execute all perspectives sequentially (default behavior). Team mode is an optional enhancement for large diffs or when explicitly requested.
