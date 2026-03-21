---
name: cmd-architecture-team-design
description: "Multi-agent system design suite — parallel specialist analysis to produce architecture documents. Use when designing a new system or documenting existing architecture. Do NOT use for quick design questions (use /cmd-architecture-arch instead)."
disable-model-invocation: true
---

# Team Design

Parse the user's input: it must contain `<directory_path>` followed by `<description>`.
- First token = directory path, remainder = system description.
- If either is missing, ask the user.

Use parallel search to conduct multi-perspective system design analysis.

---

## Multi-Perspective Development

Coordination model: **parallel analysis** with orchestrated synthesis.

### Parallel Analysis

Use when 2+ analysis tasks are independent — one doesn't affect others.

#### Task Requirements

Each analysis perspective gets:
- **Specific scope** — one subsystem, one domain
- **Clear goal** — e.g., "analyze security boundaries" not "review the system"
- **Constraints** — "focus on data flow, not implementation details"
- **Expected output** — "return summary of findings and recommendations"

#### When NOT to Parallelize

- **Shared state** — analyses would need the same context simultaneously
- **Exploratory investigation** — you don't know what's relevant yet
- **Need full context** — understanding requires seeing entire system

### Sequential Execution

Use when executing a plan task-by-task. Fresh perspective per task prevents context pollution.

#### Per-Task Flow

1. **Dispatch analysis** with full task text + scene-setting context
2. **Answer questions** if clarification is needed (don't ignore)
3. **Deliverable:** analysis + recommendations + summary report
4. **Review** — verify analysis matches requirements
5. **If issues:** revise and re-review. Repeat until pass.
6. **Mark task complete**, move to next

### Multi-Domain Pipelines

Chain specialists for cross-cutting issues:
- **DB perf:** error-detective -> db-optimizer -> perf-engineer -> devops
- **Frontend bug:** error-detective -> debugger -> ts-pro -> backend -> test-automator
- **Security vuln:** error-detective -> security-auditor -> test-automator -> code-reviewer
