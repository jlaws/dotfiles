---
name: cmd-architecture-team-design
description: "Comprehensive system design suite — produce architecture documents covering multiple perspectives. Use when designing a new system or documenting existing architecture. Do NOT use for quick design questions (use /cmd-architecture-arch instead)."
disable-model-invocation: true
---

# Team Design

Parse the user's input: it must contain `<directory_path>` followed by `<description>`.
- First token = directory path, remainder = system description.
- If either is missing, ask the user.

**Do NOT use subagents or parallel agents. Process all design perspectives linearly.**

---

## Multi-Perspective Analysis

Process each design perspective sequentially, then synthesize findings.

### Per-Perspective Flow

For each analysis perspective:
- **Specific scope** — one subsystem, one domain
- **Clear goal** — e.g., "analyze security boundaries" not "review the system"
- **Constraints** — "focus on data flow, not implementation details"
- **Expected output** — summary of findings and recommendations

1. Conduct analysis with full context
2. Answer questions if clarification is needed
3. **Deliverable:** analysis + recommendations + summary report
4. **Review** — verify analysis matches requirements
5. **If issues:** revise and re-review. Repeat until pass.
6. **Mark perspective complete**, move to next
