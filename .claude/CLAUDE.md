# Claude Code Configuration

## Communication Style

### Do
- Be concise and direct. No filler.
- Lead with the answer, explain after if needed.
- Use bullet points and code examples.
- Assume I'm an experienced developer.
- Challenge my assumptions when appropriate.
- Ask clarifying questions rather than guessing.
- Be extremely concise; sacrifice grammar for brevity.
- End plans with unresolved questions list (concise, skip grammar).
- Structure plans in multiple phases.

### Don't
- Over-explain basic concepts.
- Add unnecessary caveats or warnings.
- Repeat requirements back to me.
- Use excessive praise or encouragement.

---

## Standards Reference

### Code Quality
| Resource | Type | Path |
|----------|------|------|
| Code Style | skill | `skills/workflow/code-style/` |
| Clean Code | skill | `skills/workflow/clean-code/` |
| Anti-Patterns | skill | `skills/workflow/anti-patterns/` |
| Code Review | skill | `skills/workflow/code-review-excellence/` |
| Refactoring | agent | `agents/review/refactoring-expert.md` |
| Lint | command | `commands/code-quality/lint.md` |
| Code Improve | command | `commands/code-quality/code-improve.md` |

### Testing
| Resource | Type | Path |
|----------|------|------|
| TDD | skill | `skills/testing/test-driven-development/` |
| TDD Orchestrator | agent | `agents/testing/tdd-orchestrator.md` |
| Test Automator | agent | `agents/testing/test-automator.md` |
| Debugger | agent | `agents/testing/debugger.md` |
| Unit Tests | command | `commands/testing/automated-unit-test-generation.md` |

### Security
| Resource | Type | Path |
|----------|------|------|
| Security Checklist | skill | `skills/security/security-checklist/` |
| Secrets Management | skill | `skills/security/secrets-management/` |
| Auth Patterns | skill | `skills/security/auth-implementation-patterns/` |
| Security Auditor | agent | `agents/security/security-auditor.md` |
| SAST Scan | command | `commands/security/sast-scan.md` |

### Architecture
| Resource | Type | Path |
|----------|------|------|
| Architecture Patterns | skill | `skills/architecture/architecture-patterns/` |
| API Design | skill | `skills/architecture/api-design-principles/` |
| ADRs | skill | `skills/architecture/architecture-decision-records/` |
| System Architect | agent | `agents/architecture/system-architect.md` |
| Backend Architect | agent | `agents/architecture/backend-architect.md` |
| Architect Review | agent | `agents/review/architect-review.md` |

### Git & Workflow
| Resource | Type | Path |
|----------|------|------|
| Git Workflow | command | `commands/workflow/git-workflow.md` |
| GitHub Issues | command | `commands/code-quality/github-issue-resolution.md` |
| Technical Debt | command | `commands/code-quality/technical-debt-analysis-and-remediation.md` |
| Onboard | command | `commands/planning/onboard.md` |

### Languages
| Resource | Type | Path |
|----------|------|------|
| Swift | agent | `agents/languages/swift-pro.md` |
| iOS Development | agent | `agents/frontend/ios-developer.md` |
| TypeScript | agent | `agents/languages/typescript-pro.md` |
| Python | agent | `agents/languages/python-pro.md` |
| Go | agent | `agents/languages/golang-pro.md` |
| Rust | agent | `agents/languages/rust-pro.md` |
| Shell | agent | `agents/languages/shell-pro.md` |

### Documentation
| Resource | Type | Path |
|----------|------|------|
| Docs Generate | command | `commands/documentation/docs-generate.md` |
| Code Explain | command | `commands/documentation/code-explain.md` |
| Technical Writer | agent | `agents/documentation/technical-writer.md` |
| API Documenter | agent | `agents/documentation/api-documenter.md` |
