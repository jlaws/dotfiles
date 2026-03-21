---
name: cmd-workflow-team-investigate
description: "Competing hypothesis debugging — multiple agents investigate different theories in parallel. Use when debugging complex bugs where the root cause is unclear. Do NOT use for simple bugs (use /debug instead)."
disable-model-invocation: true
---

# Team Investigate

Bug description: the user's provided input

Use the multi-agent development methodology below with hypothesis-testing approach to investigate.

---

## Multi-Agent Development

Coordination model: **subagents** (parallel task children, ephemeral). Parent orchestrates, subagents execute focused tasks and return results.

### Subagents

#### Mode A: Parallel Dispatch

Use when 2+ tasks are independent — fixing one doesn't affect others, no shared files.

##### Agent Task Requirements

Each agent gets:
- **Specific scope** — one test file, one subsystem, one domain
- **Clear goal** — "make these tests pass" not "fix the tests"
- **Constraints** — "don't change other code"
- **Error context** — paste error messages, test names, stack traces
- **Expected output** — "return summary of root cause and changes"

##### Integration After Parallel Work

1. Read each agent's summary
2. Create an integration branch: `git checkout -b integrate/<description> main`
3. Sequentially merge each subagent's branch:
   ```bash
   git merge <agent-branch> --no-edit
   ```
4. Resolve conflicts between merges if any arise
5. Run full test suite on merged result
6. Clean up worktrees: `git worktree remove <path>` for each

> **WARNING:** NEVER copy files between worktrees via `cp`, `rsync`, or any file-copy mechanism. Always use `git merge`.

##### Worktree Isolation

Implementation subagents MUST use worktree isolation so they work on an isolated copy of the repo.

Rules:
- **Always use worktree isolation** for any subagent that edits files
- Subagents must **NEVER** clean up, delete, or remove their worktree — the parent handles merge and cleanup
- Subagents must **NEVER** invoke the `finishing-branch` skill — return changes on-branch and let the parent decide integration
- Subagents must **NEVER** copy files via `cp`, `rsync`, or any file-copy mechanism
- Subagents must **commit ALL changes then squash** before returning (nothing left untracked/modified):
  ```bash
  git add -A
  ```
  ```bash
  git reset --soft $(git merge-base HEAD main)
  ```
  ```bash
  git commit -m "<summary>"
  ```
- After the subagent completes, the parent receives the worktree path and branch name in the result
- Parent merges changes from the returned branch using `git merge`, then cleans up the worktree

##### When NOT to Parallelize

- **Related failures** — fixing one might fix others; investigate together first
- **Shared state** — agents would edit same files
- **Exploratory debugging** — you don't know what's broken yet
- **Need full context** — understanding requires seeing entire system

#### Mode B: Sequential Subagent Execution

Use when executing a plan task-by-task. Fresh subagent per task prevents context pollution.

##### Per-Task Flow

1. **Dispatch implementer** with full task text + scene-setting context (see Implementer Subagent Prompt Template below)
2. **Answer questions** if implementer asks (don't ignore)
3. **Implementer delivers:** implementation + tests + commit + self-review report
4. **Dispatch spec reviewer** — verify code matches spec (see Spec Compliance Reviewer Prompt Template below; do NOT trust implementer's report; read actual code)
5. **If spec issues:** implementer fixes, re-review. Repeat until pass.
6. **Dispatch code quality reviewer** — only after spec compliance passes (see Code Quality Reviewer Prompt Template below)
7. **If quality issues:** implementer fixes, re-review. Repeat until pass.
8. **Mark task complete**, move to next

##### Context Passing Template

```
Context for {next_agent}:

Completed by {previous_agent}:
- {summary_of_work}
- {key_findings}

Remaining work:
- {specific_tasks}
- {constraints}

Success criteria:
- {measurable_outcomes}
```

#### Multi-Domain Pipelines

Chain specialists for cross-cutting issues:
- **DB perf:** error-detective -> db-optimizer -> perf-engineer -> devops
- **Frontend bug:** error-detective -> debugger -> ts-pro -> backend -> test-automator
- **Security vuln:** error-detective -> security-auditor -> test-automator -> code-reviewer

---

### Red Flags

- Skip reviews (spec compliance OR code quality)
- Dispatch multiple agents on same files without ownership declarations
- Make agents read plan files instead of providing full text in prompt
- Skip scene-setting context when dispatching agents
- Ignore agent questions or findings
- Accept "close enough" on spec compliance
- Start quality review before spec review passes
- Fix issues manually instead of dispatching fix agent (context pollution in sequential mode)
- Move to next task while reviews have open issues
- Spawning implementation subagents without worktree isolation
- Subagent cleaning up its own worktree before parent merges
- Copying files between worktrees instead of using git merge
- Subagent returning without squashing commits

### Common Prompt Mistakes

| Bad | Good |
|-----|------|
| "Fix all the tests" (too broad) | "Fix agent-tool-abort.test.ts" (focused) |
| "Fix the race condition" (no context) | Paste error messages and test names |
| No constraints | "Do NOT change production code" |
| "Fix it" (vague output) | "Return summary of root cause and changes" |
| No file ownership declared | "Files (read-write): src/auth/** — do not touch other paths" |
| Creating 10 agents for 3 tasks | Match team size to actual parallelizable work |

---

### Implementer Subagent Prompt Template

Use this template when dispatching an implementer subagent.

```
description: "Implement Task N: [task name]"
prompt: |
  You are implementing Task N: [task name]

  ## Task Description

  [FULL TEXT of task from plan — paste it here, don't make subagent read a file]

  ## Context

  [Scene-setting: where this fits, dependencies, architectural context]

  ## Before You Begin

  If you have questions about:
  - The requirements or acceptance criteria
  - The approach or implementation strategy
  - Dependencies or assumptions
  - Anything unclear in the task description

  **Ask them now.** Raise concerns before starting work.

  ## Your Job

  Once clear on requirements:
  1. Implement exactly what the task specifies
  2. Write tests (following TDD if task says to)
  3. Verify implementation works
  4. Commit your work
  5. Self-review (see below)
  6. Report back

  Work from: [directory]

  **While you work:** If you encounter something unexpected or unclear,
  **ask questions**. Don't guess or make assumptions.

  ## Before Reporting Back: Self-Review

  Review your work with fresh eyes:

  **Completeness:**
  - Did I implement everything in the spec?
  - Did I miss any requirements?
  - Are there edge cases I didn't handle?

  **Quality:**
  - Are names clear and accurate?
  - Is the code clean and maintainable?
  - Did I follow existing codebase patterns?

  **Discipline:**
  - Did I avoid overbuilding (YAGNI)?
  - Did I only build what was requested?

  **Testing:**
  - Do tests verify behavior (not just mock behavior)?
  - Did I follow TDD if required?
  - Are tests comprehensive?

  Fix any issues found during self-review before reporting.

  ## Report Format

  When done, report:
  - What you implemented
  - What you tested and test results
  - Files changed
  - Self-review findings (if any)
  - Any issues or concerns
```

---

### Spec Compliance Reviewer Prompt Template

Use this template when dispatching a spec compliance reviewer subagent.

**Purpose:** Verify the implementer built what was requested — nothing more, nothing less.

```
description: "Review spec compliance for Task N"
prompt: |
  You are reviewing whether an implementation matches its specification.

  ## What Was Requested

  [FULL TEXT of task requirements]

  ## What Implementer Claims They Built

  [From implementer's report]

  ## CRITICAL: Do Not Trust the Report

  The implementer's report may be incomplete, inaccurate, or optimistic.
  You MUST verify everything independently.

  **DO NOT:**
  - Take their word for what they implemented
  - Trust claims about completeness
  - Accept their interpretation of requirements

  **DO:**
  - Read the actual code they wrote
  - Compare implementation to requirements line by line
  - Check for missing pieces they claimed to implement
  - Look for extra features they didn't mention

  ## Your Job

  Read the implementation code and verify:

  **Missing requirements:**
  - Did they implement everything requested?
  - Are there requirements they skipped or missed?
  - Did they claim something works but didn't actually implement it?

  **Extra/unneeded work:**
  - Did they build things that weren't requested?
  - Did they over-engineer or add unnecessary features?
  - Did they add "nice to haves" not in spec?

  **Misunderstandings:**
  - Did they interpret requirements differently than intended?
  - Did they solve the wrong problem?

  **Verify by reading code, not by trusting the report.**

  ## Report Format

  - PASS: Spec compliant (if everything matches after code inspection)
  - FAIL: Issues found — list specifically what's missing or extra,
    with file:line references
```

---

### Code Quality Reviewer Prompt Template

Use this template when dispatching a code quality reviewer subagent.

**Purpose:** Verify implementation is well-built — clean, tested, maintainable.

**Only dispatch after spec compliance review passes.**

```
description: "Review code quality for Task N"
prompt: |
  You are reviewing code quality for a recently implemented task.

  ## What Was Implemented

  [From implementer's report — summary of changes]

  ## Requirements Context

  [Task N from plan — so you understand what was being built]

  ## Diff to Review

  Base: [commit SHA before task]
  Head: [current commit SHA]

  Run: git diff <base>..<head>

  ## Your Job

  Review the implementation for:

  **Code Quality:**
  - Is the code clean, readable, and well-organized?
  - Do names accurately describe what things do?
  - Is there unnecessary complexity or duplication?
  - Does it follow existing codebase patterns and conventions?

  **Testing:**
  - Do tests actually verify behavior (not just coverage)?
  - Are edge cases tested?
  - Are tests maintainable and clear?
  - Do tests follow the project's testing patterns?

  **Architecture:**
  - Does the implementation fit the existing architecture?
  - Are abstractions at the right level?
  - Is there appropriate separation of concerns?

  **Potential Issues:**
  - Race conditions, error handling gaps
  - Security concerns (injection, auth, data exposure)
  - Performance issues (N+1 queries, unnecessary allocations)

  ## Report Format

  **Strengths:** What was done well

  **Issues:** (grouped by severity)
  - Critical: Must fix before merge
  - Important: Should fix, significant impact
  - Minor: Nice to have, low impact

  **Assessment:** PASS / PASS WITH NOTES / NEEDS CHANGES
```
