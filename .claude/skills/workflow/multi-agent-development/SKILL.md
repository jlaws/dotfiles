---
name: multi-agent-development
description: "Multi-agent coordination patterns for subagents and agent teams with task sizing, file ownership, and communication protocols. Use when coordinating multiple agents — subagents for simple parallel/sequential tasks, or agent teams for complex multi-perspective review, research, and implementation. Do NOT use for general Claude Code workflow optimization (use code-agent-meta-patterns)."
compatibility: claude-code
allowed-tools: Read, Grep, Glob, Bash
---

# Multi-Agent Development

Two coordination models: **subagents** (Task tool children, ephemeral) and **agent teams** (TeamCreate/TaskCreate/SendMessage, persistent).

## Decision: Subagents vs Agent Teams

| Dimension | Subagents (Task tool) | Agent Teams (TeamCreate) |
|-----------|----------------------|--------------------------|
| Lifetime | Single task, then gone | Persistent across multiple tasks |
| Communication | Return result to parent only | Bidirectional messaging (DM + broadcast) |
| Coordination | Parent orchestrates sequentially | Shared task list with dependencies |
| Shared state | None — each gets fresh context | Task list visible to all teammates |
| Best for | Focused, independent work units | Multi-perspective analysis, phased implementation |
| Overhead | Low — one prompt, one result | Higher — team setup, task management, shutdown |
| File conflicts | Parent must prevent | File ownership declarations prevent |
| Context window | Fresh per dispatch | Each agent maintains own context |

### When to Use Subagents

- 2-5 independent tasks with no shared files
- Sequential plan execution with review gates
- One-shot research or analysis queries
- Tasks where agents don't need to talk to each other

### When to Use Agent Teams

- Multi-perspective review (security + quality + testing)
- Large implementation with module boundaries
- Research requiring synthesis across agents
- Adversarial analysis (competing hypotheses)
- Work that benefits from persistent agent context across subtasks

---

## Part 1: Subagents

### Mode A: Parallel Dispatch

Use when 2+ tasks are independent — fixing one doesn't affect others, no shared files.

#### Agent Task Requirements

Each agent gets:
- **Specific scope** — one test file, one subsystem, one domain
- **Clear goal** — "make these tests pass" not "fix the tests"
- **Constraints** — "don't change other code"
- **Error context** — paste error messages, test names, stack traces
- **Expected output** — "return summary of root cause and changes"

#### Integration After Parallel Work

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

#### Worktree Isolation

Implementation subagents MUST use `isolation: "worktree"` so they work on an isolated copy of the repo:

```
Agent(prompt="...", isolation="worktree", subagent_type="general-purpose")
```

Rules:
- **Always set `isolation: "worktree"`** for any subagent that edits files
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

#### When NOT to Parallelize

- **Related failures** — fixing one might fix others; investigate together first
- **Shared state** — agents would edit same files
- **Exploratory debugging** — you don't know what's broken yet
- **Need full context** — understanding requires seeing entire system

### Mode B: Sequential Subagent Execution

Use when executing a plan task-by-task. Fresh subagent per task prevents context pollution.

#### Per-Task Flow

1. **Dispatch implementer** with full task text + scene-setting context (see `references/implementer-prompt.md` for template)
2. **Answer questions** if implementer asks (don't ignore)
3. **Implementer delivers:** implementation + tests + commit + self-review report
4. **Dispatch spec reviewer** — verify code matches spec (see `references/spec-reviewer-prompt.md` for template; do NOT trust implementer's report; read actual code)
5. **If spec issues:** implementer fixes, re-review. Repeat until pass.
6. **Dispatch code quality reviewer** — only after spec compliance passes (see `references/code-quality-reviewer-prompt.md` for template)
7. **If quality issues:** implementer fixes, re-review. Repeat until pass.
8. **Mark task complete**, move to next

#### Context Passing Template

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

### Multi-Domain Pipelines

Chain specialists for cross-cutting issues:
- **DB perf:** error-detective -> db-optimizer -> perf-engineer -> devops
- **Frontend bug:** error-detective -> debugger -> ts-pro -> backend -> test-automator
- **Security vuln:** error-detective -> security-auditor -> test-automator -> code-reviewer

---

## Part 2: Agent Teams

### Team Lifecycle

1. **TeamCreate** — create team with shared task list
2. **TaskCreate** — define work items with dependencies (blockedBy/blocks)
3. **Task tool with `team_name`** — spawn teammates into the team
4. **TaskUpdate** — assign tasks (set owner), track progress, manage dependencies
5. **SendMessage** — inter-agent communication (DM or broadcast)
6. **Shutdown** — graceful teammate termination via `shutdown_request` + `TeamDelete`

### Agent Type Selection

| Agent Type | Tools Available | Use For |
|------------|----------------|---------|
| Explore | Read-only (Glob, Grep, Read, WebFetch) | Review, research, analysis |
| general-purpose | All tools | Implementation, editing, testing |
| Plan | Read-only + plan output | Architecture planning |
| Bash | Bash only | Command execution, CI/CD tasks |

### Team Composition Patterns

See references/composition-patterns.md for review, research, implementation, and adversarial team archetypes.

### Task Sizing Heuristics

**Split when:**
- Task touches 3+ unrelated files or modules
- Task has independent subtasks that don't share state
- Task would take a single agent more than ~100 tool calls
- Different parts need different expertise (security vs performance vs testing)

**Keep together when:**
- Changes are tightly coupled (modifying a function + its callers)
- Task requires understanding full context to make decisions
- Splitting would create merge conflicts
- The task is small enough for one agent to handle efficiently

### File Conflict Avoidance

Declare file ownership per task to prevent conflicts:

```
Files (read-write): src/auth/**
Files (read-only): src/shared/types.ts
Constraint: Do NOT modify files outside declared paths
```

Rules:
- No two teammates get read-write access to the same file
- Shared types/interfaces are read-only for all; lead integrates changes
- If ownership overlap is unavoidable, serialize those tasks (blockedBy)

### Communication Patterns

**Use DM (SendMessage type: "message") for:**
- Responding to a specific teammate
- Requesting clarification on a single task
- Sharing findings relevant to one other agent

**Use broadcast (SendMessage type: "broadcast") sparingly for:**
- Critical blocking issues that affect everyone
- Major discoveries that change the overall approach
- Announcing completion of a dependency that unblocks multiple tasks

**Message content conventions:**
- Lead with the actionable point
- Include file paths and line numbers when referencing code
- Keep messages concise — teammates have their own context

---

## Conventions for Team-Enabled Skills

See references/team-conventions.md for team config blocks, single-agent fallback, synthesis protocol, and file ownership conventions.

---

## Lead Modes

### Delegate Mode (Shift+Tab)

Restricts the lead to coordination-only — no file editing, no code writing. The lead can only read files, manage tasks, and send messages. Use for:
- Review teams where the lead synthesizes findings but shouldn't modify code
- Research teams where the lead coordinates but doesn't investigate
- Any scenario where you want to prevent accidental lead edits

Toggle with **Shift+Tab** during a session.

### Plan Approval Mode

Spawn teammates with `mode: "plan"` to require lead approval before they edit files:

```
Task(subagent_type="general-purpose", mode="plan", team_name="my-team", ...)
```

The teammate will:
1. Explore the codebase and write a plan
2. Call ExitPlanMode, which sends a `plan_approval_request` to the lead
3. Wait for the lead's `plan_approval_response` before proceeding

Use for risky implementation work where you want a review gate before edits begin.

---

## Display Modes

Configure `teammateMode` via environment or CLI flag:

| Mode | Behavior |
|------|----------|
| `auto` | tmux split panes if tmux is available, in-process otherwise |
| `tmux` | Always use tmux split panes (fails if tmux not running) |
| `in-process` | Teammates run in the same terminal (background, output via notifications) |

Navigate between teammate panes: **Shift+Up** / **Shift+Down**.

---

## Quality Gate Hooks

Use Claude Code hooks to enforce standards on teammate output:

### TeammateIdle Hook
Fires when a teammate finishes a turn. Exit code 2 keeps the teammate working (prevents premature idle).

### TaskCompleted Hook
Fires when a teammate marks a task complete. Exit code 2 blocks completion (forces the teammate to address issues first).

Example: a hook that runs tests before allowing task completion, rejecting if tests fail.

---

## Task Sizing

- Target **5-6 tasks per teammate** — enough to stay productive, not so many that context gets diluted
- Each task should be completable in one focused session (roughly 50-150 tool calls)
- If a task needs more, split it into subtasks with dependencies

---

## Known Limitations

- **No session resumption** for in-process teammates — if the lead's session ends, teammates are lost
- **One team per session** — cannot run multiple TeamCreate in the same conversation
- **No nested teams** — a teammate cannot create its own sub-team
- **Shutdown can be slow** — teammates finish their current turn before processing shutdown_request
- **Token cost** — teams use significantly more tokens than subagents; prefer subagents for routine parallel work where agents don't need to communicate

---

## Red Flags (Both Modes)

- Skip reviews (spec compliance OR code quality)
- Dispatch multiple agents on same files without ownership declarations
- Make agents read plan files instead of providing full text in prompt
- Skip scene-setting context when dispatching agents
- Ignore agent questions or findings
- Accept "close enough" on spec compliance
- Start quality review before spec review passes
- Fix issues manually instead of dispatching fix agent (context pollution in sequential mode)
- Move to next task while reviews have open issues
- Broadcast when a DM would suffice (wastes all agents' context)
- Create teams for work a single subagent could handle (unnecessary overhead)
- Forget to shut down teammates after work completes (resource leak)
- Spawning implementation subagents without `isolation: "worktree"`
- Subagent cleaning up its own worktree before parent merges
- Copying files between worktrees instead of using git merge
- Subagent returning without squashing commits

## Common Prompt Mistakes

| Bad | Good |
|-----|------|
| "Fix all the tests" (too broad) | "Fix agent-tool-abort.test.ts" (focused) |
| "Fix the race condition" (no context) | Paste error messages and test names |
| No constraints | "Do NOT change production code" |
| "Fix it" (vague output) | "Return summary of root cause and changes" |
| No file ownership declared | "Files (read-write): src/auth/** — do not touch other paths" |
| Broadcasting status updates | DM the lead with your status |
| Creating 10 agents for 3 tasks | Match team size to actual parallelizable work |
| Assigning overlapping files to teammates | Serialize tasks or split file ownership |
