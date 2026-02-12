---
name: multi-agent-development
description: Use when coordinating multiple agents — subagents for simple parallel/sequential tasks, or agent teams for complex multi-perspective review, research, and implementation.
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
2. Verify fixes don't conflict
3. Run full test suite
4. Integrate all changes

#### When NOT to Parallelize

- **Related failures** — fixing one might fix others; investigate together first
- **Shared state** — agents would edit same files
- **Exploratory debugging** — you don't know what's broken yet
- **Need full context** — understanding requires seeing entire system

### Mode B: Sequential Subagent Execution

Use when executing a plan task-by-task. Fresh subagent per task prevents context pollution.

#### Per-Task Flow

1. **Dispatch implementer** with full task text + scene-setting context
2. **Answer questions** if implementer asks (don't ignore)
3. **Implementer delivers:** implementation + tests + commit + self-review report
4. **Dispatch spec reviewer** — verify code matches spec (do NOT trust implementer's report; read actual code)
5. **If spec issues:** implementer fixes, re-review. Repeat until pass.
6. **Dispatch code quality reviewer** — only after spec compliance passes
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

#### Review Team

- Lead gathers context (diff, changed files, branch info)
- Distributes full context to specialist reviewers (security, quality, testing, language)
- Each reviewer focuses on one perspective using Explore agents
- Lead merges findings, deduplicates, produces unified report

#### Research Team

- Lead defines scope, distributes query sets
- Each researcher explores different sources/angles in parallel
- Lead synthesizes findings, builds taxonomy or comparison matrix

#### Implementation Team

- Lead creates phased plan with file ownership boundaries
- Teammates implement independent modules in parallel (general-purpose agents)
- Reviewer teammate validates each module
- Lead integrates, runs full test suite

#### Adversarial Team

- Lead poses a question or problem
- Multiple agents investigate competing hypotheses independently
- Agents challenge each other's findings via messaging
- Lead synthesizes with confidence-weighted conclusions

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
