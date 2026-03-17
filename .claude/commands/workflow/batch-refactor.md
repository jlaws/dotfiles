---
name: batch-refactor
description: "Parallel batch refactoring across many files using worktree-isolated agents. Use when refactoring a concept, pattern, or API across 10+ files. Do NOT use for small refactors touching <5 files (just do it directly) or behavior changes (use /brainstorm + /write-plan)."
argument-hint: "<refactoring description>"
---

Refactoring: $ARGUMENTS

If no arguments provided, ask what the user wants to refactor.

## Coordination Plan

Follow this sequence:

### Phase 1: Discovery
1. Identify EVERY file that needs changes (use Grep/Glob exhaustively)
2. Group files into independent batches (default: 5 batches). Files in the same batch must not depend on each other.
3. Present the batch plan to the user for approval before proceeding.

### Phase 2: Parallel Execution
For each batch, spawn a Task agent with `isolation: "worktree"`:

```
Agent(
  prompt="<batch-specific instructions with file list and refactoring rules>",
  isolation="worktree",
  subagent_type="general-purpose"
)
```

Each agent must:
- Make the specified changes to its assigned files only
- Run lint/format on changed files
- Commit changes to its isolated branch
- Stage and commit ALL changes, then squash before returning:
  ```bash
  git add -A
  ```
  ```bash
  git reset --soft $(git merge-base HEAD main)
  ```
  ```bash
  git commit -m "<batch N: summary>"
  ```
- NOT clean up its worktree (parent handles merge)
- NOT invoke finishing-branch skill

Pre-flight: verify each worktree is fully isolated before dispatching.

### Phase 3: Integration
After all agents complete:
1. Create integration branch: `git checkout -b refactor/<description> main`
2. Sequentially merge each agent's branch:
   ```bash
   git merge <agent-branch> --no-edit
   ```
3. Run full lint validation on the merged result
4. Run tests if applicable
5. Clean up worktrees: `git worktree remove <path>` for each

> **WARNING:** NEVER use `cp`/`rsync` to copy files between worktrees. Always use `git merge`.

### Phase 4: PR
Use the create-pr workflow to open the PR with a summary of all changes.

---

## Multi-Agent Development

Two coordination models: **subagents** (Task tool children, ephemeral) and **agent teams** (TeamCreate/TaskCreate/SendMessage, persistent).

### Decision: Subagents vs Agent Teams

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

#### When to Use Subagents

- 2-5 independent tasks with no shared files
- Sequential plan execution with review gates
- One-shot research or analysis queries
- Tasks where agents don't need to talk to each other

#### When to Use Agent Teams

- Multi-perspective review (security + quality + testing)
- Large implementation with module boundaries
- Research requiring synthesis across agents
- Adversarial analysis (competing hypotheses)
- Work that benefits from persistent agent context across subtasks

---

### Part 1: Subagents

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

### Part 2: Agent Teams

#### Team Lifecycle

1. **TeamCreate** — create team with shared task list
2. **TaskCreate** — define work items with dependencies (blockedBy/blocks)
3. **Task tool with `team_name`** — spawn teammates into the team
4. **TaskUpdate** — assign tasks (set owner), track progress, manage dependencies
5. **SendMessage** — inter-agent communication (DM or broadcast)
6. **Shutdown** — graceful teammate termination via `shutdown_request` + `TeamDelete`

#### Agent Type Selection

| Agent Type | Tools Available | Use For |
|------------|----------------|---------|
| Explore | Read-only (Glob, Grep, Read, WebFetch) | Review, research, analysis |
| general-purpose | All tools | Implementation, editing, testing |
| Plan | Read-only + plan output | Architecture planning |
| Bash | Bash only | Command execution, CI/CD tasks |

#### Task Sizing Heuristics

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

#### File Conflict Avoidance

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

#### Communication Patterns

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

### Lead Modes

#### Delegate Mode (Shift+Tab)

Restricts the lead to coordination-only — no file editing, no code writing. The lead can only read files, manage tasks, and send messages. Use for:
- Review teams where the lead synthesizes findings but shouldn't modify code
- Research teams where the lead coordinates but doesn't investigate
- Any scenario where you want to prevent accidental lead edits

Toggle with **Shift+Tab** during a session.

#### Plan Approval Mode

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

### Display Modes

Configure `teammateMode` via environment or CLI flag:

| Mode | Behavior |
|------|----------|
| `auto` | tmux split panes if tmux is available, in-process otherwise |
| `tmux` | Always use tmux split panes (fails if tmux not running) |
| `in-process` | Teammates run in the same terminal (background, output via notifications) |

Navigate between teammate panes: **Shift+Up** / **Shift+Down**.

---

### Quality Gate Hooks

Use Claude Code hooks to enforce standards on teammate output:

#### TeammateIdle Hook
Fires when a teammate finishes a turn. Exit code 2 keeps the teammate working (prevents premature idle).

#### TaskCompleted Hook
Fires when a teammate marks a task complete. Exit code 2 blocks completion (forces the teammate to address issues first).

Example: a hook that runs tests before allowing task completion, rejecting if tests fail.

---

### Task Sizing

- Target **5-6 tasks per teammate** — enough to stay productive, not so many that context gets diluted
- Each task should be completable in one focused session (roughly 50-150 tool calls)
- If a task needs more, split it into subtasks with dependencies

---

### Known Limitations

- **No session resumption** for in-process teammates — if the lead's session ends, teammates are lost
- **One team per session** — cannot run multiple TeamCreate in the same conversation
- **No nested teams** — a teammate cannot create its own sub-team
- **Shutdown can be slow** — teammates finish their current turn before processing shutdown_request
- **Token cost** — teams use significantly more tokens than subagents; prefer subagents for routine parallel work where agents don't need to communicate

---

### Red Flags (Both Modes)

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

### Common Prompt Mistakes

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

---

### Implementer Subagent Prompt Template

Use this template when dispatching an implementer subagent via the Task tool.

```
Task tool (general-purpose):
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
Task tool (general-purpose):
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
Task tool (general-purpose):
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

---

### Conventions for Team-Enabled Skills

#### Convention 1: Team Configuration Block

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

#### Convention 2: Single-Agent Fallback

Every team-enabled skill MUST work as a single agent too. Team mode is an optional enhancement, not a requirement. The skill's core workflow remains the same — team mode parallelizes it.

#### Convention 3: Synthesis Protocol

1. Collect all teammate findings
2. Deduplicate across perspectives
3. Resolve contradictions (flag for user if unresolvable)
4. Merge into the skill's standard output template
5. Attribute findings to the perspective that caught them

#### Convention 4: File Ownership Declaration

For implementation teams, each task declares owned files:

```
Files: src/auth/**
Constraint: Do NOT modify files outside this path
```

Lead enforces boundaries during task creation. If a task needs files owned by another agent, add a dependency (blockedBy) so they run sequentially.

#### Convention 5: Worktree Lifecycle

Subagents and teammates that edit code MUST run in worktree isolation. The lifecycle is:

1. **Parent creates** — sets `isolation: "worktree"` when spawning
2. **Agent works** — stages ALL changes, commits to the worktree branch, then squashes before returning:
   ```bash
   git add -A
   ```
   ```bash
   git reset --soft $(git merge-base HEAD main)
   ```
   ```bash
   git commit -m "<summary>"
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

---

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

---

## Using Git Worktrees

**Announce at start:** "I'm using the using-git-worktrees skill to set up an isolated workspace."

### Directory Selection (Priority Order)

#### 1. Check Existing Directories
```bash
ls -d .worktrees 2>/dev/null     # Preferred (hidden)
ls -d worktrees 2>/dev/null      # Alternative
```
If both exist, `.worktrees` wins.

#### 2. Check CLAUDE.md
```bash
grep -i "worktree.*director" CLAUDE.md 2>/dev/null
```
If preference specified, use it.

#### 3. Ask User
```
No worktree directory found. Where should I create worktrees?
1. .worktrees/ (project-local, hidden)
2. ~/.config/superpowers/worktrees/<project-name>/ (global)
```

### Safety Verification

**For project-local directories: MUST verify ignored before creating.**

```bash
git check-ignore -q .worktrees 2>/dev/null
```

**If NOT ignored:** Add to .gitignore, commit, then proceed.

**For global directory (~/.config/superpowers/worktrees):** No verification needed.

### Creation Steps

```bash
# 1. Detect project
project=$(basename "$(git rev-parse --show-toplevel)")

# 2. Create worktree
git worktree add "$path" -b "$BRANCH_NAME"
cd "$path"

# 3. Auto-detect and run setup
[ -f package.json ] && npm install
[ -f Cargo.toml ] && cargo build
[ -f requirements.txt ] && pip install -r requirements.txt
[ -f pyproject.toml ] && poetry install
[ -f go.mod ] && go mod download

# 4. Verify clean baseline
# Run project-appropriate test command
# If tests fail: report failures, ask whether to proceed

# 5. Report
# "Worktree ready at <path>, tests passing (N tests, 0 failures)"
```

### Completing Work in a Worktree

Before returning or signaling completion:

1. **Stage and commit** all changes (nothing untracked or modified)
2. **Squash** into a single commit (three separate Bash tool calls):
   ```bash
   git add -A
   ```
   ```bash
   git reset --soft $(git merge-base HEAD main)
   ```
   ```bash
   git commit -m "<summary of changes>"
   ```
3. **Report** your branch name and worktree path to the parent/caller
4. Do NOT remove the worktree, merge to main, or invoke `finishing-branch`

> The parent agent is responsible for `git merge` and `git worktree remove`.

### Quick Reference

| Situation | Action |
|-----------|--------|
| `.worktrees/` exists | Use it (verify ignored) |
| `worktrees/` exists | Use it (verify ignored) |
| Both exist | Use `.worktrees/` |
| Neither exists | Check CLAUDE.md, then ask user |
| Directory not ignored | Add to .gitignore + commit |
| Tests fail in baseline | Report failures + ask |

### Examples

**Trigger:** "Start isolated feature work without stashing current changes"
**Action:** Create a new git worktree with a feature branch, set up the environment
**Result:** Two independent working directories — original branch untouched, new feature branch ready

### Integration

- **Called by:** brainstorming (after design approved), any skill needing isolation
- **Pairs with:** finishing-a-development-branch (cleanup after), executing-plans / subagent-driven-development (work happens here)

---

## Refactoring & Technical Debt

### The Discipline

```
TEST -> REFACTOR -> VERIFY -> COMMIT
Never skip a step. Never combine refactoring with behavior changes.
```

A refactoring changes structure without changing behavior. If you're adding features or fixing bugs at the same time, you're not refactoring -- you're gambling.

#### The Cadence

1. **Ensure tests pass** (run full suite, green baseline)
2. **Make ONE structural change** (single refactoring operation)
3. **Run tests again** (must still pass -- if not, revert immediately)
4. **Commit** (small atomic commit, message describes the refactoring)
5. **Repeat**

Each commit is a safe rollback point. If anything breaks, `git revert` is trivial.

### Code Smell to Refactoring Mapping

| Code Smell | Refactoring(s) |
|------------|-----------------|
| Long method (>30 lines) | Extract function |
| Long parameter list (>3) | Introduce parameter object |
| Duplicated code | Extract function / Extract base class |
| Feature envy (method uses another class's data) | Move method |
| Data clumps (same fields grouped repeatedly) | Extract class / Introduce parameter object |
| Switch on type in multiple places | Replace conditional with polymorphism |
| Divergent change (class changed for multiple reasons) | Extract class (split by responsibility) |
| Shotgun surgery (one change touches many files) | Move method / Inline class (consolidate) |
| Primitive obsession (strings/ints for domain concepts) | Introduce value object |
| Speculative generality (unused abstractions) | Inline class / Remove dead code |
| Dead code | Delete it. Git has history. |

### Safe Refactoring Sequences

| Goal | Sequence |
|------|----------|
| Break up god class | Extract methods -> Group related -> Extract classes -> Define interfaces |
| Remove inheritance | Push down unused methods -> Extract interface -> Replace with delegation -> Remove base |
| Simplify complex conditional | Extract each branch into named function -> Replace with lookup map or polymorphism |
| Migrate to new API | Introduce adapter -> Route calls through adapter -> Swap implementation -> Inline adapter |

### IDE-Assisted vs Manual

| Refactoring | IDE? | Notes |
|-------------|:---:|-------|
| Rename | Yes | Always use IDE -- catches all references |
| Extract function | Yes | IDE infers parameters and return type |
| Move file/module | Yes | Updates import paths automatically |
| Change signature | Yes | IDE updates all call sites |
| Replace conditional with polymorphism | No | Requires architectural judgment |
| Strangler fig / Branch by abstraction | No | Strategy, not mechanical transformation |

**Rule**: If your IDE can do it, let your IDE do it.

### Large-Scale Refactoring Strategies

#### Strangler Fig

Gradually replace a legacy system by routing new functionality to a new implementation while keeping the old one running.

1. Identify the boundary (API, module interface, route)
2. Build new implementation behind the same interface
3. Route traffic/calls incrementally (feature flag, proxy, or router)
4. Monitor both paths (correctness + performance)
5. Remove old implementation when 100% migrated

**Progressive rollout:** 5% -> 25% -> 50% -> 100% (24h observation between increases)
**Rollback triggers:** Error rate >1%, latency >2x baseline

#### Branch by Abstraction

1. Create an interface/protocol wrapping the current implementation
2. Update all callers to use the abstraction (test + commit)
3. Build new implementation behind the same abstraction
4. Switch (toggle, config, or swap) to new implementation
5. Remove old implementation and (optionally) the abstraction

#### Parallel Implementation

Run old and new code simultaneously, compare outputs, converge when confident. Best for high-risk logic changes where correctness is critical (payments, data pipelines).

### Refactoring Catalog — Code Examples

#### Extract Function / Method

**When**: Code block needs a comment to explain intent, or is duplicated.

```python
# Before
def process_order(order):
    # Calculate discount
    if order.customer.is_premium and order.total > 100:
        discount = order.total * 0.15
    elif order.total > 200:
        discount = order.total * 0.10
    else:
        discount = 0
    order.total -= discount
    # ... more processing

# After
def calculate_discount(order):
    if order.customer.is_premium and order.total > 100:
        return order.total * 0.15
    if order.total > 200:
        return order.total * 0.10
    return 0

def process_order(order):
    order.total -= calculate_discount(order)
    # ... more processing
```

#### Extract Class / Module

**When**: A class has multiple responsibilities or a module is >300 lines with distinct sections.

Split by responsibility. Each resulting unit should have a single reason to change.

#### Inline Function / Variable

**When**: Indirection adds no clarity. The function body is as clear as the name.

```typescript
// Before
function isEligible(age: number): boolean {
    return age >= 18;
}
const eligible = isEligible(user.age);

// After (if only used once and meaning is obvious)
const eligible = user.age >= 18;
```

#### Replace Conditional with Polymorphism

**When**: Switch/if-else chain on a type field that appears in 3+ places.

```typescript
// Before
function getArea(shape: Shape): number {
    switch (shape.type) {
        case 'circle': return Math.PI * shape.radius ** 2;
        case 'rectangle': return shape.width * shape.height;
        case 'triangle': return 0.5 * shape.base * shape.height;
    }
}

// After
interface Shape {
    getArea(): number;
}
class Circle implements Shape {
    getArea() { return Math.PI * this.radius ** 2; }
}
class Rectangle implements Shape {
    getArea() { return this.width * this.height; }
}
```

#### Replace Inheritance with Composition

**When**: Subclass only uses a fraction of parent, or "is-a" relationship is forced.

```python
# Before
class AudioPlayer(MediaWidget):  # inherits 50 methods, uses 5
    pass

# After
class AudioPlayer:
    def __init__(self):
        self.media = MediaWidget()  # delegate what you need
```

#### Introduce Parameter Object

**When**: 3+ parameters travel together across multiple functions.

```go
// Before
func createUser(name string, email string, age int, role string, dept string) {}

// After
type CreateUserParams struct {
    Name  string
    Email string
    Age   int
    Role  string
    Dept  string
}
func createUser(params CreateUserParams) {}
```

#### Replace Magic Values with Constants

```python
# Before
if response.status_code == 429:
    time.sleep(60)

# After
RATE_LIMIT_STATUS = 429
RATE_LIMIT_COOLDOWN_SECONDS = 60
if response.status_code == RATE_LIMIT_STATUS:
    time.sleep(RATE_LIMIT_COOLDOWN_SECONDS)
```

#### Strangler Fig — Detailed Example

```python
# Phase 1: Facade over legacy
class PaymentFacade:
    def process_payment(self, order):
        return self.legacy_processor.doPayment(order.to_legacy())

# Phase 2: New service alongside
class PaymentService:
    def process_payment(self, order): ...

# Phase 3: Feature-flagged migration
class PaymentFacade:
    def process_payment(self, order):
        if feature_flag("use_new_payment"):
            return self.new_service.process_payment(order)
        return self.legacy.doPayment(order.to_legacy())
```

#### Parallel Implementation

```python
def process(data):
    old_result = old_implementation(data)
    new_result = new_implementation(data)
    if old_result != new_result:
        log.warning(f"Mismatch: {old_result} vs {new_result}")
    return old_result  # switch to new_result when confident
```

#### Metrics Dashboard Template

```yaml
cyclomatic_complexity: { current: 15.2, target: 10.0 }
code_duplication: { current: 23%, target: 5% }
test_coverage: { unit: 45%, integration: 12%, target: 80%/60% }
dependency_health: { outdated_major: 12, security_vulns: 7 }
```

#### Impact Assessment Example

```
Debt Item: Duplicate user validation logic (5 files)
Time Impact: 2 hrs/bug fix, 4 hrs/feature change
Monthly: ~20 hours | Annual: 240 hrs x $150/hr = $36,000
```

### Technical Debt Inventory

#### Code Debt
- **Duplicated Code**: Exact duplicates, similar logic, repeated rules
- **Complex Code**: Cyclomatic complexity >10, nesting >3 levels, methods >50 lines, god classes >500 lines
- **Poor Structure**: Circular dependencies, feature envy, shotgun surgery

#### Architecture Debt
- Missing/leaky abstractions, violated boundaries, monolithic components
- Outdated frameworks, deprecated APIs, unsupported dependencies

#### Testing Debt
- Coverage gaps, missing integration/performance tests
- Brittle/flaky/slow tests

#### Infrastructure Debt
- Manual deployment, no rollback, missing monitoring

### Impact Assessment

| Risk Level | Criteria |
|------------|----------|
| Critical | Security vulnerabilities, data loss risk |
| High | Performance degradation, frequent outages |
| Medium | Developer frustration, slow delivery |
| Low | Code style, minor inefficiencies |

**Quantify**: `Debt Item -> hrs/incident x frequency = monthly cost x rate = annual cost`

### Prioritized Remediation

| Horizon | Examples | Criteria |
|---------|----------|----------|
| Quick wins (1-2 weeks) | Extract shared module, add monitoring, automate deploy | High savings/effort ratio |
| Medium-term (1-3 months) | Refactor god classes, framework upgrade | Moderate effort, measurable ROI |
| Long-term (2-4 quarters) | DDD migration, comprehensive test suite | Strategic investment |

### Prevention

```yaml
pre_commit_hooks:
  - complexity_check: "max 10"
  - duplication_check: "max 5%"
  - test_coverage: "min 80% for new code"
ci_pipeline:
  - dependency_audit: "no high vulnerabilities"
  - performance_test: "no regression >10%"
  - architecture_check: "no new violations"
```

### Forcing Functions

- **Canonical run scripts**: Provide scripts for running services locally. If setup is broken, someone finds out immediately
- **Encode standards in tooling**: Linters, formatters, pre-commit hooks, coding agent prompts -- not just wikis
- **Tickets over TODOs**: File tickets with deadlines rather than adding `// TODO` comments that rot
- **Continuous releases**: If deployment is painful, that pain surfaces immediately and gets fixed

### Boy Scout Rule

Leave the code a little better than you found it.

- When encountering tech debt while working on a feature, **default towards fixing it** rather than working around it
- In PR reviews, ask others to consider taking care of nearby technical debt
- **Proportionality**: small nearby improvements (rename, extract helper, fix docstring) are encouraged; full refactors should be separate tickets

### When NOT to Refactor

| Situation | Why Not |
|-----------|---------|
| No tests covering the code | Can't verify behavior preservation. Write tests first. |
| Under deadline pressure | Ship first, refactor next sprint. |
| Code being deleted soon | Don't polish what you're throwing away. |
| "While I'm in here..." scope creep | File a ticket, do it separately. Exception: boy scout rule. |
| Single implementation | Don't create abstractions for one concrete case. Wait for the second. |

### Gotchas

- Refactoring without tests is walking a tightrope without a net. Write characterization tests first if coverage is low
- "Refactoring" that changes behavior is rewriting -- separate into distinct commits
- Large PRs labeled "refactoring" are suspicious; each step should be its own atomic green commit
- Branch by abstraction's "temporary" interface layer tends to become permanent; set a deadline

### Stakeholder Summary Template

```markdown
## Executive Summary
- Current debt score: [X] (High)
- Monthly velocity loss: [X]%
- Recommended investment: [X] hours
- Expected ROI: [X]% over 12 months

## Key Risks
1. [Critical risk with impact]

## Proposed Actions
1. Immediate: [this week]
2. Short-term: [1 month]
3. Long-term: [6 months]
```

### Agent Team Mode

For comprehensive debt audits of large codebases.

```yaml
team:
  recommended_size: 4
  agent_roles:
    - name: code-debt-analyst
      type: Explore
      focus: "Duplication, complexity, code smell inventory"
      skills: ["workflow:refactoring-and-debt", "workflow:code-quality"]
    - name: arch-debt-analyst
      type: Explore
      focus: "Boundary violations, dependency analysis, pattern drift"
      skills: ["workflow:refactoring-and-debt"]
      references: [".claude/references/architecture/architecture-decision-records.md"]
    - name: test-debt-analyst
      type: Explore
      focus: "Coverage gaps, flaky tests, missing integration tests"
      skills: ["workflow:refactoring-and-debt", "testing:language-testing-patterns"]
    - name: infra-debt-analyst
      type: Explore
      focus: "Deployment gaps, monitoring holes, dependency health"
      skills: ["workflow:refactoring-and-debt"]
      references: [".claude/references/devops/observability.md"]
  file_ownership: "shared-read-only"
  lead_mode: "hands-on"
```

### Cross-References

- **workflow:code-quality** -- code smells, style conventions
- **workflow:verification-before-completion** -- ensuring refactoring is verified before claiming done
