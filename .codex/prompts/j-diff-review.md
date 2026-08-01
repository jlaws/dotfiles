---
name: j-diff-review
description: "Deep multi-perspective diff review — code quality, security, testing gaps, documentation drift, observability gaps, and language-specific gotchas. Use when reviewing a diff or PR before merge. Do NOT use for simple code questions (ask directly instead)."
argument-hint: "<diff-ref-or-branch>"
---

Use the diff-review workflow below to review: $ARGUMENTS

---

## Code Review Patterns

### Review Mindset

**Goals:** Catch bugs/edge cases, ensure maintainability, share knowledge, enforce standards, improve design.

**Not goals:** Show off knowledge, nitpick formatting (use linters), block progress unnecessarily, rewrite to preference.

---

### Giving Reviews

#### Review Process (time-boxed)

##### Phase 1: Context (2-3 min)
1. Read PR description and linked issue
2. Check PR size (>400 lines? Ask to split)
3. CI/CD status passing?
4. Understand the business requirement

##### Phase 2: High-Level (5-10 min)
- Does solution fit the problem? Simpler approaches?
- Consistent with existing patterns? Will it scale?
- Are there tests? Do they cover edge cases?

##### Phase 3: Line-by-Line (10-20 min)
- **Logic**: Edge cases, off-by-one, null checks, race conditions
- **Security**: Input validation, SQL injection, XSS, data exposure
- **Performance**: N+1 queries, unnecessary loops, memory leaks, blocking ops
- **Maintainability**: Clear names, SRP functions, magic numbers extracted

##### Phase 4: Summary (2-3 min)
1. Summarize key concerns
2. Highlight what worked well
3. Clear decision: Approve / Comment / Request Changes
4. Offer to pair if complex

#### Feedback Severity Labels

```
[blocking]    - Must fix before merge
[important]   - Should fix, discuss if disagree
[nit]         - Nice to have, not blocking
[suggestion]  - Alternative approach to consider
[learning]    - Educational, no action needed
```

#### Feedback Techniques

- **Ask questions** instead of stating problems: "What happens if `items` is empty?"
- **Suggest, don't command**: "Would it make sense to extract this? It appears in 3 places."
- **Be specific and actionable**: "Race condition when concurrent access — consider a mutex here."

#### Handling Disagreements

1. **Seek understanding**: "What led you to choose this pattern?"
2. **Acknowledge valid points**: "That's a fair consideration about X."
3. **Provide data**: "Can we add a benchmark to validate?"
4. **Escalate if needed**: Get architect/senior to weigh in
5. **Let go if non-critical**: Perfection is the enemy of progress

#### Common Pitfalls

Perfectionism (blocking for style) | Scope creep ("while you're at it...") | Delayed reviews | Rubber stamping | Bike shedding

---

### Pre-Submission Diff Review

Self-review workflow for current branch changes vs main. Catch issues before reviewers do.

**Do NOT use `gh` or any GitHub CLI commands. All information must come from local git.**

#### Step 1 — Identify Changes

```bash
git diff main...HEAD
git log main..HEAD --oneline
```

FAIL FAST if current branch IS main or has no commits ahead of main.

#### Step 2 — Gather Full Context

```bash
git diff main...HEAD --name-only
```

**Read every changed file in full** — not just diff hunks. Context beyond changed lines catches cross-cutting issues.

#### Step 3 — Detect Review Scope

Inspect file extensions. Load matching language-specific review patterns:

| Extension | Pattern Set |
|-----------|-------------|
| `.py` | Python patterns |
| `.js`, `.ts`, `.tsx` | JS/TS patterns |
| `.go` | Go concurrency patterns |
| `.sh` | Bash defensive patterns |
| `.swift` | Swift patterns |
| `.rs` | Rust project patterns |

Flag missing tests if diff modifies source but includes no test changes.

#### Step 4 — Multi-Perspective Analysis

Analyze the diff from each perspective independently, then merge findings.

**4.1 Code Review** — Edge cases, error handling, logic errors, missing validation (Phase 3 checklist above).

**4.2 Code Quality** — Smells, naming, DRY violations, unnecessary complexity, coupling.

**4.3 Security** — STRIDE threats, injection, XSS, SSRF, path traversal, auth gaps, secrets in code, insecure defaults.

**4.4 Testing** — See Language Testing Patterns section below: coverage gaps, test quality (behavior vs implementation), missing integration tests, flaky indicators.

**4.5 Language-Specific Gotchas** — Apply auto-detected language patterns. See Language-Specific Review Gotchas subsection below.

**4.6 Documentation** — Cross-reference `workflow:documentation-validation`: does the diff change public surface (API, CLI, config) or documented behavior without updating README/API docs/CHANGELOG? Flag stale docs as a finding.

**4.7 Observability** — New code paths with no logging, metrics, or tracing; errors swallowed without a log; new endpoints or jobs with no SLO or alert; analytics events absent from the tracking plan.

#### Step 5 — Structured Findings Report

```markdown
## Diff Review — {BRANCH_NAME}

### Critical
- {finding} — {file:line} — {perspective}

### High
- {finding} — {file:line} — {perspective}

### Medium
- {finding} — {file:line} — {perspective}

### Test Gaps
- {description of missing coverage}

### Doc Gaps
- {doc that no longer matches the code, or public surface with no doc}

### Observability Gaps
- {code path with no instrumentation, or error path with no signal}

### What Looks Good
- {positive observation}
```

Omit empty severity sections. Always include "What Looks Good".

#### Step 6 — Disposition Ladder

The agents reported; none of them changed anything. Every finding is now yours to dispose of. For each one, take the first rung that applies — do not skip ahead:

1. **Fix it** — if the fix is reasonably scoped: clear defect, inside the diff's boundary, verifiable. Apply it, run the project's checks, commit atomically.
2. **Add it to the active plan** — if a plan-mode plan is open, append the finding there as a future phase. **Never create a new plan file.**
3. **Add it to the repository's future-work mechanism** — if the repo has one, follow its convention (`TODO.md`, `docs/plans/`, GitHub issues, a tracker named in AGENTS.md or CONTRIBUTING.md). Detect it; do not invent one.
4. **Ask** — nothing above applied. Ask the user, carrying a recommendation and the research behind it.

Read the code a finding touches before deciding. Reaching rung 4 without having researched is the failure mode; so is skipping rung 1 for something you could simply have fixed.

**Handing back a findings list with no disposition is a failed run.** State the rung for every finding.

Scope guard: fixing a finding does not license unrelated refactors. A fix that grows past the diff's boundary drops to rung 2. Follow PR Comment Resolution Step 4 (scope guard, atomic commits, verify before push).

---

### Receiving & Responding to Reviews

#### Response Pattern

```
1. READ: Complete feedback without reacting
2. UNDERSTAND: Restate requirement in own words (or ask)
3. VERIFY: Check against codebase reality
4. EVALUATE: Technically sound for THIS codebase?
5. RESPOND: Technical acknowledgment or reasoned pushback
6. IMPLEMENT: One item at a time, test each
```

#### Handling Unclear Feedback

If ANY item is unclear, stop. Do not implement partially.

```
Understand 1,2,3,6. Unclear on 4,5.
RIGHT: "Understand 1,2,3,6. Need clarification on 4 and 5 before proceeding."
WRONG: Implement 1,2,3,6 now, ask about 4,5 later
```

#### From External Reviewers — Verify Before Implementing

Before implementing: technically correct for THIS codebase? Breaks existing functionality? Reason for current implementation? Works on all platforms? Does reviewer understand full context?

If suggestion seems wrong: push back with technical reasoning.
If conflicts with prior architectural decisions: stop and discuss with project owner.

#### When to Push Back

- Suggestion breaks existing functionality
- Reviewer lacks full context
- Violates YAGNI (unused feature)
- Technically incorrect for this stack
- Legacy/compatibility reasons exist
- Conflicts with prior architectural decisions

**How:** Technical reasoning, specific questions, reference working tests/code.

#### Implementation Order for Multi-Item Feedback

1. Clarify anything unclear FIRST
2. Blocking issues (breaks, security)
3. Simple fixes (typos, imports)
4. Complex fixes (refactoring, logic)
5. Test each fix individually
6. Verify no regressions

#### GitHub Thread Replies

Reply inline in comment threads (`gh api repos/{owner}/{repo}/pulls/{pr}/comments/{id}/replies`), not as top-level PR comments.

---

### Multi-Perspective Review

Analyze the diff from all six perspectives below. They are independent:

1. **Security** — STRIDE analysis, vulnerability patterns, secrets detection — covers Step 4.3
2. **Code Quality** — Code smells, edge cases, error handling, naming, DRY — covers Steps 4.1, 4.2
3. **Testing** — Coverage gaps, test quality, missing integration tests — covers Step 4.4
4. **Language-Specific** — Language-specific gotchas, idiom violations — covers Step 4.5
5. **Documentation** — Stale docs, undocumented public surface, drifted paths and counts — covers Step 4.6
6. **Observability** — Missing instrumentation, silent error paths, absent SLOs and alerts — covers Step 4.7

#### Workflow
1. Execute Steps 1-3 (identify changes, gather context, detect scope)
2. Dispatch all six agents below in parallel, in a single message, on every review. Give each the same frozen packet: the diff, the changed-file list, and the branch name
3. Deduplicate findings across perspectives, resolve contradictions; verify each agent's findings against the diff
4. Produce Step 5 structured report + Step 6 disposition ladder

#### Agent per Perspective

Delegate each perspective to its specialist agent (each loads the listed skills + references).

Every dispatched agent **reports only** — it returns findings and edits nothing. The report-only default in Code Review Patterns governs the agents that load it; disposition is the outer agent's job (Step 6). An agent whose lens does not apply to this diff returns "no findings — surface not present" rather than manufacturing material.

| Perspective | Agent | Loads |
|---|---|---|
| Security | `security-reviewer` | code-review-patterns + `.agents/references/security/` (security-analysis, auth-implementation-patterns, secrets-management) |
| Code Quality | `code-reviewer` | code-review-patterns, output-completeness + `.agents/references/workflow/`; apply `code-quality` for smell detection |
| Testing | `test-writer` | test-driven-development, language-testing-patterns + `.agents/references/testing/` |
| Language-Specific | `language-specialist` | language-testing-patterns, test-driven-development + `.agents/references/languages/` for the languages the diff touches |
| Documentation | `documentation-writer` | documentation-validation, post-ship-doc-sync + `.agents/references/documentation/` |
| Observability | `devops-engineer` | `.agents/references/devops/` (observability, sre-practices, incident-management) + `.agents/references/architecture/error-handling-patterns` |

---

### Adversarial Debate Mode (optional — high-risk diffs)

For risky or high-stakes diffs, escalate the multi-perspective pass into a debate:

1. **Freeze a shared packet** — the diff plus context, identical for every reviewer.
2. **Fan out the perspective agents blind to each other** (scale the count to change risk).
3. **Cross-critique for 1-2 rounds** — broadcast round-1 findings; each reviewer may revise, but any change of position **must state a technical reason** ("good point" is banned).
4. **Report survivors and disputes** — findings that survive cross-critique are high-confidence; flag unresolved disputes for the human rather than forcing consensus.

Every finding cites `file:line`. Conclude with a **PASS / CONCERNS / FAIL / BLOCKED** verdict and P0-P3 severities. Run the fan-out with the harness Workflow / `deep-research` primitives.

### Cross-References

- Code quality: Code smell detection, anti-pattern identification
- Security analysis: STRIDE model, vulnerability pattern matching
- Auth implementation patterns: Auth review checklist
- Secrets management: Secrets detection patterns
- Language testing patterns: See Language Testing Patterns section below
- Test-driven development: Test design principles
- Verification before completion: See Verification Before Completion section below
- PR comment resolution: Comment response patterns, inline reply workflow
- Language-specific patterns: Auto-detected language-specific review lenses
- Documentation validation: Per-change docs gate and change-type matrix
- Post-ship doc sync: Staleness heuristic behind the documentation perspective
- Observability: Golden signals, metric design, tracing strategy, alerting
- Existing code discipline: See Existing Code Discipline section below

### Existing Code Discipline

Rules for working within an existing codebase. Check these during review.

#### Read Before Modifying

- **Read the entire file** before changing any part of it — not just the section you plan to edit
- Understand the file's structure, conventions, and how your target section relates to the rest
- Check for file-level comments, configuration blocks, or initialization that affects your change

#### Match Existing Patterns

- **Never introduce a new pattern** alongside an existing one without explicitly flagging the inconsistency
- If the codebase uses pattern A for X, use pattern A — even if you prefer pattern B
- If you believe a pattern should change, propose the migration as a separate task

#### Understand Before Deleting

Code may be used in ways not visible through static analysis:
- Reflection, dynamic dispatch, string-based lookup
- External consumers (APIs, plugins, downstream repos)
- Build scripts, code generation, or test infrastructure
- Feature flags or environment-conditional paths

**If unsure whether code is used: ask, don't delete.**

#### Scope Guard

- If a "small fix" grows mid-implementation — **STOP and ask**
- Define your change boundary before starting; resist scope creep
- A fix that touches 3 files should not silently become a fix that touches 12

#### Separate Refactoring from Features

- Different commits minimum, different branches preferred
- Never mix behavior changes with structural changes — reviewers can't tell what's intentional
- Refactoring should be verifiable independently (tests still pass, behavior unchanged)

#### Surface Hidden Assumptions

Watch for and document when you encounter:
- Implicit ordering dependencies (init before use, A before B)
- Undocumented invariants (field X is always non-null after method Y)
- Concurrency assumptions (single-threaded, lock held, queue ordering)
- Environment assumptions (only works on macOS, requires specific env vars)

### Language-Specific Review Gotchas

Quick-reference of common language-specific issues to watch for during code review.

#### Python
- Mutable default arguments (`def fn(items=[])` -- use `None`)
- Bare `except:` catching everything
- Mutable class attributes shared across instances
- Late binding closures in loops
- Iterator exhaustion (consuming a generator twice)

#### TypeScript / JavaScript
- `any` type defeating type safety (use `unknown`)
- Unhandled async errors (missing try/catch on await)
- Prop mutation in React components
- `==` vs `===` coercion bugs
- Prototype pollution

#### Go
- Goroutine leaks (missing context cancellation)
- Unchecked errors (`err` ignored)
- Nil pointer dereference
- `defer` in loops (resource accumulation)

#### Bash
- Unquoted variables causing word splitting
- Missing `set -euo pipefail`
- Using `[ ]` instead of `[[ ]]`

#### Swift
- Retain cycles (missing `[weak self]`)
- Force unwrapping (`!`) without safety check
- Main thread violations for UI updates

#### Rust
- Unnecessary `clone()` defeating borrow checker
- `unsafe` blocks without justification
- Lifetime issues from overly complex borrowing

---

## Verification Before Completion

**Core principle:** Evidence before claims, always.

**Violating the letter of this rule is violating the spirit of this rule.**

### The Iron Law

```
NO COMPLETION CLAIMS WITHOUT FRESH VERIFICATION EVIDENCE
```

If you haven't run the verification command in this message, you cannot claim it passes.

### The Gate Function

```
BEFORE claiming any status:
1. IDENTIFY: What command proves this claim?
2. RUN: Execute the FULL command (fresh, complete)
3. READ: Full output, check exit code, count failures
4. VERIFY: Does output confirm the claim?
   - If NO: State actual status with evidence
   - If YES: State claim WITH evidence
5. ONLY THEN: Make the claim
```

### Common Failures

| Claim | Requires | Not Sufficient |
|-------|----------|----------------|
| Tests pass | Test output: 0 failures | Previous run, "should pass" |
| Linter clean | Linter output: 0 errors | Partial check, extrapolation |
| Build succeeds | Build: exit 0 | Linter passing |
| Bug fixed | Original symptom: passes | Code changed, assumed fixed |
| Agent completed | VCS diff shows changes | Agent reports "success" |
| Requirements met | Line-by-line checklist | Tests passing |

### Red Flags - STOP

- Using "should", "probably", "seems to"
- Expressing satisfaction before verification
- About to commit/push/PR without verification
- Trusting agent success reports
- Relying on partial verification
- **ANY wording implying success without having run verification**

### Rationalization Prevention

| Excuse | Reality |
|--------|---------|
| "Should work now" | RUN the verification |
| "I'm confident" | Confidence is not evidence |
| "Just this once" | No exceptions |
| "Linter passed" | Linter is not compiler |
| "Agent said success" | Verify independently |
| "Partial check is enough" | Partial proves nothing |

### Key Patterns

```
Tests:     Run -> See "34/34 pass" -> THEN claim "All tests pass"
Red-Green: Write -> Run (pass) -> Revert -> Run (MUST FAIL) -> Restore -> Run (pass)
Build:     Run build -> See exit 0 -> THEN claim "Build passes"
Requirements: Re-read plan -> Checklist -> Verify each -> Report
Agent:     Agent reports -> Check VCS diff -> Verify changes -> Report actual state
```

### When To Apply

**ALWAYS before:** Any success/completion claim, any positive statement about work state, committing, PR creation, task completion, moving to next task, delegating to agents.

**No shortcuts. Run the command. Read the output. THEN claim the result.**

---

## Language Testing Patterns

### Universal Principles

#### What to Unit Test
- Pure functions, transformations, business logic
- Complex conditionals and state transitions
- Error handling paths
- Edge cases: empty arrays, null/undefined, boundary values

#### What NOT to Unit Test
- Simple getters/setters, pass-through functions
- Framework internals (React rendering, Express routing)
- Implementation details -- test behavior, not structure
- Config/settings values (defaults, env var assignments, constants)
- Constructor assignments (`this.x = x` tests the language, not your code)
- Route/endpoint registration (test handler logic instead)
- Enum values and constants
- "Renders without crashing" with no behavior assertion
- Test code (test helpers, fixtures, factories, mocks, test utilities)
- Wiring/glue code with no logic

**Every test must exercise a decision point, transformation, or behavior path.**

#### Test User Stories, Not Internals
- Focus tests on verifying key **user stories / user needs**, not implementation details
- Test **public interfaces / APIs** -- not private methods or internal state
- Coverage hierarchy: **important user story coverage > branch coverage > line coverage**
- Write a failing test for user-reported bugs **before** fixing
- Avoid testing trivial functionality (framework-generated getters/setters, `@ConfigurationProperties` classes, constructor assignments)

#### Coverage Opinion
- 80% line coverage as gate, focus on branch coverage for business logic
- High coverage != well-tested. Missing edge cases matters more than line count.
- Exclude: `.d.ts`, config files, generated code, migrations, `__repr__`, `if TYPE_CHECKING`, test files, test helpers, test factories

#### Factory Fixtures Over Inline Data
```python
# Python with faker
@pytest.fixture
def make_user(db_session):
    def _make_user(**kwargs):
        user = UserFactory.build(**kwargs)
        db_session.add(user)
        db_session.flush()
        return user
    return _make_user
```

```typescript
// JavaScript with faker
function createUser(overrides?: Partial<User>): User {
  return {
    id: faker.string.uuid(),
    name: faker.person.fullName(),
    email: faker.internet.email(),
    ...overrides,
  };
}
```

**Why**: Returns a callable -- tests create exactly what they need. Avoids "magic values" scattered across tests.

### Testing Pyramid (Shift Left)

```
        /  E2E  \          Expensive, slow, run infrequently (release testing)
       /----------\
      / Integration \       Moderate cost, run in CI
     /----------------\
    /    Unit Tests     \   Cheap, fast, run early and often
   /____________________\
```

- **Unit tests**: Bulk of test coverage. Fast, isolated, catch logic errors early
- **Integration tests**: Verify component interactions (DB, APIs, message queues). Run in CI
- **E2E tests**: Validate key user stories end-to-end. Most expensive, run for release verification
- **Shift left**: Identify defects as early as possible where they're cheapest to fix
- Rule of thumb: if a bug can be caught by a unit test, don't rely on integration/E2E to find it

### Ship Test Utilities with Components

When writing libraries or shared components, provide test utilities that make it easy for consumers to test:

- **In-memory fakes / test doubles** for your classes (e.g., `InMemoryUserRepository` alongside `UserRepository`)
- **Context managers / test fixtures** (Python: pytest fixtures; JS: setup helpers) to auto-configure test doubles
- **Spring Boot**: provide auto-configuration for test doubles via `@TestConfiguration`
- **Why**: lowers the barrier for consumers to write tests, promotes uniformity in testing patterns across the codebase

### Language-Specific Patterns

For detailed language-specific patterns, see the corresponding reference files:

- **Python (pytest)**: See `.agents/references/testing/python-testing-patterns.md` -- fixtures, monkeypatch, parametrize, conftest strategy, CI markers
- **JavaScript/TypeScript (Vitest/Jest)**: See `.agents/references/testing/javascript-testing-patterns.md` -- DI over module mocking, async testing, component testing, msw, mock hygiene

#### Python Quick Reference
- pytest + pytest-asyncio + pytest-cov
- `monkeypatch` > `unittest.mock` (auto-reverts)
- Patch where it's used, not where it's defined
- Always use `spec=True` when mocking classes
- `yield` + cleanup in fixtures, `rollback()` not `commit()`

#### JS/TS Quick Reference
- Vitest for Vite projects, Jest otherwise
- DI > module mocking (`vi.mock` is a last resort)
- `userEvent` > `fireEvent`, `getByRole` > `getByTestId`
- Always `await` async assertions
- `vi.clearAllMocks()` in `beforeEach`, not `afterEach`

### Test Generation Patterns

#### Naming Convention

Test names describe **behavior**, not implementation:

| Pattern | Example |
|---|---|
| `should [behavior] when [condition]` | `should_reject_login_when_password_expired` |
| `test_{function}_{scenario}_{expected}` | `test_calculate_discount_bulk_order_20pct` |

Avoid: `test_method_name`, `testCase1`, names referencing internal method names.

#### Arrange-Act-Assert Structure
```python
def test_user_creation_with_valid_data():
    # Arrange
    data = {"name": "Alice", "email": "alice@example.com"}

    # Act
    user = create_user(data)

    # Assert
    assert user.name == "Alice"
    assert user.email == "alice@example.com"
```

#### Coverage Gap Detection Workflow
1. Run coverage: `pytest --cov=src --cov-report=json`
2. Parse JSON for `missing_lines` per file
3. Prioritize by complexity: branches > lines, business logic > utils
4. Generate tests for uncovered paths

#### Mock Generation
```python
@pytest.fixture
def mock_api_client():
    mock = Mock(spec=APIClient)
    mock.fetch.return_value = {"status": "ok"}
    return mock
```

- Always use `spec=` to catch attribute errors
- Return realistic data shapes, not `"mocked_result"`

### Gotchas

#### Python
- Fixture scope leaks: module/session fixtures with mutable state
- `autouse` fixtures create invisible dependencies
- Patching at wrong location (where defined vs. where used)
- Missing `yield` in fixtures (cleanup never runs)
- High coverage on `tests/` directory (meaningless, exclude it)

#### JavaScript
- Using `fireEvent` instead of `userEvent` (misses real interactions)
- Snapshot tests for components (maintenance burden, no value)
- Module mocking when DI would work (breaks on refactors)
- Not awaiting async assertions (tests pass when they shouldn't)
- `data-testid` as first choice (tests implementation, not behavior)
