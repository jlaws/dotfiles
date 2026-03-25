---
name: workflow-pr-fix
description: "Resolve all PR reviewer comments — categorize, fix, reply inline, verify, and push. Use when you have open PR review comments to address. Do NOT use if there are no comments yet (wait for review)."
argument-hint: "<pr-number-or-url>"
---

Resolve PR comments using the methodology below: $ARGUMENTS

---

## PR Comment Resolution

Fetch all reviewer comments on a PR, categorize them, implement fixes, reply inline, verify, and push.

### Step 1 — Identify PR

```
IF $ARGUMENTS contains PR number or URL:
  Use provided PR identifier
ELSE:
  gh pr view --json number,url,baseRefName,headRefName

FAIL FAST if no open PR found for current branch.
```

Extract: PR number, base branch, head branch, repo owner/name.

```bash
gh repo view --json owner,name --jq '.owner.login + "/" + .name'
```

### Step 2 — Gather Full Context

Collect all PR data. GitHub stores comments in 3 separate APIs — fetch ALL of them.

```bash
# PR metadata + diff
gh pr view $PR --json title,body,author,reviewRequests,reviews,labels,files
gh pr diff $PR

# Changed files list (for scope guard)
gh pr view $PR --json files --jq '.files[].path'

# Conversation comments (top-level PR discussion)
gh api repos/{owner}/{repo}/issues/$PR/comments

# Inline review comments (line-level feedback)
gh api repos/{owner}/{repo}/pulls/$PR/comments

# Review verdicts and summaries
gh api repos/{owner}/{repo}/pulls/$PR/reviews
```

Read every changed file in full to understand the surrounding context, not just the diff hunks.

### Step 3 — Categorize Comments

Cross-reference the Code Review Patterns severity labels (see section below) when classifying.

```
For each comment/review:
  - blocking   — Must fix before merge
  - question   — Needs a response (may or may not need code change)
  - suggestion — Alternative approach worth considering
  - nit        — Minor style/preference, non-blocking
  - resolved   — Already addressed or outdated
  - unclear    — Cannot determine intent; needs clarification
```

#### Detect Language Context

Inspect file extensions of changed files and cross-reference matching language-specific patterns for implementation guidance:

| Extension | Pattern Set |
|-----------|-------------|
| `.py` | Python patterns |
| `.js`, `.ts`, `.tsx` | JS/TS patterns |
| `.go` | Go concurrency patterns |
| `.sh` | Bash defensive patterns |
| `.swift` | Swift patterns |
| `.rs` | Rust project patterns |

#### Produce Fix Plan

```
ORDERED FIX PLAN:
1. [blocking]   — Description (file:line) — from: @reviewer
2. [blocking]   — ...
3. [suggestion] — ...
4. [nit]        — ...

UNCLEAR (need clarification before implementing):
- Comment #id: "..." — What is unclear
```

**If ANY comment is classified as `unclear`: present the fix plan and ask for clarification BEFORE implementing anything.**

### Step 4 — Implement Fixes

#### Execution order
1. Blocking issues first
2. Simple fixes (typos, imports, naming)
3. Complex fixes (refactoring, logic changes)
4. Nits last

#### Rules
- **Scope guard**: Only touch files changed in the PR. If a fix requires changes outside PR scope, note it in the summary as deferred.
- **Atomic commits**: One commit per logical fix group. Use imperative mood, reference the comment.
- **Reply to threads inline**: Use the GitHub API to reply in the reviewer's thread, not as top-level comments.

```bash
# Reply to an inline review comment thread
gh api repos/{owner}/{repo}/pulls/$PR/comments/{comment_id}/replies \
  -f body="Fixed — [brief description of change]. See [commit_sha_short]."

# Reply to a top-level issue comment
gh api repos/{owner}/{repo}/issues/$PR/comments \
  -f body="Addressed — [brief description]."
```

#### For questions (no code change needed)
Reply with a clear, concise answer in the thread. Don't make unnecessary code changes.

### Step 5 — Verify & Push

Apply the Verification Before Completion methodology (see section below) — evidence before claims.

```
1. Run project tests (identify test command from package.json, Makefile, etc.)
2. Run linter if configured
3. Review full diff: git diff $(git merge-base HEAD origin/{base_branch})..HEAD
4. Confirm no unintended changes leaked in
5. Push
6. Watch CI: gh pr checks $PR --watch
7. Re-request review from original reviewers:
   gh pr edit $PR --add-reviewer {reviewer1},{reviewer2}
```

If tests or CI fail: diagnose, fix, add another commit, re-verify. Do not push failing code.

### Step 6 — Summary

Present to the user:

```markdown
## PR Comment Resolution Summary

### Comments Addressed
- [blocking] {description} — {commit_sha_short} (replied to @reviewer)
- [suggestion] {description} — {commit_sha_short}
- ...

### Questions Answered
- @reviewer: "{question}" — replied with explanation

### Deferred Items
- {description} — Reason: {out of scope / needs design discussion / ...}

### Verification
- Tests: {PASS/FAIL — evidence}
- Lint: {PASS/FAIL — evidence}
- CI: {PASS/FAIL/PENDING — link}
- Review re-requested: {reviewers}
```

Do not auto-file issues for deferred items. List them for the user to decide.

### Multi-Perspective Review

For thorough coverage, analyze the diff sequentially from each perspective:

1. **Security** — STRIDE analysis, vulnerability patterns, secrets detection — covers Step 4.3
2. **Code Quality** — Code smells, edge cases, error handling, naming, DRY — covers Steps 4.1, 4.2
3. **Testing** — Coverage gaps, test quality, missing integration tests — covers Step 4.4
4. **Language-Specific** — Language-specific gotchas, idiom violations — covers Step 4.5

#### Workflow
1. Execute Steps 1-3 (identify changes, gather context, detect scope)
2. Analyze the full diff from each perspective above, sequentially
3. Deduplicate findings across perspectives, resolve contradictions
4. Produce Step 5 structured report + Step 6 decision gate

#### Skills per Perspective

| Perspective | Skills/References to Load |
|---|---|
| Security | `security:security-analysis`, `security:auth-implementation-patterns` |
| Code Quality | `workflow:code-quality`, `workflow:code-review-patterns` |
| Testing | `testing:language-testing-patterns`, `testing:test-driven-development` |
| Language-Specific | Auto-detected `languages:*-patterns` based on file extensions |

---

### Cross-References

- Code review patterns: Severity labels, comment response patterns — see Code Review Patterns section below
- Verification before completion: Verification gate before any completion claims — see Verification Before Completion section below
- Language-specific patterns: Auto-detected language-specific implementation guidance

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

### What Looks Good
- {positive observation}
```

Omit empty severity sections. Always include "What Looks Good".

#### Step 6 — Decision Gate

**Default: report only.** Do NOT automatically implement fixes.

After presenting findings, ask:
1. Implement fixes for findings above
2. Nothing — review complete

If implementing, follow PR Comment Resolution Step 4 (scope guard, atomic commits, verify before push).

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

- **Python (pytest)**: See `references/testing/python-testing-patterns.md` -- fixtures, monkeypatch, parametrize, conftest strategy, CI markers
- **JavaScript/TypeScript (Vitest/Jest)**: See `references/testing/javascript-testing-patterns.md` -- DI over module mocking, async testing, component testing, msw, mock hygiene

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
`test_{function}_{scenario}_{expected_result}`

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
