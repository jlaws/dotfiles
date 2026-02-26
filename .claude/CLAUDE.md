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

## Behavioral Defaults
- Before creative/feature work: explore intent + requirements before implementation
- For design decisions: propose 2-3 approaches, lead with recommendation
- **Skill lookup**: Before implementation tasks involving a specific framework, language pattern, or architecture decision — scan the Skills Index below, then `Read` the matching `SKILL.md` for concrete patterns and examples. Do this before relying on training knowledge.
- Inlined rules below are always-active; skills supplement with code examples, extended catalogs, and domain-specific depth

## Verification Gate

Evidence before claims. Run the command, read the output, THEN claim the result.

### Gate Function

```
BEFORE claiming any status:
1. IDENTIFY — What command proves this claim?
2. RUN     — Execute the FULL command (fresh, complete)
3. READ    — Full output, check exit code, count failures
4. VERIFY  — Does output confirm the claim?
   - If NO: State actual status with evidence
   - If YES: State claim WITH evidence
5. CLAIM   — Only now make the claim
```

### Evidence Requirements

| Claim | Requires | Not Sufficient |
|-------|----------|----------------|
| Tests pass | Test output: 0 failures | Previous run, "should pass" |
| Build succeeds | Build: exit 0 | Linter passing |
| Bug fixed | Original symptom gone in test | Code changed, assumed fixed |
| Requirements met | Line-by-line checklist verified | Tests passing alone |

### Red Flags — STOP

- Using "should", "probably", "seems to"
- Expressing satisfaction before verification
- About to commit/push/PR without verification
- Trusting agent success reports without independent check
- Relying on partial verification
- ANY wording implying success without having run verification

## Code Quality

### Principles

| Principle | Rule |
|-----------|------|
| SRP | One reason to change per function/class |
| DRY | Extract after 2+ duplicates, not before |
| YAGNI | Solve today's problem, not tomorrow's hypothetical |
| Composition > Inheritance | Prefer protocols/interfaces |
| Explicit > Implicit | Clarity beats cleverness |
| Favor Uniformity | One way to do each thing; migrate quickly + add checks to prevent reversion |
| Follow Ecosystem Patterns | Go all-in on chosen framework's philosophy and idioms |
| External Configuration | Enable external config for components; follow ecosystem patterns |

### Code Smells

- **Naming**: Booleans `is`/`has`/`can`/`should` prefix; functions verb prefix; no abbreviations
- **Functions**: Single responsibility, <30 lines, max 3 params (use param object beyond), minimize side effects
- **Complexity**: Max 2 levels nesting; early returns; replace conditional chains with lookup maps/polymorphism

### Make Invalid States Unrepresentable
- Use generics/type hints to catch issues at compile-time/static analysis
- No `any` in TS (use `unknown`); no force unwraps in Swift (unless provably safe)
- Use `Optional`/`Option` for null safety — never return bare `None`/`null` when absence is possible
- Validate early at boundaries, convert to constrained types, pass constrained types downstream
- Priority: **compile-time > static analysis > runtime**

### Anti-Patterns

**Code**: Premature abstraction (wait for 2+) · God objects (split by responsibility) · Magic values (named constants) · Swallowed exceptions · Commented-out code (delete it, git has history)

**Process**: Large PRs · Skipping tests · Vague commits · TODOs without context/ticket

### Style Defaults

| Rule | Value |
|------|-------|
| Indentation | 2 spaces (no tabs) |
| Line endings | LF (Unix) |
| Final newline | Always |
| Line length | 80-100 soft limit |
| File size | Under 300 lines |

**Naming**: JS/TS/Swift = `camelCase`, Python/Rust/Go = `snake_case`, Types = `PascalCase`, Constants = `SCREAMING_SNAKE_CASE`

**Import order** (blank line separated): 1. Standard library → 2. Third-party → 3. Local modules

## Error Handling

### Pattern Selection

1. Can the caller reasonably recover? → Result type or checked exception
2. Is this a programming bug? → Panic/crash (fail fast)
3. Is this crossing a system boundary? → Error codes with metadata
4. Is this just "no value"? → Option type, not null

### Universal Rules

- **Fail fast, fail loud** — validate at boundaries immediately; don't propagate bad data into business logic
- **Handle at the right level** — catch where you can meaningfully act (retry, fallback, user message); don't catch just to log and re-throw
- **Preserve context** — wrap errors: `"failed to create user: <original>"` with chaining (`from e`, `%w`, `cause`)
- **Don't swallow errors** — `except Exception: pass` is never acceptable; handle meaningfully or propagate
- **Log appropriately** — Error: unexpected failures; Warning: expected failures handled; don't log every caught exception

## Shell Script Safety

### Mandatory Preamble

```bash
#!/bin/bash
set -Eeuo pipefail
trap 'echo "Error on line $LINENO" >&2' ERR
trap 'rm -rf -- "$TMPDIR"' EXIT
```

`-E` ERR trap inherited · `-e` exit on error · `-u` exit on undefined var · `-o pipefail` pipe fails if any cmd fails

### Variable Safety

- Always quote variables: `"$var"`
- Required var with message: `: "${REQUIRED_VAR:?not set}"`
- Default value: `: "${OPTIONAL:=default}"`
- Safe test (prevents `-u` trigger): `[[ -z "${VAR:-}" ]]`

### Key Gotchas

- `[[ ]]` not `[ ]` — safer, supports `&&`/`||`/regex
- `command -v` not `which` — POSIX-compliant
- `printf` not `echo` — predictable across systems
- Separate `local` from command substitution: `local val; val=$(cmd)`
- Idempotent design — scripts safe to rerun (`mkdir -p`, check before create)
- \>100 lines → rewrite in Python/Go

## TDD Discipline

### The Iron Law

```
NO PRODUCTION CODE WITHOUT A FAILING TEST FIRST
```

Write code before the test? Delete it. Start over. No exceptions without human partner's permission.

### Red-Green-Refactor

1. **RED** — Write one minimal failing test showing desired behavior
2. **Verify RED** — Run test, confirm it fails for the right reason (feature missing, not typo). **MANDATORY.**
3. **GREEN** — Write simplest code to pass. Nothing more.
4. **Verify GREEN** — Run test, confirm pass + no regressions. **MANDATORY.**
5. **REFACTOR** — Clean up (after green only). Keep tests green. Don't add behavior.

### Red Flags — Start Over

- Code before test · Test passes immediately · Rationalizing "just this once"
- "I'll write tests after" · "Keep as reference" · "Already spent X hours"

### Exceptions (ask human partner)

Throwaway prototypes · Generated code · Configuration files

## Refactoring Discipline

### The Rule

```
TEST → REFACTOR → VERIFY → COMMIT
Never skip a step. Never combine refactoring with behavior changes.
```

### The Cadence

1. Ensure tests pass (green baseline)
2. Make ONE structural change
3. Run tests (must still pass — if not, revert immediately)
4. Commit (small atomic commit describing the refactoring)
5. Repeat

### When NOT to Refactor

| Situation | Why Not |
|-----------|---------|
| No tests covering the code | Can't verify behavior preservation. Write tests first. |
| Under deadline pressure | Ship first, refactor next sprint. |
| Code being deleted soon | Don't polish what you're throwing away. |
| "While I'm in here..." scope creep | File a ticket, do it separately. (Exception: small boy-scout improvements.) |
| Single implementation | Don't create abstractions for one concrete case. Wait for the second. |

## Code Review

### Feedback Severity Labels

```
[blocking]    Must fix before merge
[important]   Should fix, discuss if disagree
[nit]         Nice to have, not blocking
[suggestion]  Alternative approach to consider
[learning]    Educational, no action needed
```

### Feedback Techniques

- **Ask questions** instead of stating problems: "What happens if `items` is empty?"
- **Suggest, don't command**: "Would it make sense to extract this? It appears in 3 places."
- **Be specific and actionable**: "Race condition when concurrent access — consider a mutex here."

### Review Process (time-boxed)

1. **Context** (2-3 min) — Read PR description, check size (<400 lines), CI status
2. **High-level** (5-10 min) — Solution fit, consistency, test coverage
3. **Line-by-line** (10-20 min) — Logic, security, performance, maintainability
4. **Summary** (2-3 min) — Key concerns, what worked well, clear verdict

## Git Workflow
- Commit messages: freeform imperative mood, <72 char subject, no period
- Prefer small, atomic commits
- Always verify changes with `git diff` before committing
- Never force push to main/master
- Branch naming: `type/short-description` (e.g., `fix/login-timeout`)

## Code Defaults
- Explicit over implicit; fail fast over silent errors
- No TODO without issue/ticket reference
- Composition over inheritance
- Test co-located with source when possible

## Team Conventions
When spawned as a teammate, follow these rules (teammates read this file on startup):
- **Task discipline**: claim via TaskUpdate (set owner), mark completed when done, check TaskList for next work
- **File ownership**: only edit files declared in your task — never touch files outside your assignment
- **Communication**: DM the lead via SendMessage; never broadcast unless truly critical (blocking issue affecting all agents)
- **Quality**: verify your work (run tests, read output) before marking a task complete
- **Shutdown**: respond to `shutdown_request` promptly — approve unless you have in-flight uncommitted work
- **Context**: include file paths and line numbers when referencing code in messages

### Subagents vs Teams

| Use | When |
|-----|------|
| **Task tool (subagent)** | Independent, self-contained work: research, exploration, single-file edits, running tests |
| **TeamCreate (full team)** | Coordinated multi-file work requiring shared task list, communication, and file ownership |

**Default to subagents** unless tasks have cross-file dependencies or require coordination.

### File Conflict Prevention
- Declare file ownership in task descriptions — one agent per file
- If you need to edit an unowned file, DM the lead first
- Never edit files another agent is working on

### Common Prompt Mistakes

| Mistake | Fix |
|---------|-----|
| Vague task description | Include specific files, acceptance criteria, and constraints |
| No file ownership declared | List exact files each agent may edit |
| Broadcasting status updates | DM the lead; only broadcast blocking issues |
| Skipping verification | Always run tests/build before marking complete |

---

## Skills Index

Consult for code examples, extended catalogs, and domain-specific depth beyond the inlined rules above.

All skills live at `.claude/skills/{category}/{skill}/SKILL.md`.

### /commands (user-initiated)
- /audit — Security threat model and vulnerability scan
- /debug — Systematic bug investigation
- /diff-review — Multi-perspective code review
- /paper-analysis — Research paper analysis
- /pr-fix — Resolve PR reviewer comments
- /skill-audit — Audit skills for conformance to Anthropic's guide
- /team-design — Multi-agent system design suite
- /team-investigate — Competing hypothesis debugging with agent teams
- /team-review — Multi-agent team code review

### architecture
api-client-sdk-design | api-design-principles | architecture-decision-records | architecture-patterns | background-job-processing | caching-strategies | error-handling-patterns | event-sourcing-patterns | grpc-and-protobuf | mcp-server-development | message-queue-patterns | microservices-patterns | ml-system-design | notification-systems | real-time-systems | saas-multi-tenancy

### ai-ml
agentic-systems-design | ai-safety-and-alignment | causal-inference-ml | continual-and-online-learning | dataset-management | demo-and-prototype-building | diffusion-model-patterns | distributed-training-at-scale | edge-and-mobile-ml | embedding-and-representation-learning | eval-and-benchmarking | federated-learning | graph-neural-networks | jax-patterns | llm-application-patterns | llm-fine-tuning | llm-pretraining | llmops-production-monitoring | ml-experiment-lifecycle | ml-model-deployment | model-compression | model-interpretability | multimodal-ml | neural-architecture-search | privacy-preserving-ml | probabilistic-programming | pytorch-ml-training | rag-and-vector-search | reinforcement-learning-patterns | structured-output-patterns | synthetic-data-generation | time-series-ml | tokenizer-design | transformer-architecture-design

### data
airflow-dag-patterns | data-lake-architecture | data-quality-frameworks | database-migration | dbt-transformation-patterns | eda-and-visualization | feature-store-design | jupyter-notebook-patterns | ml-pipeline-orchestration | nosql-data-modeling | notebook-to-production | postgresql-table-design | search-infrastructure | spark-optimization | sql-optimization-patterns | streaming-data-processing | web-scraping-and-data-collection

### devops
docker-patterns | github-actions-patterns | gitops-workflow | incident-management | k8s-manifest-generator | k8s-security-policies | monorepo-tools | observability | pipeline-design | terraform-module-library

### languages
async-python-patterns | bash-defensive-patterns | browser-extension-development | cli-tool-development | cuda-gpu-programming | fastapi-templates | go-concurrency-patterns | js-ts-patterns | nodejs-backend-patterns | pydantic-and-data-validation | python-packaging-and-distribution | python-patterns | rust-project-patterns | swift-patterns

### frontend
accessibility-testing | design-system-patterns | form-patterns | graphql-client-patterns | i18n-and-localization | nextjs-app-router-patterns | react-native-architecture | react-state-management | responsive-web-design | svelte-patterns | tailwind-design-system | web-animation-patterns

### testing
debugging-methodology | e2e-testing-patterns | language-testing-patterns | load-testing-and-perf | performance-profiling | shell-testing | test-driven-development

### security
auth-implementation-patterns | compliance-and-data-privacy | dependency-auditing | secrets-management | security-analysis

### workflow
claude-code-meta-patterns | code-quality | code-review-excellence | diff-review | feature-flags-and-ab-testing | github-issue-resolution | multi-agent-development | pr-comment-resolution | refactoring-patterns | technical-debt-remediation | using-git-worktrees | verification-before-completion | writing-skills

### research
confidence-scoring | latex-paper-writing | literature-review | paper-analysis-methodology | paper-to-code-implementation | statistical-analysis

### business
analytics-instrumentation | hiring-and-interviews | kpi-dashboard-design | mvp-development-patterns | payment-systems | team-onboarding

### cloud
cost-optimization | file-storage-patterns | gpu-compute-management | multi-cloud-architecture | serverless-patterns

### documentation
changelog-automation | openapi-spec-generation | technical-writing-for-devtools

### migration
code-migration | dependency-upgrade
