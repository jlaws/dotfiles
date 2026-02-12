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
- Skills (.claude/skills/) are deep-dive references — read the relevant SKILL.md when a task needs domain-specific depth, not as a prerequisite for every response

## Verification Gate
Evidence before claims. Run the command, read the output, THEN claim the result.
"Should work" / "I'm confident" = not evidence. No completion claims without fresh verification.
Depth: .claude/skills/workflow/verification-before-completion/SKILL.md

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

---

## Skills Index

All skills live at `.claude/skills/{category}/{skill}/SKILL.md`.

### /commands (user-initiated)
- /audit — Security threat model and vulnerability scan
- /debug — Systematic bug investigation
- /diff-review — Multi-perspective code review
- /paper-analysis — Research paper analysis
- /pr-fix — Resolve PR reviewer comments
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
