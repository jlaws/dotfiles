# Multi-Agent Orchestration

Orchestrate specialized agents to accomplish complex tasks through coordinated workflows. This command provides frameworks for agent performance optimization, issue resolution, git workflows, and multi-system coordination.

## Context
The user needs to leverage multiple specialized agents working together to accomplish complex tasks that require different expertise areas. Focus on agent coordination, context passing, performance optimization, and quality verification.

## Requirements
$ARGUMENTS

---

## Part A: Agent Performance Optimization

Systematically improve agent effectiveness through performance analysis and prompt engineering.

### 1. Performance Analysis

**Gather Baseline Metrics:**
- Task completion rate (successful vs failed)
- Response accuracy and correctness
- Tool usage efficiency (correct tools, call frequency)
- Average response time and token consumption
- User corrections and retries
- Error patterns and failure modes

**Failure Mode Classification:**
- Instruction misunderstanding
- Output format errors
- Context loss in long conversations
- Tool misuse or inefficient selection
- Constraint violations
- Edge case handling failures

### 2. Prompt Engineering Improvements

**Chain-of-Thought Enhancement:**
- Add explicit reasoning steps
- Include self-verification checkpoints
- Implement recursive decomposition for complex tasks

**Few-Shot Example Optimization:**
- Select diverse examples covering common use cases
- Include edge cases that previously failed
- Show positive and negative examples with explanations

**Role Definition Refinement:**
- Clear, single-sentence mission
- Specific expertise domains
- Behavioral traits and interaction style
- Tool proficiency guidelines
- Explicit constraints (what NOT to do)

### 3. Testing and Validation

**A/B Testing Framework:**
```
Config:
  - Agent A: Original version
  - Agent B: Improved version
  - Test set: 100+ representative tasks
  - Metrics: Success rate, speed, token usage
  - Evaluation: Blind human review + automated scoring
```

**Success Criteria:**
- Task success rate improves ≥15%
- User corrections decrease ≥25%
- No increase in safety violations
- Response time within 10% of baseline
- Cost per task doesn't increase >5%

### 4. Version Control and Rollout

**Staged Deployment:**
1. Alpha testing: Internal team (5% traffic)
2. Beta testing: Selected users (20% traffic)
3. Canary release: Gradual increase (20% → 50% → 100%)
4. Monitoring period: 7-day observation window

**Rollback Triggers:**
- Success rate drops >10%
- Critical errors increase >5%
- Cost per task increases >20%
- Safety violations detected

---

## Part B: Multi-Agent Coordination Framework

### 5. Orchestration Principles

- **Parallel Execution**: Run independent agents concurrently
- **Sequential Handoff**: Pass context between dependent agents
- **Minimal Communication Overhead**: Share only necessary context
- **Fault Tolerance**: Handle agent failures gracefully

### 6. Agent Coordination Pattern

```python
class MultiAgentOrchestrator:
    def __init__(self, agents):
        self.agents = agents
        self.results = {}

    def orchestrate(self, task):
        # Phase 1: Parallel analysis
        analysis_results = self.run_parallel([
            ('analyzer', self.agents['analyzer'].analyze, task),
            ('researcher', self.agents['researcher'].research, task)
        ])

        # Phase 2: Sequential implementation
        implementation = self.agents['implementer'].implement(
            task=task,
            context=analysis_results
        )

        # Phase 3: Parallel verification
        verification_results = self.run_parallel([
            ('tester', self.agents['tester'].test, implementation),
            ('reviewer', self.agents['reviewer'].review, implementation)
        ])

        return self.synthesize_results(verification_results)
```

### 7. Context Passing Template

```
Context for {next_agent}:

Completed by {previous_agent}:
- {summary_of_work}
- {key_findings}
- {changes_made}

Remaining work:
- {specific_tasks}
- {files_to_modify}
- {constraints}

Success criteria:
- {measurable_outcomes}
- {verification_steps}
```

### 8. Cost Optimization

- Token usage tracking per agent
- Adaptive model selection based on task complexity
- Result caching and reuse
- Efficient context compression

---

## Part C: Issue Resolution Workflow

5-phase debugging and resolution pipeline using multiple specialized agents.

### Phase 1: Error Analysis

**Agent: error-detective**
```
Analyze: error traces, logs, observability data
Output:
- Error signature (exception type, message pattern)
- Stack trace with key frames highlighted
- Reproduction steps
- User impact assessment
- Timeline analysis
```

### Phase 2: Root Cause Investigation

**Agent: debugger + code-reviewer**
```
Investigate:
- Code path analysis from entry to failure
- Variable state tracking at key decision points
- Git bisect to identify introducing commit
- Dependency compatibility check

Output:
- ROOT_CAUSE: {technical explanation with evidence}
- INTRODUCING_COMMIT: {git SHA if found}
- AFFECTED_FILES: [paths with line numbers]
- FIX_STRATEGY: {recommended approach}
```

### Phase 3: Fix Implementation

**Route to domain agent based on issue type:**
- Python → python-pro
- TypeScript → typescript-pro
- Database → database-optimizer
- Security → security-auditor

**Implementation Requirements:**
- Minimal fix addressing root cause (not symptoms)
- Unit tests for failure case reproduction
- Integration tests for end-to-end behavior
- Structured logging for debugging

### Phase 4: Verification

**Agent: test-automator + performance-engineer**
```
Verify:
- Full test suite execution (unit, integration, e2e)
- Performance benchmarks (p50, p95, p99 latency)
- Security scanning
- Cross-environment testing

Output:
- TEST_RESULTS: {passed, failed, coverage}
- PERFORMANCE_IMPACT: {before/after comparison}
- REGRESSION_DETECTED: {yes/no}
- PRODUCTION_READY: {yes/no + blockers}
```

### Phase 5: Prevention

**Agent: code-reviewer**
```
Document and prevent:
- Static analysis rules to catch similar issues
- Type system enhancements
- Monitoring and alerting setup
- Architecture improvements
- Postmortem (for high-severity incidents)
```

---

## Part D: Git Workflow Orchestration

Coordinated workflow from code review through PR creation.

### Phase 1: Pre-Commit Review

**Agent: code-reviewer**
- Code style violations
- Security vulnerabilities
- Performance concerns
- Missing error handling
- Breaking API changes

### Phase 2: Testing

**Agent: test-automator**
- Execute test suites (unit, integration, e2e)
- Generate coverage report
- Identify untested code paths
- Recommend additional tests

### Phase 3: Commit Message

**Agent: prompt-engineer**
- Categorize changes (feat/fix/docs/refactor/etc.)
- Create Conventional Commits format message
- Include breaking change notices
- Reference related issues

### Phase 4: PR Creation

**Agent: docs-architect + deployment-engineer**
- Generate PR description
- Configure reviewers and labels
- Set up auto-merge rules
- Document deployment notes

### Success Criteria:
- All critical code issues resolved
- Test coverage maintained or improved (>80%)
- All tests passing
- Commit messages follow Conventional Commits
- No merge conflicts
- PR description complete

---

## Configuration Options

### VERIFICATION_LEVEL
- **minimal**: Quick fix with basic tests (~30 min)
- **standard**: Full test coverage + code review (~2-4 hours)
- **comprehensive**: Standard + security audit + performance benchmarks (~1-2 days)

### ROLLOUT_STRATEGY
- **immediate**: Direct production deploy (hotfixes)
- **canary**: Gradual rollout to subset of traffic
- **blue-green**: Full environment switch with rollback
- **feature-flag**: Deploy code, control via flags

### OBSERVABILITY_LEVEL
- **minimal**: Basic error logging
- **standard**: Structured logs + key metrics
- **comprehensive**: Full distributed tracing + dashboards + SLOs

---

## Multi-Domain Coordination Examples

### Example 1: Database Performance Issue

**Sequence:**
1. error-detective → Identify slow queries
2. database-optimizer → Add indexes, optimize queries
3. performance-engineer → Add caching layer
4. devops-troubleshooter → Configure monitoring

### Example 2: Frontend Error in Production

**Sequence:**
1. error-detective → Analyze Sentry reports
2. debugger → Identify API response issue
3. typescript-pro → Fix frontend null handling
4. backend-architect → Fix API contract
5. test-automator → Cross-browser tests

### Example 3: Security Vulnerability

**Sequence:**
1. error-detective → Review security scan
2. security-auditor → Implement parameterized queries, validation
3. test-automator → Add security tests
4. code-reviewer → Document and create postmortem

---

## Output Format

1. **Orchestration Plan**: Agent sequence with context passing
2. **Execution Results**: Each agent's output with status
3. **Verification Summary**: All checks passed/failed
4. **Documentation**: Changes made and prevention measures
5. **Next Steps**: Follow-up tasks if any

Focus on coordinated agent workflows that leverage specialized expertise while maintaining clear context flow and quality verification.
