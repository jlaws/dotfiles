---
name: code-migration
description: "Use when migrating code across frameworks or platforms."
allowed-tools: Read, Grep, Glob, Bash, Edit, Write
skills:
  - test-driven-development
---

# Code Migration

## The Iron Rule

```
ASSESS BEFORE PLANNING. PLAN BEFORE CODING. TEST BEFORE CUTTING OVER.
```

Complete the assessment phase before migrating any code -- skipping it leads to mid-migration surprises that blow timelines.

## Phase 1: Migration Assessment

### Step 1 — Quantify Scope

```bash
# Gather hard numbers before estimating anything
find src/ -type f | wc -l                          # Total files
cloc src/ --quiet                                  # LOC by language
grep -r "import\|require" src/ | wc -l             # Dependency touchpoints
find src/ -name "*.test.*" -o -name "*_test.*" | wc -l  # Test coverage proxy
```

### Step 2 — Score Complexity (per factor: Low / Medium / High)

| Factor | Low | Medium | High |
|--------|-----|--------|------|
| **Size** | Few files, fits in working memory | Spans several modules | Spans most of the codebase |
| **Coupling** | Loose modules, clear interfaces | Some shared state, moderate coupling | Tight coupling, global state, circular deps |
| **Dependencies** | Few third-party libs, all have equivalents | Some libs need replacement or adaptation | Core libs have no equivalent, custom forks |
| **Business Logic** | Simple CRUD, straightforward rules | Moderate domain logic, some edge cases | Complex rules, financial calcs, compliance |
| **Data** | Schema unchanged or trivial mapping | Schema changes needed, data transformable | Schema redesign, lossy transformations, large volumes |

Judge overall complexity from the worst factor, not an average -- one High on Data or Coupling can dominate effort even if everything else is Low. There's no valid formula that collapses five independent risk dimensions into a single number; naming the riskiest factor is more useful than a score.

### Step 3 — Identify Risk Patterns

Scan the codebase for patterns that cause migration failures:

| Risk Pattern | What to Look For | Why It's Dangerous |
|-------------|------------------|--------------------|
| Global/shared state | `global`, `window.*=`, singletons | Hidden dependencies between modules |
| Dynamic dispatch | `eval`, `getattr`, monkey-patching | Can't statically trace call graph |
| Platform-specific code | OS calls, browser APIs, native modules | May not have equivalents on target |
| Implicit behavior | Convention-based routing, magic methods | Easy to miss during migration |
| Serialized data | Pickle, Marshal, binary formats | Format may not survive version change |

### Step 4 — Produce Assessment Document

```markdown
## Migration Assessment: [Source] → [Target]

**Complexity Rating**: Low / Medium / High (and which factor drove it)
**Estimated Effort**: [range] person-weeks
**Risk Level**: Low / Medium / High

### Scope
- Files affected: N
- LOC to migrate: N
- Dependencies to replace: N (list them)
- Test coverage: X% (measured, not guessed)

### Critical Risks
1. [Risk] — Impact: [what breaks] — Mitigation: [strategy]

### Dependencies Without Direct Equivalents
| Source Dependency | Purpose | Target Replacement | Effort |
|---|---|---|---|

### Unresolved Questions
- [List every ambiguity — do NOT proceed with assumptions]
```

**If ANY unresolved questions exist: present the assessment and ask before proceeding to Phase 2.**

## Phase 2: Strategy Selection

Pick ONE strategy. Do not combine them.

| Strategy | When to Use | Trade-off |
|----------|-------------|-----------|
| **Strangler Fig** | Large codebase, can run old+new side by side | Slower but lowest risk — migrate piece by piece behind a facade |
| **Branch by Abstraction** | Shared codebase, can't run two versions | Introduce abstraction layer, swap implementations underneath |
| **Big Bang** | Small codebase (<20 files) or forced by breaking changes | Fast but high risk — everything migrates at once |
| **Parallel Run** | Data pipelines, APIs where correctness is critical | Run both, compare outputs, cut over when equivalent |

### Strangler Fig Implementation Pattern

```
1. Identify a boundary (API endpoint, page, module)
2. Build new version behind the boundary
3. Route traffic/calls to new version
4. Verify equivalence (tests, monitoring, shadow traffic)
5. Remove old version
6. Repeat for next boundary
```

## Phase 3: Migration Plan

### Simple Migration (low complexity)

| Phase | Tasks |
|-------|-------|
| Preparation | Setup project, install deps, configure build, write comparison tests |
| Core Migration | Migrate module by module, run tests after each |
| Validation | Full test suite, performance comparison, edge cases |

### Complex Migration (medium/high complexity)

| Phase | Tasks |
|-------|-------|
| Foundation | Architecture design, PoC for riskiest component, tool selection |
| Infrastructure | Build pipeline, abstraction layers, dual runtime support |
| Incremental | Module-by-module migration with comparison tests after each |
| Cutover | Remove legacy code, optimize, final validation, rollback drill |

Duration depends on team size, module count, and how much of the touched code already has tests -- estimate from the scope numbers gathered in Step 1, not from a calendar guess.

### Migration Order

Migrate in dependency order — leaves first, roots last:

```
1. Utilities / helpers (no internal dependencies)
2. Data models / types
3. Business logic modules
4. Integration layers (APIs, database, external services)
5. UI / presentation layer
6. Configuration and build system
```

## Phase 4: Testing Strategy

### Comparison Tests (Write These FIRST)

Before migrating any module, write tests that capture current behavior:

```
1. Identify public API of the module (functions, endpoints, events)
2. Write tests against the OLD code that assert current behavior
3. Run tests — they must pass on old code
4. Migrate the module
5. Run the SAME tests against new code — they must still pass
```

This catches behavioral regressions that unit tests miss.

### Test Categories

| Category | What It Catches | When to Run |
|----------|----------------|-------------|
| Comparison tests | Behavioral regression | After each module migration |
| Unit tests | Logic errors in new code | During development |
| Integration tests | Cross-module compatibility | After each migration phase |
| Performance tests | Latency/throughput regression | Before and after cutover |
| Data validation | Data integrity issues | During and after data migration |

## Phase 5: Rollback

### Rollback Triggers — Decide These BEFORE Migrating

| Condition | Threshold | Detection |
|-----------|-----------|-----------|
| Critical functionality broken | Any P0 feature fails | Automated smoke tests |
| Performance degradation | Set relative to your own measured pre-migration p95 baseline -- a guessed percentage has no provenance | APM dashboard |
| Data corruption | Any integrity check failure | Validation job |
| Error rate spike | Set relative to your own measured pre-migration baseline -- a guessed percentage has no provenance | Error tracking |

### Rollback Procedures by Strategy

| Strategy | Rollback Method |
|----------|----------------|
| Strangler Fig | Route traffic back to old module (config change) |
| Branch by Abstraction | Swap implementation back (redeploy) |
| Big Bang | Deploy previous version (CI/CD rollback) |
| Feature Flag | Toggle flag off (config change) |

The mechanism determines the speed: a config or flag change rolls back in the time it takes to propagate; a redeploy takes as long as your deploy pipeline does.

## Large-Scale Migration

For large codebases (>50 files affected), organize migration into independent module batches:

1. Group files by module/package — each batch must be independently migratable
2. Process batches sequentially — migrate one module at a time
3. After each batch: run tests, verify no regressions, commit
4. After all batches: run full test suite, verify cross-module interactions

## Deliverables Checklist

- [ ] Migration assessment with complexity rating
- [ ] Strategy selection with rationale
- [ ] Phased migration plan with timeline
- [ ] Comparison tests for each module (written before migration)
- [ ] Rollback triggers and procedures defined
- [ ] Progress tracking per module
- [ ] Documentation updated for the target stack (README, API reference, config/CLI docs, CHANGELOG) — or N/A per change (see `documentation-validation`)
