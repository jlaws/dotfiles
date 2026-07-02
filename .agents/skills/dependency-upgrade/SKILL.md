---
name: dependency-upgrade
description: "Dependency upgrade strategy and risk assessment across all ecosystems (JS/TS, Python, Rust, Go). Use when upgrading major dependencies, resolving conflicts, planning migration paths, or managing version stepping. Do NOT use for security vulnerability scanning (use dependency-auditing)."
skills:
  - test-driven-development
  - verification-before-completion
---

# Dependency Upgrade

## Golden Rule: One Major at a Time

Never upgrade multiple major versions simultaneously. Step through each major version in sequence.

**Upgrade order**: core language/runtime → framework → routing/state/ORM → testing → build tools → dev tooling

Each upgrade = separate branch + full test run + git tag before starting.

## Risk Assessment Matrix

| Risk Factor | Low | Medium | High |
|-------------|-----|--------|------|
| **Breaking changes** | Deprecation warnings only | API renames, config changes | New paradigms, rewrite required |
| **Ecosystem impact** | No peer/transitive deps affected | 2-3 deps need co-upgrading | Entire plugin ecosystem changes |
| **Test coverage** | >80% on affected code | 50-80% | <50% |
| **Rollback cost** | Revert one commit | Revert + data migration | Can't rollback (schema changes) |
| **Models/types affected** | <10 | 10-50 | >50 (batch with codemods) |

High risk on ANY factor = dedicated spike, staging validation, phased rollout.

## Phase 1: Assessment

1. **Read migration guides** for every major version between current and target (not just the target version)
2. **Check dependency compatibility** — peer deps, transitive deps, plugin ecosystem
3. **Search GitHub issues** for the specific upgrade path — others have hit the gotchas
4. **Inventory affected code** — run grep/ast searches to quantify the blast radius
5. **Run existing test suite** as baseline — record pass count, coverage %
6. **Review supply chain risk** — see `references/security-audit-and-supply-chain.md`

### Ecosystem-Specific Assessment

| Ecosystem | Lock File | Compatibility Check | Codemod Tools |
|-----------|-----------|-------------------|---------------|
| **npm/pnpm/yarn** | `package-lock.json`, `pnpm-lock.yaml`, `yarn.lock` | `npm outdated`, `pnpm outdated` | Library-specific (next, react, eslint codemods) |
| **Python (uv/pip)** | `uv.lock`, `requirements.txt` | `uv pip compile --dry-run`, `pip-audit` | `bump-pydantic`, `django-upgrade`, `pyupgrade` |
| **Rust (cargo)** | `Cargo.lock` | `cargo outdated`, `cargo tree` | `cargo-upgrade`, `cargo-edit` |
| **Go (modules)** | `go.sum` | `go list -m -u all` | `go fix`, `gopls` |
| **Ruby (bundler)** | `Gemfile.lock` | `bundle outdated` | `rubocop --auto-correct` |

## Phase 2: Execution

1. Create feature branch from main
2. Git tag current state: `git tag pre-upgrade-<dep>-<version>`
3. Upgrade the dependency + its required co-dependencies together
4. **Fix errors in order**: type/compile errors → test failures → runtime issues
5. Run full test suite
6. Check for size/performance regressions

### Large Model/Type Migrations (>50 affected files)

When upgrading changes many models/types (e.g., Pydantic v1→v2 with 120 models):

1. **Use official codemods first** — they handle the mechanical renames
2. **Batch by dependency order** — migrate leaf models first, then models that depend on them
3. **Verify after each batch** — don't migrate all 120 at once
4. **Track progress**: create a checklist of all affected files

### Common Codemod Tools

| Library | Tool | Command |
|---------|------|---------|
| Pydantic v1→v2 | `bump-pydantic` | `bump-pydantic .` |
| Django version upgrades | `django-upgrade` | `django-upgrade --target-version 4.2 **/*.py` |
| Python version upgrades | `pyupgrade` | `pyupgrade --py311-plus **/*.py` |
| Next.js upgrades | `@next/codemod` | `npx @next/codemod@latest <transform>` |
| React upgrades | `react-codemod` | `npx react-codemod <transform>` |
| ESLint flat config | `@eslint/migrate-config` | `npx @eslint/migrate-config .eslintrc.json` |

**Always review codemod output** — they handle ~80% of cases but miss edge cases.

## Phase 3: Validation

1. Smoke test in staging with real data
2. Monitor error rates for 24-48 hours post-deploy
3. Keep rollback branch ready for one week
4. Compare performance metrics against pre-upgrade baseline

## Automated Dependency Updates

| Tool | Best For | Key Config |
|------|----------|------------|
| **Renovate** | Serious projects needing grouping, auto-merge, scheduling | Group related packages, auto-merge minor/patch with passing CI |
| **Dependabot** | Simpler projects, GitHub-native | Limit open PRs to 5, conventional commit prefixes |
| **`uv lock --upgrade`** | Python projects using uv | Run in CI, compare lock file diff |

## Common Gotchas

### Lock File Hygiene
- Always commit lock files
- Use frozen installs in CI: `npm ci`, `pnpm install --frozen-lockfile`, `uv sync --frozen`
- After upgrade: deduplicate (`npm dedupe`, `pnpm dedupe`)

### Version Override Patterns

| Ecosystem | Mechanism | Example |
|-----------|-----------|---------|
| npm | `overrides` in package.json | `"overrides": {"nth-check": ">=2.0.1"}` |
| yarn | `resolutions` in package.json | `"resolutions": {"nth-check": ">=2.0.1"}` |
| pnpm | `pnpm.overrides` | Same as npm overrides |
| Python | `uv` constraints, pip constraints file | `uv add --constraint "numpy>=2.0"` |
| Cargo | `[patch]` in Cargo.toml | `[patch.crates-io] foo = { git = "..." }` |

Document ALL overrides with comments explaining why they exist.

### Peer Dependency Conflicts
- `--legacy-peer-deps` is a bandaid, not a fix — understand why it's needed
- If a peer dep has no compatible version: check if maintainer has a beta/rc, or fork temporarily

### Workspace/Monorepo Upgrades
- Upgrade shared packages first, then consumers
- Test downstream impact per package
- Pin workspace dependencies explicitly

## Upgrade Checklist

**Pre-upgrade:**
- [ ] Read changelogs for every major version in the path
- [ ] Check peer/transitive dependency compatibility
- [ ] Inventory affected files (grep for deprecated APIs)
- [ ] Create branch, tag current state
- [ ] Run full test suite (record baseline)
- [ ] Check for official codemods

**During upgrade:**
- [ ] One major version at a time
- [ ] Run codemods first, review output
- [ ] Fix type errors → test failures → runtime issues
- [ ] For large migrations: batch by dependency order, verify after each batch
- [ ] Check bundle size / binary size / performance

**Post-upgrade:**
- [ ] Full regression test
- [ ] Deploy to staging with real data
- [ ] Monitor errors 24-48h
- [ ] Update documentation if APIs changed
- [ ] Remove any temporary overrides/workarounds
