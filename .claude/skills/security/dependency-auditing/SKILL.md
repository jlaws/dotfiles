---
name: dependency-auditing
description: "Supply chain security auditing with dependency scanning, CVE triage, SBOM generation, and license compliance. Use when auditing supply chain security. Covers dependency scanning tools, CVE triage workflows, SBOM generation, license compliance, and Dependabot/Renovate configuration. Do NOT use for application-level threat modeling or code scanning (use security-analysis)."
---

# Dependency Auditing

## Critical Rules

- **Always** commit lock files to Git -- without them, CI resolves different versions than local
- **Never** automerge production dependency updates without passing tests and human review
- **Always** pin CI/CD actions to SHA digests, not tags -- tags can be force-pushed by compromised maintainers
- **Always** fix critical CVEs (CVSS 9.0+) within 24 hours; high (7.0+) within 7 days
- **Never** run `npm install` in CI -- use `npm ci` (installs from lock file exactly)

## The Principle

```
EVERY DEPENDENCY IS AN ATTACK SURFACE.
Audit proactively, not after the breach.
```

Your app's security is only as strong as its weakest transitive dependency.

## Tool Selection by Ecosystem

| Ecosystem | Audit Tool | Lock File | Notes |
|-----------|-----------|-----------|-------|
| npm/yarn | `npm audit` / `yarn audit` | package-lock.json / yarn.lock | Built-in, JSON output for CI |
| pnpm | `pnpm audit` | pnpm-lock.yaml | Same flags as npm audit |
| Python (pip) | `pip-audit` | requirements.txt / poetry.lock | Queries OSV database |
| Python (pipenv) | `pipenv check` | Pipfile.lock | Uses Safety DB |
| Rust | `cargo audit` | Cargo.lock | RustSec advisory DB |
| Go | `govulncheck` | go.sum | Official Go tool, call-graph aware |
| Ruby | `bundler-audit` | Gemfile.lock | Ruby Advisory DB |
| Java/Kotlin | OWASP Dependency-Check / `gradle dependencyCheckAnalyze` | build.gradle.lock | NVD database |
| .NET | `dotnet list package --vulnerable` | packages.lock.json | Built-in since .NET 6 |

```bash
# Quick audit commands
npm audit --production          # Skip devDependencies
pip-audit -r requirements.txt   # Python
cargo audit                     # Rust
govulncheck ./...               # Go (analyzes actual call graph)
bundler-audit check --update    # Ruby (update advisory DB first)
```

## CVE Triage Workflow

### Step 1: Severity Assessment

| CVSS Score | Severity | SLA |
|------------|----------|-----|
| 9.0-10.0 | Critical | Fix within 24 hours |
| 7.0-8.9 | High | Fix within 7 days |
| 4.0-6.9 | Medium | Fix within 30 days |
| 0.1-3.9 | Low | Fix within 90 days or accept |

### Step 2: Exploitability Analysis

Not all CVEs are exploitable in your context. Ask:

1. **Is the vulnerable code path reachable?** (`govulncheck` does this automatically for Go)
2. **Is the vulnerable feature used?** (e.g., XML parsing CVE irrelevant if you only parse JSON)
3. **Is it exposed to untrusted input?** (internal-only service vs public API)
4. **Are there existing mitigations?** (WAF, input validation, sandboxing)

### Step 3: Decision Matrix

| Exploitable? | Upgrade Available? | Action |
|:---:|:---:|--------|
| Yes | Yes | Upgrade immediately |
| Yes | No | Apply workaround, monitor for patch, consider replacing dependency |
| No | Yes | Upgrade in next scheduled maintenance |
| No | No | Document risk acceptance with justification, set review date |

### Step 4: Response Options

| Response | When |
|----------|------|
| **Upgrade** | Patch version available, minimal breaking changes |
| **Patch** | Fork and patch if maintainer is unresponsive (last resort) |
| **Replace** | Dependency is unmaintained, better alternatives exist |
| **Mitigate** | Can't upgrade yet; add compensating controls (WAF rule, input validation) |
| **Accept** | Low risk, not exploitable in context; document and review quarterly |

## SBOM Generation

Software Bill of Materials -- inventory of all components in your software.

### Why SBOM Matters
- Required by US Executive Order 14028 for government contracts
- Enables rapid response when new CVEs drop (search your SBOM)
- License compliance auditing at scale

### Tools and Formats

| Tool | Format | Best For |
|------|--------|----------|
| `syft` (Anchore) | CycloneDX, SPDX | Container images, multi-ecosystem |
| `cdxgen` | CycloneDX | JS/Java/Python, CI-friendly |
| `trivy` | CycloneDX, SPDX | Container + filesystem scanning |
| `sbom-tool` (Microsoft) | SPDX | .NET ecosystems |

```bash
# Generate SBOM with syft
syft . -o cyclonedx-json > sbom.json
syft . -o spdx-json > sbom.spdx.json

# Scan container image
syft myimage:latest -o cyclonedx-json > sbom.json

# Scan SBOM for vulnerabilities
grype sbom:./sbom.json
```

## License Compliance

### License Categories

| Category | Licenses | Policy |
|----------|----------|--------|
| Permissive (safe) | MIT, Apache-2.0, BSD-2-Clause, BSD-3-Clause, ISC | Allow |
| Weak copyleft (review) | LGPL-2.1, LGPL-3.0, MPL-2.0 | Allow with conditions (dynamic linking OK, modifications must share) |
| Strong copyleft (restrict) | GPL-2.0, GPL-3.0, AGPL-3.0 | Block for proprietary software |
| No license | Unlicensed | Block (no license = all rights reserved) |

### Scanning Tools

```bash
# npm
npx license-checker --onlyAllow "MIT;Apache-2.0;BSD-2-Clause;BSD-3-Clause;ISC"

# Python
pip install pip-licenses
pip-licenses --fail-on "GPL-2.0;GPL-3.0;AGPL-3.0"

# Multi-ecosystem
licensee detect .
```

## Dependabot & Renovate Configuration

See references/dependabot-renovate-config.md for Dependabot and Renovate configuration templates.

## CI Integration & Supply Chain Security

See references/ci-and-supply-chain.md for CI pipeline integration, severity gating, supply chain attack patterns, and version pinning strategies.

## Gotchas

- `npm audit` reports vulnerabilities in devDependencies that never ship to production; use `--production` or `--omit=dev` to filter
- `govulncheck` is call-graph aware (only reports reachable vulns); other tools report all vulns in dependency tree regardless of reachability
- CVSS scores don't account for YOUR context; a CVSS 9.8 in a function you never call is lower risk than a CVSS 6.0 in code you expose to the internet
- Transitive dependencies are where most vulnerabilities hide; direct deps are usually well-maintained
- Lock files must be committed to git; without them, `npm install` on CI may resolve different versions than local
- `pip-audit` requires a lock file or `--strict` for reproducible results; bare `requirements.txt` with ranges is insufficient
- License scanning misses dual-licensed packages and license changes between versions; verify manually for critical dependencies
- Dependabot/Renovate automerge should only be enabled for dev dependencies with good test coverage

## Cross-References

- **security:security-analysis** -- SAST scanning, vulnerability detection, threat modeling
- **security:secrets-management** -- preventing secrets from leaking into dependencies
- **migration:dependency-upgrade** -- strategies for upgrading major dependency versions
