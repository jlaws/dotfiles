# CI Integration & Supply Chain Security

## CI Integration

### Pipeline Placement

```
PR opened -> lint -> test -> dependency audit -> SBOM -> build -> deploy
```

### GitHub Actions Example

```yaml
- name: Audit dependencies
  run: npm audit --audit-level=critical
  # Fails on critical only; high/medium are warnings

- name: Check licenses
  run: npx license-checker --onlyAllow "MIT;Apache-2.0;BSD-2-Clause;BSD-3-Clause;ISC"

- name: Generate SBOM
  run: syft . -o cyclonedx-json > sbom.json

- name: Scan SBOM
  run: grype sbom:./sbom.json --fail-on critical
```

### Severity Gating Strategy

| Severity | PR Check | Action |
|----------|----------|--------|
| Critical | Block merge | Must fix before merge |
| High | Warn (annotation) | Fix within sprint |
| Medium | Info only | Track in backlog |
| Low | Silent | Quarterly review |

## Defensive Measures

```bash
# npm: disable install scripts by default
npm config set ignore-scripts true
# Explicitly allow for known packages
npx --allow-scripts=node-gyp npm install

# Use lock files (always commit them)
npm ci          # Install from lock file exactly (not npm install)
yarn install --frozen-lockfile
pip install --require-hashes -r requirements.txt
```

## Supply Chain Attack Patterns

| Attack | Description | Mitigation |
|--------|-------------|------------|
| **Typosquatting** | `lodsah` instead of `lodash` | Verify package name carefully, use lock files |
| **Dependency confusion** | Public package name matches internal name | Scope private packages (`@company/pkg`), registry configuration |
| **Maintainer compromise** | Legitimate maintainer account hijacked | Pin exact versions, review changelogs before upgrade |
| **Malicious postinstall** | Package runs code on `npm install` | `--ignore-scripts`, review install scripts |
| **Protestware** | Maintainer inserts destructive/political code | Pin versions, review diffs, monitor advisories |
| **Star jacking** | Fake GitHub stars to build trust | Check actual download counts, contributor history |

### Defensive Measures

```bash
# npm: disable install scripts by default
npm config set ignore-scripts true
# Explicitly allow for known packages
npx --allow-scripts=node-gyp npm install

# Use lock files (always commit them)
npm ci          # Install from lock file exactly (not npm install)
yarn install --frozen-lockfile
pip install --require-hashes -r requirements.txt
```

## Container & Secrets Scanning

### Trivy Container Image Scanning

```yaml
# GitHub Actions: scan container images for vulnerabilities
- name: Build image
  run: docker build -t myapp:${{ github.sha }} .

- name: Trivy image scan
  uses: aquasecurity/trivy-action@0.28.0
  with:
    image-ref: myapp:${{ github.sha }}
    format: table
    exit-code: 1          # Fail on findings
    severity: CRITICAL,HIGH
    ignore-unfixed: true  # Skip vulns with no patch available
```

### Gitleaks Secrets Detection

```yaml
# Pre-commit hook (.pre-commit-config.yaml)
repos:
  - repo: https://github.com/gitleaks/gitleaks
    rev: v8.21.2
    hooks:
      - id: gitleaks

# GitHub Actions CI
- name: Gitleaks secrets scan
  uses: gitleaks/gitleaks-action@v2
  env:
    GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
```

### Combined SAST Pipeline Ordering

```
PR opened → lint → test → Semgrep (code) → Trivy (containers) → Gitleaks (secrets) → dependency audit → SBOM → build → deploy
```

Run static analysis in this order: code patterns first (cheapest, fastest), then container images (slower build required), then secrets (catches anything that slipped through), then dependency/supply-chain checks.

## Pinning vs Floating Versions

| Strategy | Syntax | Pros | Cons |
|----------|--------|------|------|
| Exact pin | `1.2.3` | Deterministic, safe | Must manually update |
| Tilde (patch) | `~1.2.3` | Auto-patch updates | Minor risk from patches |
| Caret (minor) | `^1.2.3` | Auto-minor updates | Breaking changes happen despite semver |
| Range | `>=1.2.3 <2.0.0` | Flexible | Unpredictable |

**Recommendation**:
- **Libraries**: Use caret (`^`) for flexibility; consumers resolve versions
- **Applications**: Use exact pins or lock files for determinism
- **CI/CD (GitHub Actions)**: Pin to SHA digests, not tags (tags can be moved)

```yaml
# Bad: tag can be force-pushed
- uses: actions/checkout@v4

# Good: SHA is immutable
- uses: actions/checkout@b4ffde65f46336ab88eb53be808477a3936bae11 # v4.1.1
```
