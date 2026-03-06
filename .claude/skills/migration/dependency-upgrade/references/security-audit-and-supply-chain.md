# Dependency Security Audit and Supply Chain

Multi-ecosystem dependency scanning, vulnerability analysis, license compliance, and supply chain security patterns.

## Multi-Ecosystem Scanning

### Ecosystem-Specific Audit Commands

```bash
# NPM
npm audit --json > npm-audit.json
npm audit fix --force

# Python
pip install safety pip-audit
safety check --json > safety-report.json
pip-audit --format=json > pip-audit.json

# Go
go install golang.org/x/vuln/cmd/govulncheck@latest
govulncheck -json ./... > govulncheck.json

# Rust
cargo install cargo-audit
cargo audit --json > cargo-audit.json
```

### Multi-Ecosystem Discovery

```python
class DependencyScanner:
    def __init__(self, project_path: str):
        self.project_path = Path(project_path)
        self.ecosystem_files = {
            'npm': ['package.json', 'package-lock.json', 'yarn.lock'],
            'python': ['requirements.txt', 'Pipfile', 'pyproject.toml'],
            'go': ['go.mod', 'go.sum'],
            'rust': ['Cargo.toml', 'Cargo.lock'],
            'ruby': ['Gemfile', 'Gemfile.lock'],
            'java': ['pom.xml', 'build.gradle'],
            'dotnet': ['*.csproj', 'packages.config']
        }

    def detect_ecosystems(self):
        detected = []
        for ecosystem, patterns in self.ecosystem_files.items():
            if any(list(self.project_path.glob(f"**/{p}")) for p in patterns):
                detected.append(ecosystem)
        return detected
```

## Vulnerability Severity Analysis

```python
def analyze_vulnerability_severity(vulnerabilities):
    severity_scores = {
        'critical': 9.0, 'high': 7.0, 'moderate': 4.0, 'low': 1.0
    }

    analysis = {
        'total': len(vulnerabilities),
        'by_severity': {'critical': [], 'high': [], 'moderate': [], 'low': []},
        'risk_score': 0,
        'immediate_action_required': []
    }

    for vuln in vulnerabilities:
        severity = vuln['severity'].lower()
        analysis['by_severity'][severity].append(vuln)

        base_score = severity_scores.get(severity, 0)
        if vuln.get('exploit_available', False):
            base_score *= 1.5
        if 'remote_code_execution' in vuln.get('description', '').lower():
            base_score *= 2.0

        vuln['risk_score'] = base_score
        analysis['risk_score'] += base_score

        if severity in ['critical', 'high'] or base_score > 8.0:
            analysis['immediate_action_required'].append({
                'package': vuln['package'],
                'severity': severity,
                'action': f"Update to {vuln['patched_versions']}"
            })

    return analysis
```

## License Compliance

```python
class LicenseAnalyzer:
    def __init__(self):
        self.license_compatibility = {
            'MIT': ['MIT', 'BSD', 'Apache-2.0', 'ISC'],
            'Apache-2.0': ['Apache-2.0', 'MIT', 'BSD'],
            'GPL-3.0': ['GPL-3.0', 'GPL-2.0'],
            'BSD-3-Clause': ['BSD-3-Clause', 'MIT', 'Apache-2.0'],
        }
        self.license_restrictions = {
            'GPL-3.0': 'Copyleft - requires source code disclosure',
            'AGPL-3.0': 'Strong copyleft - network use requires source disclosure',
            'unknown': 'License unclear - legal review required'
        }

    def analyze_licenses(self, dependencies, project_license='MIT'):
        issues = []
        for package_name, package_info in dependencies.items():
            license_type = package_info.get('license', 'unknown')
            if not self._is_compatible(project_license, license_type):
                issues.append({
                    'package': package_name,
                    'license': license_type,
                    'issue': f'Incompatible with project license {project_license}',
                    'severity': 'high'
                })
        return {
            'issues': issues,
            'compliance_status': 'FAIL' if issues else 'PASS'
        }
```

## Supply Chain Security Checks

- **Typosquatting detection**: Compare package names against popular packages
- **Maintainer change monitoring**: Alert on recent ownership changes
- **Version anomaly detection**: Flag unexpected major version jumps

## Batch Update Strategy

```python
def plan_batch_updates(dependencies):
    groups = {'patch': [], 'minor': [], 'major': [], 'security': []}

    for dep, info in dependencies.items():
        if info.get('has_security_vulnerability'):
            groups['security'].append(dep)
        else:
            groups[info['update_type']].append(dep)

    batches = []

    # Security updates: immediate, full testing
    if groups['security']:
        batches.append({
            'priority': 'CRITICAL', 'name': 'Security Updates',
            'packages': groups['security'], 'strategy': 'immediate', 'testing': 'full'
        })

    # Patch updates: safe, grouped, smoke testing
    if groups['patch']:
        batches.append({
            'priority': 'HIGH', 'name': 'Patch Updates',
            'packages': groups['patch'], 'strategy': 'grouped', 'testing': 'smoke'
        })

    # Minor updates: careful, incremental, regression testing
    if groups['minor']:
        batches.append({
            'priority': 'MEDIUM', 'name': 'Minor Updates',
            'packages': groups['minor'], 'strategy': 'incremental', 'testing': 'regression'
        })

    # Major updates: planned, individual, comprehensive testing
    if groups['major']:
        batches.append({
            'priority': 'LOW', 'name': 'Major Updates',
            'packages': groups['major'], 'strategy': 'individual', 'testing': 'comprehensive'
        })

    return batches
```

## CI/CD Integration

```yaml
name: Dependency Audit
on:
  schedule:
    - cron: '0 0 * * *'  # Daily
  push:
    paths:
      - 'package*.json'
      - 'requirements.txt'
      - 'go.mod'
  workflow_dispatch:

jobs:
  security-audit:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
    - name: Run NPM Audit
      if: hashFiles('package.json')
      run: |
        npm audit --json > npm-audit.json
        npm audit --audit-level=high
    - name: Run Python Safety Check
      if: hashFiles('requirements.txt')
      run: |
        pip install safety pip-audit
        safety check --json > safety-report.json
    - name: Check Licenses
      run: npx license-checker --json > licenses.json
    - name: Create Issue for Critical Vulnerabilities
      if: failure()
      uses: actions/github-script@v6
      with:
        script: |
          github.rest.issues.create({
            owner: context.repo.owner,
            repo: context.repo.repo,
            title: 'Critical vulnerabilities found in dependencies',
            body: 'Dependency audit found critical vulnerabilities.',
            labels: ['security', 'dependencies', 'critical']
          });
```

## Breaking Change Detection

- Parse changelogs for patterns: `BREAKING CHANGE:`, `removed`, `deprecated`, `no longer`, `renamed`, `replaced by`
- Identify safe intermediate versions (last patch of each minor)
- Create incremental upgrade paths through safe versions

## Migration Guide Template

```markdown
# Migration Guide: {package} {current} -> {target}

## Pre-Migration Checklist
- [ ] Current test suite passing
- [ ] Backup created / Git commit point marked
- [ ] Dependencies compatibility checked

## Step 1: Update Dependencies
git checkout -b upgrade/{package}-{target}
npm install {package}@{target}

## Step 2: Address Breaking Changes
{generated from changelog analysis}

## Step 3: Test & Verify
npm run lint && npm test && npm run type-check

## Rollback Plan
git checkout package.json package-lock.json && npm install
```
