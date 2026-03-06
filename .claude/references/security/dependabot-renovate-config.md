# Dependabot & Renovate Configuration

## Dependabot Configuration

```yaml
# .github/dependabot.yml
version: 2
updates:
  - package-ecosystem: "npm"
    directory: "/"
    schedule:
      interval: "weekly"
      day: "monday"
    open-pull-requests-limit: 10
    groups:
      dev-dependencies:
        dependency-type: "development"
        update-types: ["minor", "patch"]
      production-minor-patch:
        dependency-type: "production"
        update-types: ["minor", "patch"]
    # Major production upgrades get individual PRs
    ignore:
      - dependency-name: "*"
        update-types: ["version-update:semver-major"]
    # Except security updates -- always allow
    security-updates:
      open-pull-requests-limit: 20

  - package-ecosystem: "pip"
    directory: "/"
    schedule:
      interval: "weekly"
    groups:
      all-minor-patch:
        update-types: ["minor", "patch"]

  - package-ecosystem: "github-actions"
    directory: "/"
    schedule:
      interval: "weekly"
```

## Renovate Configuration

```json5
// renovate.json
{
  "$schema": "https://docs.renovatebot.com/renovate-schema.json",
  "extends": ["config:recommended"],
  "schedule": ["before 7am on Monday"],
  "timezone": "America/New_York",
  "packageRules": [
    {
      "description": "Automerge non-major dev dependencies",
      "matchDepTypes": ["devDependencies"],
      "matchUpdateTypes": ["minor", "patch"],
      "automerge": true,
      "automergeType": "pr"
    },
    {
      "description": "Group non-major production patches",
      "matchDepTypes": ["dependencies"],
      "matchUpdateTypes": ["minor", "patch"],
      "groupName": "production dependencies (non-major)"
    },
    {
      "description": "Pin GitHub Actions digests",
      "matchManagers": ["github-actions"],
      "pinDigests": true
    }
  ],
  "vulnerabilityAlerts": {
    "enabled": true,
    "labels": ["security"]
  }
}
```
