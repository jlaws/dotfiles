---
name: changelog-automation
description: Automate changelog generation from commits, PRs, and releases following Keep a Changelog format. Use when setting up release workflows, generating release notes, or standardizing commit conventions.
---

# Changelog Automation

## Conventional Commits

```
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]
```

| Type | Changelog Section | Semver Bump |
|------|-------------------|-------------|
| `feat` | Added | MINOR |
| `fix` | Fixed | PATCH |
| `perf` | Changed | PATCH |
| `refactor` | Changed | PATCH |
| `revert` | Removed | PATCH |
| `feat!` / `BREAKING CHANGE` | Breaking | MAJOR |
| `docs`, `style`, `test`, `chore`, `ci`, `build` | (excluded) | none |

## Method 1: Commitlint + Husky

```bash
npm install -D @commitlint/cli @commitlint/config-conventional husky

# commitlint.config.js
module.exports = {
  extends: ['@commitlint/config-conventional'],
  rules: {
    'subject-max-length': [2, 'always', 72],
  },
};

# Setup husky
npx husky init
echo "npx --no -- commitlint --edit \$1" > .husky/commit-msg
```

## Method 2: standard-version

```javascript
// .versionrc.js
module.exports = {
  types: [
    { type: 'feat', section: 'Features' },
    { type: 'fix', section: 'Bug Fixes' },
    { type: 'perf', section: 'Performance Improvements' },
    { type: 'revert', section: 'Reverts' },
    { type: 'docs', hidden: true }, { type: 'style', hidden: true },
    { type: 'chore', hidden: true }, { type: 'refactor', hidden: true },
    { type: 'test', hidden: true }, { type: 'build', hidden: true },
    { type: 'ci', hidden: true },
  ],
  releaseCommitMessageFormat: 'chore(release): {{currentTag}}',
};
```

```json
{ "scripts": {
    "release": "standard-version",
    "release:minor": "standard-version --release-as minor",
    "release:major": "standard-version --release-as major",
    "release:dry": "standard-version --dry-run"
} }
```

## Method 3: semantic-release (Full Automation)

```javascript
// release.config.js
module.exports = {
  branches: ['main', { name: 'beta', prerelease: true }],
  plugins: [
    '@semantic-release/commit-analyzer',
    '@semantic-release/release-notes-generator',
    ['@semantic-release/changelog', { changelogFile: 'CHANGELOG.md' }],
    ['@semantic-release/npm', { npmPublish: true }],
    ['@semantic-release/github', { assets: ['dist/**/*.js', 'dist/**/*.css'] }],
    ['@semantic-release/git', {
      assets: ['CHANGELOG.md', 'package.json'],
      message: 'chore(release): ${nextRelease.version} [skip ci]\n\n${nextRelease.notes}',
    }],
  ],
};
```

## Method 4: GitHub Actions

```yaml
name: Release
on:
  push: { branches: [main] }
  workflow_dispatch:
    inputs:
      release_type: { required: true, default: 'patch', type: choice, options: [patch, minor, major] }

permissions: { contents: write, pull-requests: write }

jobs:
  release:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with: { fetch-depth: 0, token: '${{ secrets.GITHUB_TOKEN }}' }
      - uses: actions/setup-node@v4
        with: { node-version: '20', cache: 'npm' }
      - run: npm ci
      - run: |
          git config user.name "github-actions[bot]"
          git config user.email "github-actions[bot]@users.noreply.github.com"
      - run: npx semantic-release
        env: { GITHUB_TOKEN: '${{ secrets.GITHUB_TOKEN }}', NPM_TOKEN: '${{ secrets.NPM_TOKEN }}' }
```

## Method 5: git-cliff (Rust-based)

```toml
# cliff.toml
[changelog]
body = """
{% if version %}\
    ## [{{ version | trim_start_matches(pat="v") }}] - {{ timestamp | date(format="%Y-%m-%d") }}
{% else %}\
    ## [Unreleased]
{% endif %}\
{% for group, commits in commits | group_by(attribute="group") %}
    ### {{ group | upper_first }}
    {% for commit in commits %}
        - {% if commit.scope %}**{{ commit.scope }}:** {% endif %}{{ commit.message | upper_first }}
    {% endfor %}
{% endfor %}
"""
trim = true

[git]
conventional_commits = true
filter_unconventional = true
commit_parsers = [
    { message = "^feat", group = "Features" },
    { message = "^fix", group = "Bug Fixes" },
    { message = "^perf", group = "Performance" },
    { message = "^refactor", group = "Refactoring" },
    { message = "^chore\\(release\\)", skip = true },
    { message = "^chore", group = "Miscellaneous" },
]
tag_pattern = "v[0-9]*"
sort_commits = "oldest"
```

```bash
git cliff -o CHANGELOG.md
git cliff v1.0.0..v2.0.0 -o RELEASE_NOTES.md
git cliff --unreleased --dry-run
```

## Method 6: Python (commitizen)

```toml
# pyproject.toml
[tool.commitizen]
name = "cz_conventional_commits"
version = "1.0.0"
version_files = ["pyproject.toml:version", "src/__init__.py:__version__"]
tag_format = "v$version"
update_changelog_on_bump = true
```

```bash
cz commit        # Interactive commit
cz bump --changelog  # Bump version + changelog
cz check --rev-range HEAD~5..HEAD  # Validate commits
```

## Commit Message Examples

```bash
feat(auth): add OAuth2 support for Google login
fix(checkout): resolve race condition in payment processing
Closes #123

feat(api)!: change user endpoint response format
BREAKING CHANGE: user endpoint returns `userId` instead of `id`.
```
