# Release Versioning

## Semver Decision Table

| Change Type | Version Bump | Examples |
|-------------|-------------|----------|
| Breaking API change | MAJOR (X.0.0) | Remove endpoint, change response shape, rename public API |
| New feature, backward compatible | MINOR (x.Y.0) | Add endpoint, new optional parameter, new component |
| Bug fix, backward compatible | PATCH (x.y.Z) | Fix calculation, correct typo in output, security patch |
| Pre-release | x.y.z-alpha.N | Unstable, testing, preview |

## Version File Locations by Ecosystem

| Ecosystem | File(s) | Field |
|-----------|---------|-------|
| Node.js | package.json | `version` |
| Python | pyproject.toml | `[project].version` or `[tool.poetry].version` |
| Rust | Cargo.toml | `[package].version` |
| Go | No version file | Use git tags |
| Swift | .xcconfig or Package.swift | `MARKETING_VERSION` or tag |
| Ruby | lib/*/version.rb or *.gemspec | `VERSION` constant |

## Release Checklist

1. All tests pass on release branch
2. Bump version in appropriate file(s)
3. Update CHANGELOG (see `references/documentation/changelog-automation`)
4. Create git tag: `git tag -a vX.Y.Z -m "Release X.Y.Z"`
5. Push tag: `git push origin vX.Y.Z`
6. Create GitHub release (if applicable)
7. Verify CI/CD publishes artifacts

## Monorepo Considerations

- Independent versioning per package (recommended)
- Use conventional commits to auto-detect which packages changed
- Tools: changesets, lerna, turborepo release
