## Changelog Patterns

### Keep a Changelog Format

```markdown
# Changelog

All notable changes to this project will be documented in this file.

Format based on [Keep a Changelog](https://keepachangelog.com/).

## [Unreleased]

### Added
- WebSocket support for real-time widget updates

## [2.1.0] - 2025-01-15

### Added
- `client.widgets.stream()` method for live data

### Changed
- Default timeout increased from 10s to 30s

### Deprecated
- `client.widgets.poll()` — use `stream()` instead, removal in v3.0

### Fixed
- Race condition when creating multiple widgets simultaneously (#234)

## [2.0.0] - 2025-01-01

### Breaking
- Removed `v1` endpoints; all requests must use `v2`
- `Config` type renamed to `WidgetConfig`

### Migration
- Update imports: `Config` → `WidgetConfig`
- Update base URLs: `/v1/` → `/v2/`
```

| Style | Use When | Example |
|-------|----------|---------|
| List (above) | Library/SDK, many small changes | Most open-source projects |
| Narrative | Product with fewer, bigger changes | "This release adds streaming..." |
| Commit log | Internal tools, low ceremony | Auto-generated from commits |

### Changelog Rules
- Categorize: Added, Changed, Deprecated, Removed, Fixed, Security
- Breaking changes get their own section + migration steps
- Link issue/PR numbers
- Date format: ISO 8601 (YYYY-MM-DD)
- Unreleased section at top for in-progress work
