---
name: technical-writing-for-devtools
description: "Use when writing API docs, SDK quickstarts, developer tutorials, changelogs, README files, or establishing documentation style guides for developer tools."
---

# Technical Writing for Developer Tools

## Document Type Selection

Pick the right doc type first. Wrong format = wasted effort.

| Audience Need | Document Type | Length | Update Frequency |
|---|---|---|---|
| "What is this?" | README | 1-2 pages | Every release |
| "Get me running in 5 min" | Quickstart | 1 page | Every breaking change |
| "Teach me a workflow" | Tutorial | 3-10 pages | Quarterly |
| "What does X do exactly?" | API Reference | Per-endpoint | Every release |
| "What changed?" | Changelog | Per-version | Every release |
| "How do we write docs?" | Style Guide | 5-10 pages | Annually |
| "How does this work inside?" | Architecture Doc | 3-5 pages | Major versions |
| "Something broke" | Troubleshooting | Per-issue | Continuously |

## README Structure

Badge row first, one-liner second. Developers scan top-down and bail fast.

```markdown
[![CI](https://img.shields.io/github/actions/workflow/status/org/repo/ci.yml)](...)
[![npm](https://img.shields.io/npm/v/package)](...)
[![License](https://img.shields.io/badge/license-MIT-blue)](...)

# project-name

One sentence: what it does, who it's for, why it exists.

## Install

\`\`\`bash
npm install project-name
\`\`\`

## Quickstart

\`\`\`ts
import { Client } from 'project-name';

const client = new Client({ apiKey: process.env.API_KEY });
const result = await client.doThing({ input: 'hello' });
console.log(result);
\`\`\`

## API Reference

### `Client(options)`

| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `apiKey` | `string` | required | Your API key |
| `timeout` | `number` | `30000` | Request timeout in ms |

### `client.doThing(params)`

...

## Configuration

...

## Contributing

See [CONTRIBUTING.md](./CONTRIBUTING.md).

## License

MIT
```

### README Rules
- Install block within first screenful
- Working code example that can be copy-pasted directly
- No "Table of Contents" unless doc exceeds 5 screens
- Link out to detailed docs rather than inlining everything
- Keep badges to 3-5 max; CI, version, license are standard

### Newcomer-Focused Documentation
- Write READMEs assuming the reader has **minimal context** -- aim to get them productive quickly
- Include: high-level purpose, major concepts/abstractions, how it fits into the broader ecosystem
- Give directions (or better yet, a real working example) on how to get a concrete integration running
- Keep it concise -- long docs get skimmed, short docs get read

## API Documentation Patterns

### Endpoint Documentation Template

```markdown
## Create a Widget

Creates a new widget in the specified workspace.

`POST /v1/workspaces/{workspace_id}/widgets`

### Path Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `workspace_id` | `string` | The workspace UUID |

### Request Body

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `name` | `string` | yes | Widget display name (1-128 chars) |
| `type` | `string` | yes | One of: `counter`, `gauge`, `chart` |
| `config` | `object` | no | Type-specific configuration |

### Example Request

\`\`\`bash
curl -X POST https://api.example.com/v1/workspaces/ws_123/widgets \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Daily Signups",
    "type": "counter",
    "config": { "query": "SELECT count(*) FROM signups WHERE date = today()" }
  }'
\`\`\`

### Response `201 Created`

\`\`\`json
{
  "id": "wgt_456",
  "name": "Daily Signups",
  "type": "counter",
  "created_at": "2025-01-15T10:30:00Z"
}
\`\`\`

### Error Responses

| Status | Code | Description |
|--------|------|-------------|
| `400` | `invalid_type` | Unknown widget type |
| `404` | `workspace_not_found` | Workspace does not exist |
| `409` | `name_conflict` | Widget name already exists in workspace |
| `422` | `invalid_config` | Config does not match type schema |
```

### API Doc Rules
- Always show curl first, then language-specific SDKs
- Include realistic (not `foo`/`bar`) example values
- Document every error code the endpoint can return
- Show both success and error response bodies
- Version the URL; mention deprecation timelines

## Quickstart Structure

Goal: working code in under 5 minutes. No detours.

```markdown
# Quickstart

By the end of this guide you'll have a running [thing] that [does X].

## Prerequisites

- Node.js 18+
- An API key ([get one here](https://...))

## Step 1: Install

\`\`\`bash
npm install project-name
\`\`\`

## Step 2: Configure

\`\`\`bash
export API_KEY=your_key_here
\`\`\`

## Step 3: Write your first script

\`\`\`ts
// save as demo.ts
import { Client } from 'project-name';
// ... minimal working example
\`\`\`

## Step 4: Run it

\`\`\`bash
npx tsx demo.ts
\`\`\`

Expected output:
\`\`\`
Widget created: wgt_456
\`\`\`

## Next Steps

- [Tutorial: Build a dashboard](./tutorial-dashboard.md)
- [API Reference](./api-reference.md)
- [Configuration options](./configuration.md)
```

### Quickstart Rules
- 3-5 steps max; if more, it's a tutorial
- State the concrete outcome up front
- Prerequisites as a bullet list, not prose
- Every code block must be runnable as-is (no `...` elisions)
- "Next Steps" links to deeper docs, never dead ends

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

## Writing Style Guide

### Voice and Tense

| Do | Don't |
|----|-------|
| "Run the command" (imperative) | "You should run the command" |
| "The function returns a list" (present) | "The function will return a list" |
| "Pass the config object" (active) | "The config object should be passed" |
| "You can override the default" (second person) | "One can override the default" |
| "This method throws if..." (direct) | "It should be noted that this method..." |

### Document the Why, Not the What
- In code: comments explain the **why** (reasoning, constraints, tradeoffs) -- the code already shows the what
- In READMEs: explain purpose and concepts before diving into API details
- Don't comment obvious code (`i += 1  # increment i`); do comment surprising decisions (`# Using POST not GET because payload exceeds URL length limits`)

### Code-First Principle
- Show code before explaining it
- Prefer a 3-line example over a 3-paragraph explanation
- Annotate code with inline comments, not surrounding prose
- Every concept gets a runnable example

### Sentence Structure
- Lead with the action or outcome
- One idea per sentence
- Max 25 words per sentence for instructional content
- Use "Note:" sparingly; if everything is a note, nothing is

## Information Architecture

### Progressive Disclosure

Layer docs so readers go as deep as they need:

```
Level 1: README          → "What is this, how do I install it"
Level 2: Quickstart      → "Get something working fast"
Level 3: Tutorials       → "Learn workflows end-to-end"
Level 4: API Reference   → "Every parameter, every option"
Level 5: Architecture    → "How and why it works internally"
```

### Cross-Linking Rules
- Link forward ("see API Reference for all options") not backward
- Every page should be reachable from README within 2 clicks
- Use relative links for in-repo docs, absolute for external
- Avoid circular references between same-level docs

### Content Placement Decision

| Content | Belongs In | Not In |
|---------|-----------|--------|
| Install instructions | README | Quickstart |
| "Why this tool?" | README or landing page | Tutorial |
| Step-by-step workflow | Tutorial | API Reference |
| Parameter details | API Reference | Tutorial |
| Breaking changes | Changelog + Migration Guide | README |
| Troubleshooting | Dedicated page or FAQ | Inline in tutorials |

## Gotchas

- **Stale examples**: Code examples rot faster than prose. CI-test your docs or use snapshot testing on code blocks. A wrong example is worse than no example.
- **Untested code blocks**: Every code block should be extracted and run in CI. Tools like `mdx-js/mdx`, `doctest`, or custom scripts can automate this.
- **Assuming context**: Don't assume the reader just read the previous page. Each doc should state its prerequisites and link to them.
- **Over-documenting internals**: Public docs describe behavior, not implementation. Internal architecture docs are separate.
- **Version drift**: Pin version numbers in examples. `npm install foo` today installs a different version than tomorrow.
- **Screenshot dependency**: Screenshots break on every UI change. Prefer text descriptions with code; use screenshots only for visual UI docs.
- **Wall of text**: If a section exceeds 3 paragraphs without a code block, heading, or table, refactor it.
- **Jargon without definition**: First use of any domain term gets a parenthetical definition or a glossary link.
