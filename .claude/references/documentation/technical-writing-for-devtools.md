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

See references/readme-template.md for README template, rules, and newcomer guidance.

## API Documentation Patterns

See references/api-doc-template.md for endpoint documentation template and rules.

## Quickstart Structure

See references/quickstart-template.md for quickstart template and rules.

## Changelog Patterns

See references/changelog-patterns.md for Keep a Changelog format, styles, and rules.

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
