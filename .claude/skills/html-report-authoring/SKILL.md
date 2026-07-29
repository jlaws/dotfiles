---
name: html-report-authoring
description: "Use when creating a self-contained interactive HTML report."
allowed-tools: Read, Grep, Glob, Bash, Edit, Write
---

# HTML Report Authoring

## Overview

When information has shape — comparisons, diffs, call-graphs, timelines, hierarchies — or a reader needs to scan and explore it, a self-contained HTML document beats a wall of markdown. HTML carries higher information density (tables, CSS, SVG, layout), renders natively in any browser, and is shareable as one file. Markdown flattens spatial information; HTML keeps it.

## When to Use

- Explainers: how a system, algorithm, or concept works
- Information reports: findings, analysis, or research synthesis on a topic
- Spec and design docs: mockups, data flows, side-by-side approaches
- Cross-source synthesis: pulling many inputs into one navigable page
- Skip for: quick answers, inline code comments, or plain markdown docs (use `/j-docs`)

## The Single-File Rule

Non-negotiable: everything inline in one `.html` file.

- CSS in a `<style>` block, JS in a `<script>` block, diagrams as inline `<svg>`.
- No CDNs, no external fonts, no external scripts, no remote images.
- The file must open offline by double-click and send zero network requests.
- One file means one attachment, one link, one thing to share.

## Structure for Scanning

- Table of contents with in-page jump links (`href="#section-id"`).
- Consistent section headers; one idea per section.
- Color-code by category or severity so the eye finds things fast.
- Tables and callout boxes over long prose paragraphs.
- For reports: BLUF — lead with the finding, methodology after.

## Make the Shape Visible

Render structure spatially instead of describing it in prose:

- Side-by-side comparisons → CSS grid columns.
- Process or architecture → inline SVG boxes-and-arrows.
- Sequences and dependencies → timeline.
- Before/after or diffs → two-column with margin annotations.

## Interactivity, Minimal JS

- Prefer native elements: `<details>`/`<summary>` for collapse, `<dialog>` for modals.
- Tabbed panels for code variants; sliders (`<input type="range">`) for tunable parameters.
- Keep JS to a few lines. If it needs a framework, it is too much.

## Export Affordances

Where the reader may act on content, add copy buttons ("copy as Markdown", "copy as JSON", "copy diff"). This keeps the human-in-the-loop feedback cycle tight — they can take output straight back to the agent.

## Accessibility & Robustness

- Semantic HTML (`<nav>`, `<main>`, `<section>`, `<h1>`-`<h3>`).
- Responsive: readable on a phone and a wide monitor.
- Works in light and dark via `prefers-color-scheme` and CSS variables.
- Readable typography: system font stack, comfortable line length and spacing.

## File shape

Everything lives in one document, in this order:

```html
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>…</title>
  <style>/* all CSS — theme via custom properties + prefers-color-scheme */</style>
</head>
<body>
  <h1>…</h1>
  <nav class="toc"><!-- in-page jump links --></nav>
  <section id="…">…</section>
  <script>/* the few lines of JS, if any */</script>
</body>
</html>
```

Design the visual language for the content at hand rather than reusing a fixed palette or spacing scale.
The constraints above — one file, zero network requests, semantic elements, both color schemes,
responsive — are what must hold; how it looks is yours to decide. If the page includes charts, follow
`dataviz` for color and format.

## Quick Reference

| Content shape | HTML technique |
|---------------|----------------|
| Comparison / trade-offs | CSS grid columns, side-by-side cards |
| Process / architecture | inline SVG boxes-and-arrows |
| Sequence / dependencies | timeline (ordered list + CSS, or SVG) |
| Code / variants | tabbed `<pre>` panels |
| Findings / issues | color-coded cards by severity |
| Deep detail | `<details>` collapsed by default |
| Actionable output | copy-to-clipboard button |

## Common Mistakes

- **External CDNs or fonts**: breaks offline open and sharing. Inline everything.
- **Wall of prose**: defeats the purpose. Use tables, cards, and diagrams.
- **Truncated output**: a partial HTML file is broken. Emit the whole file (see `output-completeness`).
- **Over-engineered JS**: reach for native `<details>`/`<dialog>` before writing script.
- **Unlabeled inference in reports**: mark data vs inference and state gaps (see `analysis-output-patterns`).

## Cross-References

- **output-completeness** — never truncate the generated file.
- **analysis-output-patterns** — findings-first structure and data-vs-inference labeling for reports.
- **dataviz** — color and format rules when the page includes charts.
