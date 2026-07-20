---
name: html-report-authoring
description: "Use when creating a self-contained interactive HTML report."
compatibility: claude-code
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

## Skeleton Template

```html
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>TITLE</title>
<style>
  :root {
    --bg: #ffffff; --fg: #1a1a1a; --muted: #666; --line: #e2e2e2;
    --accent: #2563eb; --card: #f7f7f8;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
  }
  @media (prefers-color-scheme: dark) {
    :root { --bg:#0f1115; --fg:#e6e6e6; --muted:#9aa0aa; --line:#2a2e37;
            --accent:#5b9dff; --card:#171a21; }
  }
  body { margin:0; background:var(--bg); color:var(--fg); line-height:1.6; }
  .wrap { max-width:1100px; margin:0 auto; padding:2rem; }
  nav.toc { position:sticky; top:0; background:var(--bg); border-bottom:1px solid var(--line);
            padding:.75rem 0; display:flex; gap:1rem; flex-wrap:wrap; font-size:.9rem; }
  nav.toc a { color:var(--accent); text-decoration:none; }
  section { border-top:1px solid var(--line); padding:1.5rem 0; }
  .grid { display:grid; grid-template-columns:1fr 1fr; gap:1rem; }
  .card { background:var(--card); border:1px solid var(--line); border-radius:8px; padding:1rem; }
  table { border-collapse:collapse; width:100%; }
  th, td { border:1px solid var(--line); padding:.5rem .75rem; text-align:left; }
  pre { background:var(--card); padding:1rem; border-radius:8px; overflow:auto; }
  button { cursor:pointer; border:1px solid var(--line); background:var(--card);
           color:var(--fg); border-radius:6px; padding:.35rem .7rem; }
</style>
</head>
<body>
<div class="wrap">
  <h1>TITLE</h1>
  <nav class="toc">
    <a href="#overview">Overview</a>
    <a href="#diagram">Diagram</a>
    <a href="#detail">Detail</a>
  </nav>

  <section id="overview">
    <h2>Overview</h2>
    <p>Lead with the finding.</p>
  </section>

  <section id="diagram">
    <h2>Diagram</h2>
    <svg viewBox="0 0 320 80" width="320" height="80" role="img" aria-label="A to B">
      <rect x="10" y="20" width="120" height="40" rx="6" fill="none" stroke="currentColor"/>
      <text x="70" y="45" text-anchor="middle" fill="currentColor">Input</text>
      <line x1="130" y1="40" x2="190" y2="40" stroke="currentColor" marker-end="url(#a)"/>
      <rect x="190" y="20" width="120" height="40" rx="6" fill="none" stroke="currentColor"/>
      <text x="250" y="45" text-anchor="middle" fill="currentColor">Output</text>
      <defs><marker id="a" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto">
        <path d="M0,0 L6,3 L0,6 Z" fill="currentColor"/></marker></defs>
    </svg>
  </section>

  <section id="detail">
    <h2>Detail</h2>
    <details>
      <summary>Expand for specifics</summary>
      <p>Collapsed by default to keep the page scannable.</p>
    </details>
    <p><button onclick="copyMd()">Copy as Markdown</button></p>
  </section>
</div>
<script>
  function copyMd() {
    const md = "# TITLE\n\nLead with the finding.";
    navigator.clipboard.writeText(md).then(() => alert("Copied"));
  }
</script>
</body>
</html>
```

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
