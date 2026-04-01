# Analysis Output Patterns

Patterns for producing clear, trustworthy analysis output in data analysis, research, and reporting contexts.

## Structure

| Section | Purpose |
|---------|---------|
| Summary | 3 bullets max. Lead with the finding, not the methodology. |
| Supporting data | Tables, charts, evidence. Numbers with units and sources. |
| Caveats and limitations | What the data doesn't show. Confidence level and why. |

## Accuracy Rules

- Never state a number without a source or derivation
- If data is missing: say so. Do not estimate silently.
- If confidence is low: state it explicitly with a reason
- Do not round aggressively. Preserve meaningful precision.

## Data vs Inference

Distinguish clearly between what the data shows and what is inferred:

| Type | Example | Marker |
|------|---------|--------|
| Data | "Revenue grew 12% QoQ" | State as fact (with source) |
| Inference | "This suggests the pricing change drove adoption" | "Based on...", "This suggests...", "Likely because..." |
| Unknown | "We don't have data on churn by cohort" | State the gap explicitly |

Never present inferences as facts. Label every inference.

## Formatting

- Tables and bullets over prose paragraphs
- Numbers must include units (%, $, ms, requests/sec)
- Never ambiguous values ("a lot", "significantly" without quantification)
- Plain pipe-character tables, safe for copy-paste into spreadsheets
- No decorative Unicode in analytical output

## Hallucination Prevention for Analysis

- Never fabricate data points, statistics, or citations
- If a claim cannot be grounded in provided data: do not make it
- If asked to analyze data you haven't been given: request the data
- State sample sizes and time ranges for any aggregation

## Cross-References

- **skill:verification-before-completion** -- verify claims with evidence
- **reference:context-efficiency** -- output density and format choices
