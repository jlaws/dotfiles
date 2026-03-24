---
name: llm-output-completeness
description: "Empirical research on why LLMs truncate outputs and proven remediation patterns. Load when diagnosing truncation issues or designing prompts for large generation tasks."
type: reference
---

# LLM Output Completeness Research

## Root Causes of Truncation

| Cause | Mechanism |
|-------|-----------|
| RLHF/compute pressure | Token cost (~$0.0001/token) creates implicit brevity pressure during training |
| Training data bias | Stack Overflow/GitHub overrepresents partial code (answers, snippets, not full files) |
| Output token asymmetry | Models accept ~2M input tokens but generate only ~8K output tokens by default |
| Cognitive shortcuts | Models deliberately skip "simple" sections — truncation is intentional, not a decoding error |
| Alignment layer | Primary constraint: RLHF reward models penalize verbose outputs even when completeness is needed |

## Empirical Findings

- **Truncation is deliberate**: Not a probabilistic decoding failure — the model makes a purposeful choice to stop
- **+45% quality improvement** from psychological/financial framing in prompts (e.g., "your career depends on this")
- **Step-by-step reasoning** improves logic accuracy 34% → 80% (chain-of-thought effect)
- **Seasonal patterns**: Outputs measurably shorter in December (attributed to training data distribution)
- **Alignment as bottleneck**: Adding explicit completeness requirements in system prompts overrides RLHF brevity bias

## Remediation Patterns

### Architectural

| Pattern | How |
|---------|-----|
| Lazy-loaded skills | ~100 token YAML stubs + on-demand markdown expansion (prevents context bloat) |
| Chunked execution | Outline → components → assembly; never one giant generation request |
| Explicit scope binding | State exact deliverable count upfront: "Generate all 7 sections, do not stop early" |

### Prompt Engineering

| Pattern | Example |
|---------|---------|
| Psychological framing | "A partial output is a broken output. Complete all sections." |
| XML-structured output | `<section name="X">...</section>` tags make sections countable and verifiable |
| Explicit syntax binding | Require tool execution + evidence blocks before task is considered done |
| Verification loops | After generation: "List every section you were asked to produce. Have you produced all of them?" |
| Prohibition list | Explicitly ban `// ...`, `// TODO`, skeleton functions, "as mentioned above" |

### Parameter Tuning

| Setting | Guidance |
|---------|----------|
| Temperature | ≤ 0.5 for code generation; higher temps increase creative shortcuts |
| Gemini `thinking_level` | `medium` or `high` for complex multi-part tasks |
| Max output tokens | Always set to model maximum for large generation tasks |

## Key Insight

The alignment layer — not architecture — is the primary constraint. Models *can* generate complete outputs; they need explicit permission and structure to do so. The most effective intervention is changing the reward signal in-context: frame completeness as the correct behavior, incompleteness as a failure.

## Cross-References

- **workflow:completeness-principle** — project-level completeness standards
- **workflow:output-completeness** — skill for enforcing complete output in active sessions
