# LLM Application Patterns

## Architecture Pattern Selection

| Pattern | Use When | Complexity |
|---------|----------|-----------|
| **Single prompt** | Classification, extraction, simple Q&A | Low |
| **Chain/pipeline** | Multi-step transformations, routing | Medium |
| **RAG** | Knowledge retrieval from docs | Medium |
| **Agent with tools** | External actions, multi-step reasoning | High |
| **Multi-agent** | Complex workflows, specialized sub-tasks | Very High |

**Decision rule**: Use the simplest pattern that solves the problem. A single well-structured prompt beats a complex chain 80% of the time.

## Prompting Strategies

### Strategy Selection

| Task Type | Strategy | Avoid |
|-----------|----------|-------|
| Classification | Few-shot with labels | CoT (overthinks simple tasks) |
| Reasoning / Math | CoT with verification; on reasoning models, thinking + `effort` instead | Zero-shot (unreliable) |
| Multi-step tasks | ReAct / tool-use | Single-shot (misses steps) |
| Extraction | Structured output + schema | Free-form (inconsistent) |
| Creative | System prompt + constraints | Over-constraining |

### Few-Shot Prompting

```python
SENTIMENT_PROMPT = """Classify the sentiment as positive, negative, or neutral.

Review: "The food was amazing and the service was quick."
Sentiment: positive

Review: "Waited 45 minutes and the order was wrong."
Sentiment: negative

Review: "It was okay, nothing special."
Sentiment: neutral

Review: "{review}"
Sentiment:"""
```

- 3-5 examples is the sweet spot (diminishing returns after)
- Cover all label classes in examples
- Vary example order across runs to check for position bias

### Chain-of-Thought (CoT)

```python
COT_PROMPT = """Solve step by step. Show reasoning, then give final answer as "Answer: <value>".

Question: {question}

Let me think step by step:"""
```

"Let's think step by step" works for large models (70B+). Smaller models often produce plausible-sounding but wrong reasoning. Verify CoT actually helps on your task before committing.

On reasoning models (Claude with adaptive thinking, and similar), skip the CoT phrase: the model reasons in thinking blocks, and `effort` sets the depth.

### Structured Output

See the dedicated [Structured Output](#structured-output) section below for method selection, schema design, and gotchas.

## ReAct / Tool Use

```python
# Anthropic tool use
import anthropic

client = anthropic.Anthropic()

# Model IDs name one model; they do not advance to the next generation.
# Keep the ID in config and bump it when a new model ships.
MODEL = "claude-sonnet-5-5"

tools = [
    {
        "name": "search_database",
        "description": "Search the internal knowledge base by keyword. Returns up to 5 matching documents with titles and snippets. Use for company-specific facts; it does not cover the public web.",
        "input_schema": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    },
    {
        "name": "calculate",
        "description": "Evaluate an arithmetic expression (+ - * / ** and parentheses) and return the number. Use instead of computing by hand; it does not solve equations or handle units.",
        "input_schema": {
            "type": "object",
            "properties": {"expression": {"type": "string"}},
            "required": ["expression"],
        },
    },
]

def agent_loop(question: str, max_steps: int = 5) -> str:
    messages = [{"role": "user", "content": question}]

    for _ in range(max_steps):
        response = client.messages.create(
            model=MODEL, max_tokens=16000,  # thinking shares this budget
            tools=tools, messages=messages,
        )

        if response.stop_reason == "end_turn":
            return "".join(b.text for b in response.content if b.type == "text")

        # Execute tool calls
        tool_results = []
        for block in response.content:
            if block.type == "tool_use":
                result = execute_tool(block.name, block.input)
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": str(result),
                })

        messages.append({"role": "assistant", "content": response.content})
        messages.append({"role": "user", "content": tool_results})

    return "Max steps reached"
```

## Memory / Context Management

| Conversation Length | Strategy | Implementation |
|-------------------|----------|---------------|
| < 10 messages | Full history | Pass all messages directly |
| 10-50 messages | Sliding window | Keep last K messages + system prompt |
| 50+ messages | Summarize + recent | Summarize old turns, keep recent 5-10 |
| Entity tracking | Structured state | Extract entities into dict, inject as context |
| Large corpus | Semantic retrieval | Embed messages, retrieve relevant history |

Pattern: check token count -> if over limit, keep system prompt + last K turns -> if still over, summarize old turns and prepend as context.

### Token Reduction Techniques

Savings are unmeasured estimates from general practice; measure on your own workload before relying on them.

| Technique | How | Estimated savings |
|-----------|-----|---------|
| Two-phase retrieval | Search/filter first, fetch only relevant items | 50-80% fewer input tokens |
| Filter parameters | Request only needed fields from APIs (`fields=id,name`) | 30-60% per response |
| Summary responses | Ask model to summarize rather than echo source material | 40-70% output tokens |
| Data cleaning (HTML→MD) | Strip tags, nav, ads before injecting into context | 2-3x reduction |
| Deterministic serialization | `json.dumps(data, sort_keys=True)` for cache-friendly output | Enables response caching |

### Stable Prefix / KV Cache

LLM providers cache the key-value computations for identical prompt prefixes. When your system prompt is identical across requests, subsequent requests skip recomputing those tokens. A change anywhere in the prefix invalidates the cache from that point onward.

**Rules (any provider):**
- Keep system instructions identical across sessions (no timestamps, counters, per-request IDs)
- Place dynamic content (user query, conversation history) at the END, not the beginning
- Reorder tool definitions consistently (alphabetical or by frequency)
- Version prompt templates deliberately; an edit near the top misses the cache for everything after it
- Send mid-session instruction updates as new messages, not edits to the system prompt
- Track cache-hit rate alongside token counts (see ai-ml:llmops-production-monitoring)

**Anthropic API specifics** (per [Anthropic's prompt caching docs](https://platform.claude.com/docs/en/build-with-claude/prompt-caching), its [cost blog](https://claude.com/blog/reducing-cost-and-improving-performance-with-claude-platform) of 2026-09-08, and its [cost-and-intelligence guide](https://platform.claude.com/docs/en/about-claude/models/optimizing-for-cost-and-intelligence) fetched 2026-10-02). Other providers differ: OpenAI caches prefixes automatically, Gemini uses explicit cached-content objects.
- Cache reads are byte-exact across the whole prefix, up to each `cache_control` breakpoint
- Effort and thinking settings render ahead of your content. Changing the top-level setting misses the cache from that point onward, and on some models from the tools and system prompt as well. A per-message effort change, where supported, keeps the prefix
- Mark rarely used tools `defer_loading`; they stay out of the cached prefix and load through tool search on demand
- Pre-warm or keep warm with `max_tokens: 0`. To pre-warm, send the first request's prefix with a cache breakpoint before real traffic (for example at startup). To keep a prefix warm, re-send the previous request byte-for-byte, with the same headers including any `anthropic-beta`, minus `stream`, within 4 minutes of the previous request's start and every 4 minutes after that. A cold pre-warm pays the cache write; a keep-alive on a warm entry bills only the read, though whether it refreshes an existing entry is unmeasured on some models. It is rejected with `thinking.type: "enabled"`, structured outputs, a forced `tool_choice`, the top-level `compaction` parameter, or inside a batch
- The default TTL is 5 minutes from the start of the request, and it refreshes free on each hit. Count the gaps between consecutive requests. Use the 1-hour TTL (write 2x base input vs 1.25x) once more than about 1 gap in 20 falls between 5 and 60 minutes and gaps over an hour are rare. Stay on 5 minutes when turns arrive seconds apart, or when most pauses over 5 minutes also run past an hour. On models whose cache read is far below 0.1x input, keep-alive requests can beat the 1-hour TTL; measure on the model you run
- Health check: agent loops on first-party traffic read a median 84% of input from cache. Below about 80%, look for a breaker
- Breakers beyond a system-prompt edit: any per-request value ahead of the prefix (on Anthropic's triage agent, a 25-token status line raised one run from $0.59 to $4.24), setting or changing an output format, adding, removing, or reordering a tool, changing a task budget, and many small context-editing passes. On Anthropic, that new message is a mid-conversation system message where the model supports it; and make unavoidable cache-breaking changes at natural breaks

## RAG Integration

### Chunking Strategy

| Document Type | Chunk Size | Overlap |
|---------------|------------|---------|
| Technical docs | 500-1000 tokens | 10-20% |
| Code | 300-500 tokens | 50 tokens |
| Chat logs | 200-300 tokens | 50 tokens |

### Retrieval Pipeline
1. Multi-query: generate 3-5 query variations for ambiguous questions
2. Hybrid search: dense (vector) + sparse (BM25) with RRF fusion
3. Rerank: cross-encoder on top 20-50 candidates → return top 3-5
4. Cite: include source markers `[1]`, `[2]` in generation prompt

## Prompt Versioning & Evaluation

- Version prompts by hashing `template + model + generation settings` (temperature, or effort and thinking on Claude; SHA256 prefix)
- Store as dataclass with `name`, `template`, `model`, `settings`, `version`
- Evaluate by running test cases through the prompt, comparing predictions to expected values
- Track accuracy per version to detect regressions when prompts change

## Production Guardrails

### Cost Control
- Cache identical queries (hash prompt + model + generation settings)
- Route by cost per completed task, not per-token price (see Cost per Completed Task)
- Summarize history before exceeding context window
- Monitor token usage by endpoint
- Upload tables through a files API and query them with code execution instead of pasting them: 25 of 25 aggregate questions correct versus 6 of 25 pasted, at about a twelfth of the cost (Anthropic, Sonnet 5, one 1,862-row CSV)
- Order and figures for Anthropic models: see Cost Levers and Effort and the sections after it

### Cost Levers and Effort

Two Anthropic sources inform this section: its [cost blog](https://claude.com/blog/reducing-cost-and-improving-performance-with-claude-platform) (2026-09-08) and its [cost-and-intelligence guide](https://platform.claude.com/docs/en/about-claude/models/optimizing-for-cost-and-intelligence) (fetched 2026-10-02). Free levers come first: caching, trimming context, a prompt audit against the current model, and the Batch API (50% off every token, cached ones included) for work no one waits on. Then come the tradeoffs: model choice, effort, budgets, and multi-model setups.

For Claude models, cut instructions that frontier models take literally: verification rituals ("double-check your work"), emphasis boosters ("CRITICAL: YOU MUST ALWAYS"), and fixed step scaffolds. Rewrite them as a goal plus its reason rather than deleting the constraint. Safety constraints (confirm before destructive actions, treat fetched content as data, keep secrets out of output) stay; they get a stated reason, not a louder voice. Task-level verification such as CoT with a check step (see Strategy Selection) is a separate technique.

Effort sets how much thinking, tool calling, and self-verification the model does, and the curve depends on the work. On research and knowledge-work benchmarks (Fable 5), the default bought nothing measurable over `medium`. On long-horizon coding (Opus 5.5, SWE-bench Pro), `xhigh` cost 2.5x `high` for 1.4 points more. Sweep two or three levels on a sample of your own traffic, each in its own session. Where outcomes are checkable, run low and re-run only the failures higher: on that coding set this matched an all-`high` pass rate at a little over half the cost. Use it for the saving, not the lift, and only with a failure signal you trust; each failure costs two runs of latency.

The blog's best cases, on four public benchmarks, were combined savings of about 24% to 73% against an Opus 5.5 baseline. The blog also reports Fable 5.1 at low effort matching Fable 5 at high effort for a third of the cost on one benchmark. The guide measured the same upgrade at matched effort: on DeepResearch Bench II, Fable 5.1 cost 41% more per task than Fable 5 at `high`. Measure on your workload before changing a default.

### Cost per Completed Task

Price lists are per token; you pay per completed task, and a stronger model often finishes with
fewer turns and less re-reading. On Anthropic's SWE-bench Pro subset, Fable 5.1 at `low` solved 11
points more than Sonnet 5 at its default for 35% less per solved task; on DeepResearch Bench II the
same pair ran the other way, with Fable 5.1 at `low` costing about four times as much per task. On
that coding subset, Opus 5.5 at its default matched Fable 5.1 at its default for about a fifth of
the cost per solved task. Price the hardest tenth of your tasks, not the median. A failed task bills
its tokens and then the retry, and the tail carries the spend even when nothing fails: on one
20-problem WideSearch run, two problems carried 43% of it.

### Budgets and Output Length

- A model-visible task budget saves money because the model plans around it. Anthropic's is a beta: advisory, with a 20,000-token floor, set once on the first request because a change invalidates the cache. A generous budget cut cost per task 44% for about 3 points on Fable 5.1
- `max_tokens` is an invisible safety cap: lowering it cut cost per attempt, not cost per solved task. For agentic work set it high (the cost-and-intelligence guide uses 64,000, or 128,000 where one cut-off is costly), stream the response, and treat `stop_reason: max_tokens` as a failure
- Ask for the answer you will read. On Anthropic's triage agent, a one-line final answer cost 14% less than a two-line one and a memo cost 2.8x the one-liner, with accuracy within noise across all three
- An agent loop cannot see a clock. Telling it that time matters, and sending the elapsed time before each later turn, cut run time 33% to 69% at up to 1.9 points lower score (Fable 5.1). Check the score on your own tasks first

### Multi-Model Strategies

Sweep effort first; most workloads end there. Then price the stronger model alone at low effort;
that is the number an advisor pairing has to beat.
- **Advisor**: a cheaper executor consults a frontier model on hard decisions. It pays only when the executor asks, and a low-effort executor can stop asking. An Opus 5.5 executor at `high` with a Fable 5.1 advisor gained 1.7 points over Opus 5.5 alone at `high` for about 2.1x the cost, roughly what more effort buys
- **Orchestrator**: a frontier lead plans, and cheaper workers take the bulk in parallel. It saved money in two measured cases: work larger than one context window, and a long cost tail on routine tasks. When the work was one dependent chain, or fit one context without a long cost tail, the lead's model alone at lower effort came out ahead every time

### Measuring Cost on Your Workload

1. Pull real tasks weighted like traffic, write an outcome check for each, and record cost per task from the response `usage`. Price five classes at their own rates: uncached input, 5-minute and 1-hour cache writes (1.25x and 2x input), cache reads, and output. See ai-ml:llmops-production-monitoring for a tracker
2. Baseline each model tier across effort levels, and plot score against spend
3. Add a multi-model strategy only if effort cannot close the gap, then re-run the suite
4. Shadow the winner on a traffic slice before cutover, and keep the suite running

These figures are Anthropic-internal and directional, priced at list rates when measured. Recheck them at each model release.

### Reliability
- Set timeout limits on all LLM calls
- Implement retry with exponential backoff for rate limits
- Fallback to simpler model on primary model failure
- Validate tool inputs before execution

### Observability
- Log: prompt version, model, tokens used, latency, response hash
- Track agent tool selection accuracy
- Monitor hallucination rate via groundedness checks
- Alert on latency p95/p99 regressions

## Gotchas

### Position Bias
Models favor options at certain positions (often first/last). For MCQ eval, rotate answer positions and average.

### Lost-in-the-Middle
Information in the middle of long contexts is retrieved less reliably. Put critical context at the beginning or end.

### Common Anti-Patterns
- Building complex chains when a single well-structured prompt suffices
- Temperature=0 for creative tasks (deterministic != best quality)
- Not testing adversarial/edge cases in prompt evaluation
- Assuming a prompt that works on a frontier model transfers to smaller models
- Storing entire conversation history without windowing (context overflow + cost explosion)
- Generic tool descriptions (confuses agent tool selection)
- No fallback for LLM failures (always handle rate limits and timeouts)
- Embedding per-request timestamps in system prompts (invalidates KV cache prefix)
- Returning full documents when summaries suffice (output token waste)
- Skipping data cleaning on fetched content (HTML often inflates tokens 2-3x, an unmeasured estimate)

## Structured Output

### Method Selection

| Method | Provider | Guarantees Schema? | Best For |
|--------|----------|-------------------|----------|
| **OpenAI Structured Outputs** | OpenAI | Yes (constrained decoding) | Production extraction with OpenAI models |
| **Anthropic Structured Outputs** | Anthropic | Yes (constrained decoding) | Production extraction with Claude models |
| **Instructor** | Any (wrapper) | Yes (retry + validation) | Multi-provider, complex validation |
| **Outlines** | Local models | Yes (constrained decoding) | Open-source models, custom grammars |
| **JSON mode** | OpenAI/others | JSON only (no schema) | Simple cases, no strict schema |

**Decision rule**: Use provider-native structured outputs first. Use Instructor for cross-provider compatibility or complex Pydantic validation. Use Outlines for local/open-source models.

### Quick Start -- Anthropic

Pass a Pydantic model; `messages.parse` constrains the output to its schema and validates it. That guarantees shape, not truth: required fields get filled even on empty or adversarial input, so validate values too.

```python
from typing import Literal

import anthropic
from pydantic import BaseModel, Field

class CompanyInfo(BaseModel):
    company_name: str
    revenue_millions: float | None = Field(None, description="Revenue in millions USD")
    sentiment: Literal["positive", "negative", "neutral"]

MODEL = "claude-sonnet-5-5"  # from config

client = anthropic.Anthropic()
response = client.messages.parse(
    model=MODEL,
    max_tokens=16000,
    messages=[{"role": "user", "content": f"Extract info from: {text}"}],
    output_format=CompanyInfo,
)
result = response.parsed_output  # CompanyInfo, or None on refusal or truncation
if result is None:
    raise ValueError(f"no structured output (stop_reason={response.stop_reason})")
```

### Quick Start -- OpenAI

```python
from openai import OpenAI
from pydantic import BaseModel

class CompanyInfo(BaseModel):
    company_name: str
    revenue_millions: float | None = None
    sentiment: str

client = OpenAI()
completion = client.beta.chat.completions.parse(
    model=OPENAI_MODEL,  # current OpenAI model ID, from config
    messages=[{"role": "user", "content": f"Extract info from: {text}"}],
    response_format=CompanyInfo,
)
result = completion.choices[0].message.parsed  # CompanyInfo instance
```

For Instructor, Outlines, nested schemas, and multi-entity extraction, see [ai-ml/llm-application-patterns/provider-examples.md](ai-ml/llm-application-patterns/provider-examples.md).

### Schema Design Tips

| Tip | Why |
|-----|-----|
| Use `enum` for categorical fields | Prevents hallucinated categories |
| Make uncertain fields `optional` | Model fills None instead of guessing |
| Add `description` to every field | Guides the model on what to extract |
| Keep schemas under 15 fields | Accuracy drops with complex schemas |
| Use nested objects for related fields | Groups logically, reduces confusion |

For anti-patterns catalog and Pydantic validation strategies, see [ai-ml/llm-application-patterns/schema-anti-patterns.md](ai-ml/llm-application-patterns/schema-anti-patterns.md).

### Structured Output Gotchas

- **OpenAI strict mode** requires `additionalProperties: false` and all fields in `required`. Use Pydantic defaults -- fields still appear in `required` but the model can output `null`.
- **Forced `tool_choice`** (`any` / `tool`) returns a 400 on Claude Opus 5.5, Sonnet 5.5, and Fable 5.1; use structured outputs for extraction, or `strict: true` under `tool_choice: auto` when a real tool is involved.
- **Temperature**: On providers and models that accept sampling parameters, use `temperature=0` for extraction. Current Claude models (Opus 4.7 and later, Sonnet 5 and later, Fable) reject non-default sampling values; rely on the schema for shape and validate values.
- **Nested arrays (3+ levels)**: Models struggle. Flatten or extract in multiple passes.
- **Pydantic V2 required**: Instructor and OpenAI SDK need V2. Key changes: `@field_validator` replaces `@validator`, `model_dump()` replaces `.dict()`.
- **Long documents**: Chunk first, extract per chunk, merge/deduplicate. Don't rely on truncation.

For retry strategies and provider fallback patterns, see [ai-ml/llm-application-patterns/retry-and-fallback.md](ai-ml/llm-application-patterns/retry-and-fallback.md).

## Cross-References

- **ai-ml:rag-and-vector-search** -- retrieval-augmented generation, chunking, embedding strategies
- **ai-ml:agentic-systems-design** -- tool use, multi-agent orchestration, planning loops
- **languages:pydantic-and-data-validation** -- Pydantic v2 models for extraction schemas
- **ai-ml:llmops-production-monitoring** -- cache-hit rate, token cost tracking, batch discounts
- **workflow:context-efficiency** -- token reduction, KV cache, U-shaped attention for Claude Code workflows
