---
name: llm-application-patterns
description: "Use when building LLM applications: prompt engineering, structured output extraction (JSON mode, function calling, constrained decoding, schema design), agents, RAG integration, memory management, or production deployment. Framework-agnostic patterns using raw SDK calls."
---

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
| Reasoning / Math | CoT with verification | Zero-shot (unreliable) |
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

### Structured Output

See the dedicated [Structured Output](#structured-output) section below for method selection, schema design, and gotchas.

## ReAct / Tool Use

```python
# Anthropic tool use
import anthropic

client = anthropic.Anthropic()
tools = [
    {
        "name": "search_database",
        "description": "Search internal knowledge base. Returns relevant documents.",
        "input_schema": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    },
    {
        "name": "calculate",
        "description": "Evaluate a math expression.",
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
            model="claude-sonnet-4-5-20250929", max_tokens=1024,
            tools=tools, messages=messages,
        )

        if response.stop_reason == "end_turn":
            return response.content[0].text

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

- Version prompts by hashing `template + model + temperature` (SHA256 prefix)
- Store as dataclass with `name`, `template`, `model`, `temperature`, `version`
- Evaluate by running test cases through the prompt, comparing predictions to expected values
- Track accuracy per version to detect regressions when prompts change

## Production Guardrails

### Cost Control
- Cache identical queries (hash prompt + model + temperature)
- Route simple tasks to cheaper/smaller models
- Summarize history before exceeding context window
- Monitor token usage by endpoint

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
- Assuming a prompt that works on GPT-4 transfers to smaller models
- Storing entire conversation history without windowing (context overflow + cost explosion)
- Generic tool descriptions (confuses agent tool selection)
- No fallback for LLM failures (always handle rate limits and timeouts)

## Structured Output

### Method Selection

| Method | Provider | Guarantees Schema? | Best For |
|--------|----------|-------------------|----------|
| **OpenAI Structured Outputs** | OpenAI | Yes (constrained decoding) | Production extraction with GPT-4o |
| **Anthropic tool_use** | Anthropic | Yes (schema-validated) | Extraction with Claude models |
| **Instructor** | Any (wrapper) | Yes (retry + validation) | Multi-provider, complex validation |
| **Outlines** | Local models | Yes (constrained decoding) | Open-source models, custom grammars |
| **JSON mode** | OpenAI/others | JSON only (no schema) | Simple cases, no strict schema |

**Decision rule**: Use provider-native structured outputs first. Use Instructor for cross-provider compatibility or complex Pydantic validation. Use Outlines for local/open-source models.

### Quick Start -- Anthropic

Force the model to call a "tool" matching your desired schema. No actual tool execution needed.

```python
import anthropic

client = anthropic.Anthropic()
response = client.messages.create(
    model="claude-sonnet-4-5-20250929",
    max_tokens=1024,
    tools=[{
        "name": "extract_info",
        "description": "Extract structured information from text",
        "input_schema": {
            "type": "object",
            "properties": {
                "company_name": {"type": "string"},
                "revenue_millions": {"type": "number", "description": "Revenue in millions USD"},
                "sentiment": {"type": "string", "enum": ["positive", "negative", "neutral"]},
            },
            "required": ["company_name", "sentiment"],
        },
    }],
    tool_choice={"type": "tool", "name": "extract_info"},
    messages=[{"role": "user", "content": f"Extract info from: {text}"}],
)
result = response.content[0].input  # Parsed dict
```

### Quick Start -- OpenAI

```python
from openai import OpenAI
from pydantic import BaseModel, Field

class CompanyInfo(BaseModel):
    company_name: str
    revenue_millions: float | None = None
    sentiment: str

client = OpenAI()
completion = client.beta.chat.completions.parse(
    model="gpt-4o",
    messages=[{"role": "user", "content": f"Extract info from: {text}"}],
    response_format=CompanyInfo,
)
result = completion.choices[0].message.parsed  # CompanyInfo instance
```

For Instructor, Outlines, nested schemas, and multi-entity extraction, see [references/provider-examples.md](references/provider-examples.md).

### Schema Design Tips

| Tip | Why |
|-----|-----|
| Use `enum` for categorical fields | Prevents hallucinated categories |
| Make uncertain fields `optional` | Model fills None instead of guessing |
| Add `description` to every field | Guides the model on what to extract |
| Keep schemas under 15 fields | Accuracy drops with complex schemas |
| Use nested objects for related fields | Groups logically, reduces confusion |

For anti-patterns catalog and Pydantic validation strategies, see [references/schema-anti-patterns.md](references/schema-anti-patterns.md).

### Structured Output Gotchas

- **OpenAI strict mode** requires `additionalProperties: false` and all fields in `required`. Use Pydantic defaults -- fields still appear in `required` but the model can output `null`.
- **Anthropic `tool_choice`** forces a tool call even on empty input. Validate for garbage extractions.
- **Temperature**: Use `temperature=0` for extraction. Higher temperature = creative but wrong values.
- **Nested arrays (3+ levels)**: Models struggle. Flatten or extract in multiple passes.
- **Pydantic V2 required**: Instructor and OpenAI SDK need V2. Key changes: `@field_validator` replaces `@validator`, `model_dump()` replaces `.dict()`.
- **Long documents**: Chunk first, extract per chunk, merge/deduplicate. Don't rely on truncation.

For retry strategies and provider fallback patterns, see [references/retry-and-fallback.md](references/retry-and-fallback.md).

## Cross-References

- **ai-ml:rag-and-vector-search** -- retrieval-augmented generation, chunking, embedding strategies
- **ai-ml:agentic-systems-design** -- tool use, multi-agent orchestration, planning loops
- **languages:pydantic-and-data-validation** -- Pydantic v2 models for extraction schemas
