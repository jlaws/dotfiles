# Agentic Systems Design

## Agent Architecture Selection

| Architecture | Use When | Complexity | Latency |
|-------------|----------|-----------|---------|
| **Single-agent ReAct** | 1-5 tools, linear reasoning | Low | Low |
| **Plan-and-execute** | Multi-step tasks needing upfront planning | Medium | Medium |
| **Tree-of-Thought** | Tasks with branching solutions, math/logic | Medium | High |
| **LATS (Language Agent Tree Search)** | Complex search + evaluation loops | High | Very High |
| **Multi-agent supervisor** | Specialized sub-tasks, delegation | High | Medium |
| **Multi-agent debate** | Tasks needing verification, fact-checking | High | High |
| **Multi-agent chain** | Sequential pipeline, each agent transforms output | Medium | Medium-High |

**Decision rule**: Start with single-agent ReAct. Escalate to plan-and-execute if the agent frequently fails mid-task. Use multi-agent only when a single model cannot hold all required expertise in context.

## Planning Patterns

### ReAct (Reasoning + Acting)

The default pattern. Model alternates between reasoning (think) and acting (tool call). Loop until `stop_reason == "end_turn"`, appending the full `response.content` each turn so `tool_use` blocks are preserved.

### Plan-and-Execute

Separate planning from execution. Model generates a plan upfront, then executes steps sequentially, then synthesizes a final answer from the accumulated step results.

### Tree-of-Thought

Generate multiple reasoning paths, evaluate each, expand the most promising. BFS over reasoning states, scoring each candidate and keeping the top `breadth` per level.

## Tool Design Principles

### Schema Design

```python
# Good: specific description, constrained types, clear required fields
{
    "name": "search_orders",
    "description": "Search customer orders by order ID, customer email, or date range. Returns up to 10 matching orders with status and total.",
    "input_schema": {
        "type": "object",
        "properties": {
            "order_id": {"type": "string", "description": "Exact order ID (e.g., ORD-12345)"},
            "customer_email": {"type": "string", "format": "email"},
            "date_from": {"type": "string", "description": "ISO 8601 date (YYYY-MM-DD)"},
            "date_to": {"type": "string", "description": "ISO 8601 date (YYYY-MM-DD)"},
            "status": {"type": "string", "enum": ["pending", "shipped", "delivered", "cancelled"]},
        },
        "required": [],  # All optional -- at least one should be provided
    },
}
```

**Tool description rules**:
- Start with a verb: "Search", "Create", "Calculate", "Retrieve"
- Mention return format: "Returns a JSON list of...", "Returns a single..."
- Include example inputs in description when format is ambiguous
- Keep under 200 words; models parse long descriptions less reliably

### Error Handling in Tool Results

```python
def execute_tool(name: str, inputs: dict) -> str:
    try:
        result = TOOL_REGISTRY[name](**inputs)
        return json.dumps({"status": "success", "data": result})
    except KeyError:
        return json.dumps({"status": "error", "message": f"Unknown tool: {name}"})
    except ValidationError as e:
        return json.dumps({"status": "error", "message": f"Invalid input: {e}"})
    except Exception as e:
        return json.dumps({"status": "error", "message": f"Tool execution failed: {e}"})
```

Always return structured errors. Models recover better from `{"status": "error", "message": "..."}` than from raw exceptions or empty strings.

## Multi-Agent Patterns

### Supervisor Pattern

One orchestrator agent delegates to specialist agents. Expose delegation as a single `delegate` tool whose `specialist` property is an `enum` of the available specialist names -- this constrains routing to real specialists instead of letting the model invent one.

### Debate Pattern

Two agents argue for/against, a judge decides.

## Agent Evaluation

| Metric | What It Measures | How to Compute |
|--------|-----------------|----------------|
| **Task completion** | Did the agent solve the problem? | Human eval or automated check against gold answer |
| **Tool accuracy** | Did it call the right tools with right args? | Compare tool call trace to expected trace |
| **Step efficiency** | How many steps to solve? | Count tool calls; compare to optimal path |
| **Cost** | Total tokens consumed | Sum input + output tokens across all turns |
| **Hallucination rate** | Did it fabricate tool results or facts? | Check claims against tool outputs |

## Guardrails

| Guardrail | Default | Why |
|-----------|---------|-----|
| **Max iterations** | 10-15 | Prevents infinite loops |
| **Timeout** | 60-120s total | Caps wall-clock time |
| **Token budget** | 50K-100K per task | Caps cost per execution |
| **Human-in-the-loop** | On destructive actions | Prevents irreversible damage |
| **Tool allowlist** | Explicit per agent | Limits blast radius |
| **Output validation** | Schema check on final output | Ensures usable result |

## Gotchas

### Tool Description Quality
Vague tool descriptions cause wrong tool selection. "Gets data" is bad. "Retrieves customer order history by email address, returning last 30 days of orders with status and totals" is good.

### Infinite Loops
Agents can loop calling the same tool with the same args. Track call history and inject "You already called {tool} with these args. Try a different approach." after 2 duplicate calls.

### Context Window Overflow
Long agent runs accumulate tokens fast. Summarize older tool results once context exceeds 50% of window. Keep the last 2-3 tool results verbatim.

### Overly Eager Tool Use
Models sometimes call tools when they already have the answer in context. Add "Only use a tool if you cannot answer from information you already have" to the system prompt.

### Multi-Agent Communication Overhead
Each handoff between agents adds latency and token cost. Minimize cross-agent calls. If two agents always work together, merge them into one with a richer tool set.

### Evaluation Pitfalls
- Don't evaluate agents only on final answer; inspect the full tool call trace
- Agent behavior is non-deterministic; run evals 3-5 times and report variance
- Test adversarial inputs: ambiguous questions, impossible tasks, tasks requiring tools the agent doesn't have
