---
name: mcp-server-development
description: "Use when building MCP servers or clients, designing tool/resource/prompt schemas, choosing transport patterns, or integrating with Claude Code and other MCP hosts."
---

# MCP Server Development

## Transport Selection

| Transport | Use When | Pros | Cons |
|-----------|----------|------|------|
| **stdio** | Local tools, CLI integrations, Claude Code | Simple, no networking, secure | Single client only |
| **SSE (Server-Sent Events)** | Remote servers, multiple clients | HTTP-based, firewall-friendly | Unidirectional (server-to-client events) |
| **Streamable HTTP** | Production APIs, stateless deployments | Scalable, standard HTTP | Newer, less tooling support |

**Decision rule**: Use stdio for local dev tools and Claude Code integrations. Use streamable HTTP for production remote servers. SSE is legacy but still widely supported.

## Quick Start (Python FastMCP)

```bash
pip install mcp
```

```python
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("my-server")

@mcp.tool()
def search_docs(query: str, limit: int = 5) -> str:
    """Search documentation by keyword. Returns matching doc titles and snippets."""
    results = db.search(query, limit=limit)
    return "\n".join(f"- {r.title}: {r.snippet}" for r in results)

if __name__ == "__main__":
    mcp.run()  # Defaults to stdio transport
```

For complete Python and TypeScript server examples including resources, prompts, and transport variants, see [references/server-examples.md](references/server-examples.md).

## Tool Schema Design

**Schema rules**:
- Use Python type hints; FastMCP converts them to JSON Schema automatically
- Docstring becomes the tool description -- make it count
- Use `str | None` for optional fields, never `Optional[str]` (deprecated style)
- Constrain values: use `Literal["low", "medium", "high"]` or enums
- Keep parameter count under 7; group related params into a nested object if needed
- Use `Annotated[str, "description"]` for per-argument descriptions

```python
@mcp.tool()
def create_issue(
    title: str,
    body: str,
    labels: list[str] | None = None,
    assignee: str | None = None,
    priority: str = "medium",
) -> dict:
    """Create a new issue in the project tracker.

    Args:
        title: Short issue title (max 100 chars)
        body: Detailed description in markdown
        labels: Optional list of label names (e.g., ["bug", "urgent"])
        assignee: GitHub username to assign, or None for unassigned
        priority: One of: low, medium, high, critical

    Returns dict with issue_id and url.
    """
    ...
```

## Resource Patterns

| Pattern | URI Example | Use Case |
|---------|-------------|----------|
| Static | `schema://database` | Fixed data (DB schema, config) |
| Templated | `logs://{service}/{date}` | Parameterized lookups |
| Subscription | `metrics://dashboard` | Live data with change notifications |

For full resource implementation examples, see [references/server-examples.md](references/server-examples.md).

## Primitive Selection

| Primitive | Purpose | Model Controls? |
|-----------|---------|-----------------|
| **Tools** | Actions, mutations, computations | Yes -- model decides when to call |
| **Resources** | Data/context loading | No -- host/application decides |
| **Prompts** | Reusable prompt templates | No -- user selects explicitly |

## Integration

### .mcp.json (Project-Level)

```json
{
  "mcpServers": {
    "project-tools": {
      "command": "python",
      "args": ["./tools/mcp_server.py"],
      "env": { "PROJECT_ROOT": "." }
    }
  }
}
```

### Testing

```bash
# Test with MCP Inspector before integrating with any host
npx @modelcontextprotocol/inspector python my_server.py
```

For programmatic testing patterns and full Claude Desktop config, see [references/testing-and-integration.md](references/testing-and-integration.md).

For Docker, systemd, and auth deployment patterns, see [references/deployment.md](references/deployment.md).

## Gotchas

### stdio Servers Must Not Print to stdout
Any `print()` call corrupts the MCP protocol stream. Use `stderr` for logging:
```python
import sys
print("debug info", file=sys.stderr)  # Safe
print("debug info")  # BREAKS MCP PROTOCOL
```

### Tool Names Must Be Unique
MCP requires globally unique tool names within a server. Differentiate clearly: `search_docs_by_keyword` vs `search_docs_by_date`.

### Large Tool Results
Hosts may truncate large results. Keep tool output under 10K characters. Paginate or summarize if needed.

### Resource URIs Are Opaque to the Model
The model doesn't "browse" resources. The host decides which resources to load. Design URIs to be human-readable and descriptive.

### Error Handling
Return errors as content, not exceptions. Exceptions crash the tool call; structured error messages let the model retry:
```python
@mcp.tool()
def risky_tool(param: str) -> str:
    """Tool that might fail."""
    try:
        return do_work(param)
    except NotFoundError:
        return "Error: Resource not found. Check the ID and try again."
    except PermissionError:
        return "Error: Insufficient permissions for this operation."
```
