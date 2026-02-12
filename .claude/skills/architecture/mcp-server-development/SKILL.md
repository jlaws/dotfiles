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

## Python SDK Setup (FastMCP)

### Installation

```bash
pip install mcp
```

### Minimal Server with Tools

```python
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("my-server")

@mcp.tool()
def search_docs(query: str, limit: int = 5) -> str:
    """Search documentation by keyword. Returns matching doc titles and snippets."""
    results = db.search(query, limit=limit)
    return "\n".join(f"- {r.title}: {r.snippet}" for r in results)

@mcp.tool()
def get_user(user_id: str) -> dict:
    """Retrieve user profile by ID. Returns name, email, and role."""
    user = db.get_user(user_id)
    if not user:
        return {"error": f"User {user_id} not found"}
    return {"name": user.name, "email": user.email, "role": user.role}

if __name__ == "__main__":
    mcp.run()  # Defaults to stdio transport
```

### Running with Different Transports

```python
# stdio (default) -- for Claude Code
mcp.run()

# SSE transport
mcp.run(transport="sse", host="0.0.0.0", port=8080)

# Streamable HTTP
mcp.run(transport="streamable-http", host="0.0.0.0", port=8080)
```

### Resources

```python
@mcp.resource("config://app")
def get_app_config() -> str:
    """Current application configuration."""
    return json.dumps(load_config(), indent=2)

@mcp.resource("users://{user_id}/profile")
def get_user_profile(user_id: str) -> str:
    """User profile data. URI: users://{user_id}/profile"""
    user = db.get_user(user_id)
    return json.dumps({"name": user.name, "email": user.email})
```

### Prompt Templates

```python
@mcp.prompt()
def review_code(code: str, language: str = "python") -> str:
    """Generate a code review prompt for the given code."""
    return f"""Review this {language} code for:
1. Bugs and correctness issues
2. Performance concerns
3. Security vulnerabilities
4. Style and readability

Code:
```{language}
{code}
```"""
```

## TypeScript SDK Setup

### Installation

```bash
npm install @modelcontextprotocol/sdk
```

### Minimal Server

```typescript
import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { z } from "zod";

const server = new McpServer({
  name: "my-server",
  version: "1.0.0",
});

server.tool(
  "search_docs",
  "Search documentation by keyword. Returns matching titles and snippets.",
  {
    query: z.string().describe("Search query"),
    limit: z.number().default(5).describe("Max results to return"),
  },
  async ({ query, limit }) => {
    const results = await db.search(query, limit);
    return {
      content: [
        {
          type: "text",
          text: results.map((r) => `- ${r.title}: ${r.snippet}`).join("\n"),
        },
      ],
    };
  }
);

server.resource(
  "config://app",
  "config://app",
  async (uri) => ({
    contents: [{ uri: uri.href, text: JSON.stringify(loadConfig(), null, 2), mimeType: "application/json" }],
  })
);

const transport = new StdioServerTransport();
await server.connect(transport);
```

## Tool Schema Design

### Good Tool Schemas

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

**Schema rules**:
- Use Python type hints; FastMCP converts them to JSON Schema automatically
- Docstring becomes the tool description -- make it count
- Use `str | None` for optional fields, never `Optional[str]` (deprecated style)
- Constrain values: use `Literal["low", "medium", "high"]` or enums
- Keep parameter count under 7; group related params into a nested object if needed

### Argument Descriptions via Annotated

```python
from typing import Annotated

@mcp.tool()
def query_database(
    sql: Annotated[str, "SQL SELECT query. No mutations allowed."],
    database: Annotated[str, "Database name: 'production' or 'staging'"] = "production",
    timeout_ms: Annotated[int, "Query timeout in milliseconds"] = 5000,
) -> str:
    """Execute a read-only SQL query against the specified database."""
    ...
```

## Resource Patterns

### Static Resources

```python
@mcp.resource("schema://database")
def get_db_schema() -> str:
    """Database schema for all tables."""
    tables = db.get_all_tables()
    return "\n\n".join(
        f"CREATE TABLE {t.name} (\n{format_columns(t.columns)}\n);"
        for t in tables
    )
```

### Dynamic Resources with URI Templates

```python
@mcp.resource("logs://{service}/{date}")
def get_service_logs(service: str, date: str) -> str:
    """Fetch logs for a service on a given date (YYYY-MM-DD)."""
    logs = log_store.query(service=service, date=date, limit=100)
    return "\n".join(f"[{l.timestamp}] {l.level}: {l.message}" for l in logs)
```

### Resource Subscriptions (Notify on Change)

```python
# Server notifies client when resource changes
@mcp.resource("metrics://dashboard")
def get_metrics() -> str:
    """Live system metrics."""
    return json.dumps(collect_metrics())

# In your update loop:
async def on_metrics_update():
    await mcp.notify_resource_changed("metrics://dashboard")
```

## Testing MCP Servers

### Using the MCP Inspector

```bash
# Test stdio server interactively
npx @modelcontextprotocol/inspector python my_server.py

# Test remote server
npx @modelcontextprotocol/inspector http://localhost:8080
```

### Programmatic Testing (Python)

```python
import pytest
from mcp.server.fastmcp import FastMCP
from mcp import ClientSession
from mcp.client.stdio import stdio_client, StdioServerParameters

@pytest.fixture
async def client():
    params = StdioServerParameters(command="python", args=["my_server.py"])
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            yield session

@pytest.mark.asyncio
async def test_search_tool(client):
    result = await client.call_tool("search_docs", {"query": "authentication"})
    assert result.content[0].type == "text"
    assert "auth" in result.content[0].text.lower()

@pytest.mark.asyncio
async def test_list_tools(client):
    tools = await client.list_tools()
    tool_names = [t.name for t in tools.tools]
    assert "search_docs" in tool_names
    assert "get_user" in tool_names

@pytest.mark.asyncio
async def test_resource(client):
    result = await client.read_resource("config://app")
    data = json.loads(result.contents[0].text)
    assert "database" in data
```

## Claude Code Integration

### claude_desktop_config.json

```json
{
  "mcpServers": {
    "my-server": {
      "command": "python",
      "args": ["/absolute/path/to/my_server.py"],
      "env": {
        "DATABASE_URL": "postgresql://localhost/mydb"
      }
    },
    "remote-server": {
      "url": "http://localhost:8080/mcp"
    }
  }
}
```

### .mcp.json (Project-Level Config)

```json
{
  "mcpServers": {
    "project-tools": {
      "command": "python",
      "args": ["./tools/mcp_server.py"],
      "env": {
        "PROJECT_ROOT": "."
      }
    }
  }
}
```

## Deployment Patterns

### Docker

```dockerfile
FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8080
CMD ["python", "server.py"]
```

```python
# server.py -- use streamable HTTP for containerized deployment
mcp = FastMCP("my-server")
# ... register tools, resources ...
mcp.run(transport="streamable-http", host="0.0.0.0", port=8080)
```

### systemd (Linux)

```ini
[Unit]
Description=MCP Server
After=network.target

[Service]
Type=simple
User=mcp
WorkingDirectory=/opt/mcp-server
ExecStart=/opt/mcp-server/.venv/bin/python server.py
Restart=on-failure
RestartSec=5
Environment=DATABASE_URL=postgresql://localhost/mydb

[Install]
WantedBy=multi-user.target
```

### Authentication for Remote Servers

```python
from starlette.middleware import Middleware
from starlette.middleware.authentication import AuthenticationMiddleware

@mcp.tool()
def protected_tool(query: str) -> str:
    """This tool requires authentication."""
    ...

# Add auth middleware for HTTP transports
# MCP spec recommends OAuth 2.0 for remote servers
# Validate Bearer tokens on each request
```

## Gotchas

### Tool Names Must Be Unique
MCP requires globally unique tool names within a server. If two tools do similar things, differentiate clearly: `search_docs_by_keyword` vs `search_docs_by_date`.

### stdio Servers Must Not Print to stdout
Any `print()` call corrupts the MCP protocol stream. Use `stderr` for logging:
```python
import sys
print("debug info", file=sys.stderr)  # Safe
print("debug info")  # BREAKS MCP PROTOCOL
```

### Large Tool Results
MCP has no formal size limit, but hosts may truncate large results. Keep tool output under 10K characters. Paginate or summarize if needed.

### Resource URIs Are Opaque to the Model
The model doesn't "browse" resources. The host (Claude Code) decides which resources to load based on relevance. Design resource URIs to be human-readable and descriptive.

### Error Handling
Return errors as content, not exceptions. Exceptions crash the tool call; structured error messages let the model retry or adjust:
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

### Testing Tip
Always test with the MCP Inspector before integrating with a host. It shows the exact JSON-RPC messages exchanged, making protocol issues visible immediately.
