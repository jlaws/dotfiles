# MCP Server Implementation Examples

Complete server examples for Python (FastMCP) and TypeScript SDKs, including resources, prompts, and transport configuration.

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
# stdio (default) -- for Codex
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

### Resource Patterns

#### Static Resources

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

#### Dynamic Resources with URI Templates

```python
@mcp.resource("logs://{service}/{date}")
def get_service_logs(service: str, date: str) -> str:
    """Fetch logs for a service on a given date (YYYY-MM-DD)."""
    logs = log_store.query(service=service, date=date, limit=100)
    return "\n".join(f"[{l.timestamp}] {l.level}: {l.message}" for l in logs)
```

#### Resource Subscriptions (Notify on Change)

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
