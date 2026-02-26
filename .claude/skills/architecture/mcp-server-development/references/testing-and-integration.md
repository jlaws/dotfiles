# MCP Testing and Host Integration

Testing strategies for MCP servers and configuration for Claude Code / Claude Desktop.

## Testing MCP Servers

### Using the MCP Inspector

```bash
# Test stdio server interactively
npx @modelcontextprotocol/inspector python my_server.py

# Test remote server
npx @modelcontextprotocol/inspector http://localhost:8080
```

Always test with the MCP Inspector before integrating with a host. It shows the exact JSON-RPC messages exchanged, making protocol issues visible immediately.

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
