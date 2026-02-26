# MCP Server Deployment

Production deployment patterns for MCP servers: Docker, systemd, and authentication.

## Docker

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

## systemd (Linux)

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

## Authentication for Remote Servers

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
