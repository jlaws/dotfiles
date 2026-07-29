# MCP Client Configuration

How to consume MCP (Model Context Protocol) servers from an agent CLI (Claude Code, Codex, Gemini). This is the consumer side; for building a server, see `mcp-server-development.md`.

## Config scopes & precedence

Servers are declared in config; scope sets precedence (most specific wins):

| Scope | Location (Claude Code) | Applies to |
|-------|------------------------|-----------|
| Local / project | `.mcp.json` or project settings | This repo |
| User / global | user settings (`~`) | All projects |

Precedence: local/project > user. Keep secrets out of committed project config.

## Transport

| Transport | Use | Notes |
|-----------|-----|-------|
| stdio | Local process | Simplest; server runs as a child process |
| HTTP / SSE | Remote server | Needs URL + auth; a network dependency |
| Docker | Isolation | Run an untrusted or heavy server in a container |

## Credential hygiene

- Reference secrets by env var (`${GITHUB_TOKEN}`), never inline them in committed config.
- Prefer OAuth over long-lived API keys where the server supports it.
- Grant least privilege — read-only scopes/flags when the task only reads.
- Keep machine-local server config in an untracked/gitignored file.

## Verify & troubleshoot

- List configured servers and their status: `claude mcp list`.
- Confirm the tools a server exposes before relying on them.

| Problem | Fix |
|---------|-----|
| Server not found | Check scope/precedence; a user-scope server may be shadowed by project config |
| Auth failure | Verify the env var is set in the launching shell; check token scopes |
| Tools missing | Server started but registered no tools — check its logs/stderr |
| Slow every turn | Too many servers enabled; each injects tool definitions each turn |

## Context cost

Each enabled MCP server injects its tool definitions into context every turn. So:

- Enable servers **per-project**, not globally, unless you use one everywhere.
- Prefer lazy-loading / tool-search over always-on tool definitions when the client supports it.
- Prefer **official** servers over community ones for security and stability; vet community servers before enabling.

## Cross-References

- `mcp-server-development` — producer side: building an MCP server
- `security/secrets-management` — credential handling
- `config-security-audit` (skill) — scans MCP config for leaked secrets and over-broad scopes
