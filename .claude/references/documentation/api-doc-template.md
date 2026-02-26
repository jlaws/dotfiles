## API Documentation Patterns

### Endpoint Documentation Template

```markdown
## Create a Widget

Creates a new widget in the specified workspace.

`POST /v1/workspaces/{workspace_id}/widgets`

### Path Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `workspace_id` | `string` | The workspace UUID |

### Request Body

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `name` | `string` | yes | Widget display name (1-128 chars) |
| `type` | `string` | yes | One of: `counter`, `gauge`, `chart` |
| `config` | `object` | no | Type-specific configuration |

### Example Request

\`\`\`bash
curl -X POST https://api.example.com/v1/workspaces/ws_123/widgets \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Daily Signups",
    "type": "counter",
    "config": { "query": "SELECT count(*) FROM signups WHERE date = today()" }
  }'
\`\`\`

### Response `201 Created`

\`\`\`json
{
  "id": "wgt_456",
  "name": "Daily Signups",
  "type": "counter",
  "created_at": "2025-01-15T10:30:00Z"
}
\`\`\`

### Error Responses

| Status | Code | Description |
|--------|------|-------------|
| `400` | `invalid_type` | Unknown widget type |
| `404` | `workspace_not_found` | Workspace does not exist |
| `409` | `name_conflict` | Widget name already exists in workspace |
| `422` | `invalid_config` | Config does not match type schema |
```

### API Doc Rules
- Always show curl first, then language-specific SDKs
- Include realistic (not `foo`/`bar`) example values
- Document every error code the endpoint can return
- Show both success and error response bodies
- Version the URL; mention deprecation timelines
