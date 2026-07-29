## README Structure

Badge row first, one-liner second. Developers scan top-down and bail fast.

```markdown
[![CI](https://img.shields.io/github/actions/workflow/status/org/repo/ci.yml)](...)
[![npm](https://img.shields.io/npm/v/package)](...)
[![License](https://img.shields.io/badge/license-MIT-blue)](...)

# project-name

One sentence: what it does, who it's for, why it exists.

## Install

\`\`\`bash
npm install project-name
\`\`\`

## Quickstart

\`\`\`ts
import { Client } from 'project-name';

const client = new Client({ apiKey: process.env.API_KEY });
const result = await client.doThing({ input: 'hello' });
console.log(result);
\`\`\`

## API Reference

### `Client(options)`

| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `apiKey` | `string` | required | Your API key |
| `timeout` | `number` | `30000` | Request timeout in ms |

### `client.doThing(params)`

...

## Configuration

...

## Contributing

See [CONTRIBUTING.md](./CONTRIBUTING.md).

## License

MIT
```

### README Rules
- Install block within first screenful
- Working code example that can be copy-pasted directly
- No "Table of Contents" unless doc exceeds 5 screens
- Link out to detailed docs rather than inlining everything
- Keep badges to 3-5 max; CI, version, license are standard
