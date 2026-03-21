## Quickstart Structure

Goal: working code in under 5 minutes. No detours.

```markdown
# Quickstart

By the end of this guide you'll have a running [thing] that [does X].

## Prerequisites

- Node.js 18+
- An API key ([get one here](https://...))

## Step 1: Install

\`\`\`bash
npm install project-name
\`\`\`

## Step 2: Configure

\`\`\`bash
export API_KEY=your_key_here
\`\`\`

## Step 3: Write your first script

\`\`\`ts
// save as demo.ts
import { Client } from 'project-name';
// ... minimal working example
\`\`\`

## Step 4: Run it

\`\`\`bash
npx tsx demo.ts
\`\`\`

Expected output:
\`\`\`
Widget created: wgt_456
\`\`\`

## Next Steps

- [Tutorial: Build a dashboard](./tutorial-dashboard.md)
- [API Reference](./api-reference.md)
- [Configuration options](./configuration.md)
```

### Quickstart Rules
- 3-5 steps max; if more, it's a tutorial
- State the concrete outcome up front
- Prerequisites as a bullet list, not prose
- Every code block must be runnable as-is (no `...` elisions)
- "Next Steps" links to deeper docs, never dead ends
