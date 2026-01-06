---
name: token-efficiency-patterns
description: "Optimize context usage and reduce token consumption through compression, summarization, and strategic context management."
---

# Token Efficiency Patterns

## Overview

Maximize the effective use of context window by applying compression, selective inclusion, and smart summarization strategies. Reduce token usage by 30-50% while maintaining information quality.

## The Process

### 1. Audit Current Context
Before optimization, understand what's consuming tokens:
- Count approximate tokens in each context section
- Identify redundant or repetitive information
- Flag verbose content that can be compressed
- Note sections that could be summarized vs. kept verbatim

### 2. Apply Compression Strategies

**Code Context:**
- Include only relevant functions, not entire files
- Use signatures + docstrings instead of full implementations when behavior is obvious
- Reference file:line instead of pasting when code is already known
- Collapse repeated patterns with "...similar pattern for X, Y, Z"

**Conversation History:**
- Summarize completed discussions into key decisions
- Remove exploratory messages that led to dead ends
- Compress back-and-forth into "User requested X, we agreed on Y"
- Keep only the most recent iteration of revised content

**Documentation:**
- Extract only sections relevant to current task
- Paraphrase verbose documentation into bullet points
- Reference docs by name instead of including full text
- Use "as documented in X" for well-known patterns

### 3. Strategic Inclusion

**Include Verbatim:**
- Error messages and stack traces (exact wording matters)
- API contracts and type definitions
- Code that will be modified
- User requirements and acceptance criteria

**Summarize:**
- Background context and history
- Exploratory discussions
- Alternative approaches that were rejected
- General architectural context

**Reference Only:**
- Well-known libraries and frameworks
- Standard patterns (e.g., "standard REST CRUD")
- Previously discussed and agreed-upon decisions
- Code that won't be modified

### 4. Dynamic Context Management

**Progressive Disclosure:**
- Start with high-level summary
- Add detail only when needed
- Remove detail once decision is made

**Rotating Window:**
- Keep recent context fresh
- Archive older context as summaries
- Reload archived context on demand

## Compression Techniques

### Code Compression
```
BEFORE (50 tokens):
function processUser(user) {
  if (!user) throw new Error('User required');
  if (!user.email) throw new Error('Email required');
  return { id: user.id, email: user.email.toLowerCase() };
}

AFTER (20 tokens):
processUser(user): validates user/email, returns {id, email (lowercased)}
```

### Discussion Compression
```
BEFORE (100+ tokens):
User: "Should we use Redis or Memcached?"
Assistant: "Redis offers persistence and data structures..."
User: "Good point, let's use Redis"
Assistant: "I'll set up Redis with..."

AFTER (15 tokens):
Decision: Use Redis for caching (persistence + data structures)
```

### Context Reference
```
BEFORE: [paste entire 500-line file]
AFTER: See src/auth/middleware.ts - standard JWT validation pattern
```

## Token Budget Guidelines

| Context Type | Budget | Strategy |
|-------------|--------|----------|
| Current task | 40-50% | Full detail |
| Relevant code | 20-30% | Signatures + key logic |
| History/decisions | 10-15% | Summaries only |
| Documentation | 5-10% | References + excerpts |

## Key Principles

- **Compression over omission**: Summarize rather than cut entirely
- **Semantic density**: More meaning per token
- **Recoverability**: Can always expand compressed context on demand
- **Relevance filtering**: Only include what affects current decision
- **Progressive detail**: Start sparse, add density where needed
