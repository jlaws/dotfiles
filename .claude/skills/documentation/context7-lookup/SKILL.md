---
name: context7-lookup
description: Use when implementing features with external libraries, debugging library errors, or needing API reference for any framework/library
---

# Context7 Documentation Lookup

## Overview
Context7 MCP provides up-to-date library documentation. Use proactively without asking user.

**Reference:** https://github.com/upstash/context7

## When to Use
- Implementing with external libraries
- Library-related errors/debugging
- Unsure about API, parameters, patterns
- Framework-specific code (React, Django, FastAPI, SwiftUI, etc.)

## Tools

### resolve-library-id
Convert library name to Context7 ID.

Parameters:
- `query`: Your question/task
- `libraryName`: Library to find

### query-docs
Fetch documentation.

Parameters:
- `libraryId`: Context7 ID (e.g., `/vercel/next.js`, `/mongodb/docs`)
- `query`: Specific question

## Workflow

1. **Unknown ID:** resolve-library-id → query-docs
2. **Known ID:** Skip to query-docs

## Common Library IDs
- React: `/facebook/react`
- Next.js: `/vercel/next.js`
- Django: `/django/django`
- FastAPI: `/tiangolo/fastapi`
- MongoDB: `/mongodb/docs`
- Supabase: `/supabase/supabase`

## Example
Task: "Add MongoDB connection pooling"

1. resolve-library-id(query="connection pooling", libraryName="mongodb")
2. query-docs(libraryId="/mongodb/docs", query="connection pooling configuration")
