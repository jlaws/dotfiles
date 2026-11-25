---
description: "CLAUDE.md holds only what the model cannot infer, has no volatile content, links instead of restating, lists each command with its quiet flag and healthy-run line, and has a Compact instructions section; .claude/skills/code-agent-meta-patterns/SKILL.md; hard because timestamps and counts look harmless and generic advice reads as worth keeping"
tags: [effects, effect]
max_turns: 40
timeout_seconds: 900
allowed_tools: [Skill, Read, Glob, Grep, Agent, TodoWrite]
---

Here is the CLAUDE.md for our Python service. Rewrite it into ./CLAUDE.md so it is as useful as possible for an agent while costing as little context as possible. It is loaded on every turn, should keep the cached prompt prefix stable, and the agent must still know the essentials after a long session is compacted: add a section naming what compaction must keep. List each command with its quiet flag and the line a healthy run prints, so a run can be checked without reading its full output. Link to anything documented elsewhere instead of restating it.

---
# Billing Service
Last updated: 2026-11-02 14:31 UTC. Currently 412 tests, 87% coverage, 38 open issues.

Always write clean, readable code and use meaningful variable names. Remember to write tests. Python is a dynamically typed language created by Guido van Rossum.

## Commands
- Run tests: pytest
- Lint: ruff check .
- Start dev server: make dev (ready when it prints `Uvicorn running on http://127.0.0.1:8000`)

## Architecture
Invoices are immutable once finalized; corrections are new credit-note rows, never edits (an audit requirement). The `ledger/` package is the only code allowed to write to the `entries` table.

## API docs
POST /invoices creates an invoice. POST /invoices/{id}/finalize finalizes it. GET /invoices/{id} returns it. (Full OpenAPI spec is in docs/openapi.yaml.)

## Footguns
Tests that touch `ledger/` need `BILLING_TEST_DB` set, or they silently hit the shared dev database.
---
