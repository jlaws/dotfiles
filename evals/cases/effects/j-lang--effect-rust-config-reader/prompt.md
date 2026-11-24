---
tags: [effects, effect]
max_turns: 40
timeout_seconds: 900
allowed_tools: [Skill, Read, Glob, Grep, Agent, TodoWrite]
description: "write a Rust CLI with no unwrap or expect, errors through a Result and a typed error enum, and unit tests in a separate file; .claude/commands/j-lang.md; hard because quick Rust tools reach for unwrap and inline test modules"
---

/kb:j-lang write a small Rust command-line tool in ./main.rs that reads a TOML-like file of key=value lines (path from the first argument), prints the value of the key passed as the second argument, and exits with a nonzero status and a clear message when the file cannot be read, a line is malformed, or the key is missing. Requirements: no unwrap() or expect() anywhere in the file; errors flow through a Result-returning function with a typed error enum; unit tests live in a separate ./tests.rs file, not inline in main.rs.
