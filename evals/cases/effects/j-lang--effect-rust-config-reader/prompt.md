---
tags: [effects, effect]
max_turns: 40
timeout_seconds: 900
allowed_tools: [Skill, Read, Glob, Grep, Agent, TodoWrite]
---

/kb:j-lang write a small Rust command-line tool in ./main.rs that reads a TOML-like file of key=value lines (path from the first argument), prints the value of the key passed as the second argument, and exits with a nonzero status and a clear message when the file cannot be read, a line is malformed, or the key is missing. Requirements: no unwrap() or expect() anywhere in the file; errors flow through a Result-returning function with a typed error enum; unit tests live in a separate ./tests.rs file, not inline in main.rs.
