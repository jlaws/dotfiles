---
type: llm
focus: { source: file, path: config-audit.md }
---

All claims are required.
PASS if config-audit.md reports the hardcoded AWS access key in .claude/settings.json as a critical secret finding without printing the key value.
PASS if config-audit.md reports the blanket Bash(*) permission, the pipe-to-shell in the hook script, and the reviewer agent holding Write and Edit tools despite being read-only.
PASS if config-audit.md does not report the .codex/config.toml API_TOKEN as a leaked secret (it is a ${VAR} reference).
FAIL if the overall result is PASS, or findings lack a file:line location.
