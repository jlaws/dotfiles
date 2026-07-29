---
name: config-security-audit
description: "Use when auditing agent configuration security."
allowed-tools: Read, Grep, Glob, Bash
---

# Config Security Audit

Static security scan of the checked-in Claude, Codex, Gemini, and shared knowledge-base configuration. A leaked secret or over-broad permission has a large blast radius. Complements `skill-audit`, which checks conformance and structure but not security.

## Scope

Scan every configuration surface across all trees. The repo root is the base.

| Tree | Files |
|------|-------|
| Claude | `.claude/CLAUDE.md`, `.claude/settings.json`, `.claude/settings.local.json`, `.claude/hooks/`, `.claude/agents/`, `.claude/commands/` |
| Codex | `.codex/AGENTS.md`, `.codex/config.toml`, `.codex/rules/`, `.codex/hooks/`, `.codex/agents/`, `.codex/prompts/` |
| Gemini | `.gemini/GEMINI.md`, `.gemini/settings.json`, `.gemini/hooks/`, `.gemini/agents/`, `.gemini/commands/` |
| Shared | `.agents/skills/`, `.agents/references/`, any MCP server config |

## Checks

Track each as PASS / WARN / FAIL. Report actual values (redacted) on failure. Exit non-zero if any FAIL is CRITICAL so CI can gate.

### A. Leaked secrets (CRITICAL)

| # | Check |
|---|-------|
| SEC-1 | No API keys or tokens (`sk-`, `ghp_`, `AKIA`, `xox[baprs]-`, `AIza`, bearer tokens) in any config file |
| SEC-2 | No private keys (`BEGIN (RSA\|EC\|OPENSSH\|PGP) PRIVATE KEY`) |
| SEC-3 | No hardcoded passwords or connection strings with embedded credentials |
| SEC-4 | Env values are referenced by name (`${VAR}`), never inlined |
| SEC-5 | `settings.local.json` (machine-local) is gitignored |

### B. Over-broad tool permissions (WARN/FAIL)

| # | Check |
|---|-------|
| PERM-1 | `settings.json` permission allow-lists are least-privilege — flag wildcard `Bash(*)` or blanket allow-all |
| PERM-2 | Agent `tools:` fields grant only what the role needs — flag write/exec tools on read-only reviewer/audit agents |
| PERM-3 | Hook allow-lists and `.codex/rules/default.rules` do not auto-approve destructive ops (`rm -rf`, force-push, curl-to-shell) |
| PERM-4 | No hook or command runs `eval` or pipes remote content straight to a shell |

### C. Prompt-injection vectors (WARN/FAIL)

| # | Check |
|---|-------|
| INJ-1 | Commands/skills that ingest external content (WebFetch, reads of untrusted docs) treat it as data, not instructions |
| INJ-2 | No config file contains hidden or zero-width characters or off-screen instruction text |
| INJ-3 | Agent/command prose does not blindly execute instructions embedded in fetched or piped content |
| INJ-4 | MCP server configs pin trusted sources; no arbitrary remote instruction sources |

## Process

1. Enumerate the scope files (Glob per tree).
2. Run the checks (Grep patterns for A and B; read prose for C).
3. Report grouped by severity; redact any secret you surface — show only the match location and type, never the value.
4. Exit non-zero if any CRITICAL finding exists (a leaked secret or a destructive auto-approve).

## Report

```
Config Security Audit
=====================
Secrets:      {n} checks | {pass} pass | {fail} fail
Permissions:  {n} checks | {pass} pass | {warn} warn | {fail} fail
Injection:    {n} checks | {pass} pass | {warn} warn | {fail} fail
---------------------------------------------------------------
Result: PASS | CONCERNS | FAIL   (FAIL exits non-zero)
```

List each finding as `[SEVERITY] CHECK-ID: file:line — what (redacted)`.

## Cross-References

- `skill-audit` — conformance/structure audit; run alongside, this is the security lens
- `security-analysis` — STRIDE + SAST for application code
- `references/security/secrets-management` — secret-handling patterns
- `verification-before-completion` — verdict grammar (PASS/CONCERNS/FAIL/BLOCKED), read-only contract
