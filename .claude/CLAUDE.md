# Claude Code Configuration

## #1 Rule: Verify Before Claiming
Evidence before claims. Never claim success without fresh command output confirming it.
Full methodology: load skill `workflow/verification-before-completion`.

## Behavioral Defaults
- Before creative/feature work: explore intent + requirements before implementation
- For design decisions: propose 2-3 approaches, lead with recommendation
- **Skill lookup**: Before ANY implementation action, check `.claude/skills/` and `.claude/references/` for applicable workflows. Full methodology: load skill `workflow/skill-lookup-discipline`. Process skills first (design-first, debugging), then implementation skills (code-quality, TDD).

## Context Preservation
- On compaction: preserve current task, file paths being edited, test results, and key decisions. Discard exploration output and intermediate reasoning.

## Git Workflow
- Commit messages: freeform imperative mood, <72 char subject, no period
- Prefer small, atomic commits
- Always verify changes with `git diff` before committing
- Never force push to main/master
- Branch naming: `type/short-description` (e.g., `fix/login-timeout`)

## Team Conventions
When spawned as a teammate: load and follow `workflow/multi-agent-development` skill.
Key rules: claim tasks via TaskUpdate, only edit declared files, DM the lead (never broadcast), verify before marking complete.

---

## Knowledge Base Structure
- **skills/**: Cross-cutting workflows loaded on demand (code review, debugging, TDD, etc.)
- **references/**: Domain knowledge loaded on-demand by agents and commands
- **agents/**: Specialist roles that read from references/
- **commands/**: Entry points that gather context then invoke agents/skills

---

## Communication Style

### Do
- Be concise and direct. No filler.
- Lead with the answer, explain after if needed.
- Use bullet points and code examples.
- Assume I'm an experienced developer.
- Challenge my assumptions when appropriate.
- Ask clarifying questions rather than guessing.
- Be extremely concise; sacrifice grammar for brevity.
- End plans with unresolved questions list (concise, skip grammar).
- Structure plans in multiple phases.

### Don't
- Over-explain basic concepts.
- Add unnecessary caveats or warnings.
- Repeat requirements back to me.
- Use excessive praise or encouragement.

---
