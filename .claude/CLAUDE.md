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
- Store intermediate results in filesystem (HANDOFF.md, scratch files) rather than relying on conversation memory for long-running work.
- When approaching context limits: summarize completed work into a handoff file BEFORE context degrades.
- For multi-step investigations: write findings to files progressively; don't accumulate everything in conversation.

## Context Efficiency
- Critical info at **beginning or end** of prompts/files — middle content gets lower attention weight.
- Prefer tables and code over prose (higher information density per token).
- Search first (Glob/Grep), then Read only confirmed-relevant files — avoid speculative bulk reads.
- When processing web/external content: strip boilerplate, nav, ads; convert HTML to Markdown.
- Link to detailed docs; never inline >50 lines into CLAUDE.md or skills.

## Git Workflow
- Commit messages: freeform imperative mood, <72 char subject, no period
- Prefer small, atomic commits
- Always verify changes with `git diff` before committing
- Never force push to main/master
- Branch naming: `type/short-description` (e.g., `fix/login-timeout`)

## Bash Commands

Never chain commands with `&&`, `||`, or `;`. Run each command as a separate Bash tool call to avoid compound-command permission prompts.

## Worktree Rules

**Always load** skill `workflow/multi-agent-development` when spawning subagents with worktree isolation.

When running in a worktree (`isolation: "worktree"`):
- **Commit ALL changes** before returning — uncommitted work is invisible to `git merge`
- **Squash into one commit**:
  ```bash
  git add -A && git reset --soft $(git merge-base HEAD main 2>/dev/null || git merge-base HEAD master) && git commit -m "<summary>"
  ```
- **NEVER** copy files out (`cp`, `rsync`, file-copy) — parent uses `git merge` to integrate
- **NEVER** clean up your own worktree — parent handles merge + `git worktree remove`
- **NEVER** invoke `finishing-branch` skill — return changes on-branch to parent

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
