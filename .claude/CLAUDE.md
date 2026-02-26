# Claude Code Configuration

## Verification Gate
Evidence before claims. Never claim success without fresh command output confirming it.
Full methodology: load skill `workflow/verification-before-completion`.

## Behavioral Defaults
- Before creative/feature work: explore intent + requirements before implementation
- For design decisions: propose 2-3 approaches, lead with recommendation
- **Skill lookup**: Before implementation tasks involving a specific framework, language pattern, or architecture decision — check `.claude/skills/` and `.claude/references/` for relevant patterns before relying on training knowledge.

## Context Preservation
- On compaction: preserve current task, file paths being edited, test results, and key decisions. Discard exploration output and intermediate reasoning.

## Git Workflow
- Commit messages: freeform imperative mood, <72 char subject, no period
- Prefer small, atomic commits
- Always verify changes with `git diff` before committing
- Never force push to main/master
- Branch naming: `type/short-description` (e.g., `fix/login-timeout`)

## Team Conventions
When spawned as a teammate, follow these rules (teammates read this file on startup):
- **Task discipline**: claim via TaskUpdate (set owner), mark completed when done, check TaskList for next work
- **File ownership**: only edit files declared in your task — never touch files outside your assignment
- **Communication**: DM the lead via SendMessage; never broadcast unless truly critical (blocking issue affecting all agents)
- **Quality**: verify your work (run tests, read output) before marking a task complete
- **Shutdown**: respond to `shutdown_request` promptly — approve unless you have in-flight uncommitted work
- **Context**: include file paths and line numbers when referencing code in messages

### Subagents vs Teams

| Use | When |
|-----|------|
| **Task tool (subagent)** | Independent, self-contained work: research, exploration, single-file edits, running tests |
| **TeamCreate (full team)** | Coordinated multi-file work requiring shared task list, communication, and file ownership |

**Default to subagents** unless tasks have cross-file dependencies or require coordination.

### File Conflict Prevention
- Declare file ownership in task descriptions — one agent per file
- If you need to edit an unowned file, DM the lead first
- Never edit files another agent is working on

### Common Prompt Mistakes

| Mistake | Fix |
|---------|-----|
| Vague task description | Include specific files, acceptance criteria, and constraints |
| No file ownership declared | List exact files each agent may edit |
| Broadcasting status updates | DM the lead; only broadcast blocking issues |
| Skipping verification | Always run tests/build before marking complete |

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
