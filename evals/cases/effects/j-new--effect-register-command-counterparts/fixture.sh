set -euo pipefail
cp -R "$1/kb-mini/." .
mkdir -p .agents/skills/cmd-j-demo .codex/prompts .gemini/antigravity-cli/skills/j-demo
cat > .agents/skills/cmd-j-demo/SKILL.md <<'MD'
---
name: cmd-j-demo
description: "Use when invoking the j-demo workflow."
---

# Demo

Print the arguments the user passed.
MD
cat > .codex/prompts/j-demo.md <<'MD'
---
name: j-demo
description: "Demo command. Use when demonstrating the command format."
---

Print the arguments the user passed: $ARGUMENTS
MD
cat > .gemini/antigravity-cli/skills/j-demo/SKILL.md <<'MD'
---
name: j-demo
description: "Demo command. Use when demonstrating the command format."
---

# Demo

Print the arguments the user passed: $ARGUMENTS
MD
