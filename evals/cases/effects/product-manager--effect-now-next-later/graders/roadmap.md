---
type: llm
focus: { source: file, path: roadmap.md }
---

All claims are required.
PASS if roadmap.md has Now, Next, and Later tiers with at most 2 items in Now, and a RICE score shown for each of the five candidates.
PASS if each Now item has Given/When/Then acceptance criteria, each Next item has a problem statement with evidence, and each Later item is a single line.
PASS if the AI chatbot is in the Not building list (or lowest tier) because it has no user evidence, and the Not building list gives each entry a reason and a revisit condition.
FAIL if bulk user invite, the item with the strongest evidence, is not in Now or Next.
