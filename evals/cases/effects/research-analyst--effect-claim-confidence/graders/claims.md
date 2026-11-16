---
type: llm
focus: { source: file, path: claims.md }
---

All claims are required.
PASS if Claim A scores no higher than 0.6 and is described as a single, biased (author of X) source.
PASS if Claim B scores highest of the four, in the 0.8 to 0.95 range, because three independent high-quality sources agree.
PASS if Claim C names both positions (docs say yes, forum says no), notes the changelog resolves the recency conflict, and flags the contradiction.
PASS if Claim D is scored below Claim B and no higher than 0.8, with the two blog posts treated as community-quality sources (about 0.5 to 0.6 each).
FAIL if claims.md scores Claim A above 0.6 or omits the contradiction in Claim C.
