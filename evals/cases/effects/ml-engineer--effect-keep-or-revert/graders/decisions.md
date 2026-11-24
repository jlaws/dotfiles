---
type: llm
focus: { source: file, path: keep-or-revert.md }
---

All claims are required.
PASS if keep-or-revert.md keeps A and reverts C, giving C's intervals including zero (a gain within noise) as the reason.
PASS if keep-or-revert.md reverts B because its gain shows on train only while its test interval includes zero.
PASS if keep-or-revert.md keeps E because the largest loss its test interval allows (1.5 points) is inside the 2-point margin.
PASS if keep-or-revert.md reverts D because its test interval allows a loss of 2.6 points, beyond the 2-point margin, even though its point estimate (-0.8) is inside it.
FAIL if keep-or-revert.md keeps B or keeps D.
