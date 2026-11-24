---
type: llm
focus: { source: file, path: src/Tabs.tsx }
---

All claims are required.
PASS if the tab strip has role="tablist", each tab has role="tab" with aria-selected and aria-controls, and each panel has role="tabpanel" with aria-labelledby naming its tab.
PASS if the Left and Right arrow keys move between tabs, and the Home and End keys move to the first and the last tab.
FAIL if Home or End has no key handling, or the arrow keys do not move between tabs.
