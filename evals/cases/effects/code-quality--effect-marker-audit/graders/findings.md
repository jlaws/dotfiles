---
type: llm
focus: { source: file, path: quality-review.md }
---

All claims are required.
PASS if the file flags the SIMPLIFIED marker on login (lines 7-9) as a finding because it sits on an authentication control, regardless of the fields it carries.
PASS if the file flags the marker at line 18 as incomplete because it names neither a ceiling nor an upgrade path.
PASS if the file flags the bare TODO at line 25 (no ticket, no reason, and an empty pass body).
PASS if the Sanctioned line names find_item (lines 11-16) and does not name login.
FAIL if find_item is reported as a finding merely for being a linear scan, or if the login marker is called acceptable.
