---
type: llm
focus: { source: file, path: docs/upgrade.md }
---

All claims are required.
PASS if the instruction to back up legacy_entries (`ledgerctl backup legacy_entries`) appears before the statement that --drop-legacy permanently deletes the table, so the reader meets the action before the risk.
PASS if the repair step states its condition (verify reports checksum mismatches) before `ledgerctl repair --from-backup`, and the cache flush step states its condition (upgrading from a release older than 4.2) before `ledgerctl cache flush`.
FAIL if any of the seven commands from the draft (systemctl stop ledger-worker, ledgerctl backup legacy_entries, ledgerctl migrate --drop-legacy, ledgerctl verify, ledgerctl repair --from-backup, systemctl start ledger-worker, ledgerctl cache flush) is missing.
