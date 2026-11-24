set -euo pipefail
mkdir -p docs
cat > docs/upgrade.md <<'MD'
# Upgrading ledger to 5.0

Stop the worker first, you can do this with `systemctl stop ledger-worker`, and then run the
migration with `ledgerctl migrate --drop-legacy`. Warning: `--drop-legacy` permanently deletes the
legacy_entries table, so you should take a backup with `ledgerctl backup legacy_entries` before you
run the migration.

After that run `ledgerctl verify`. Run `ledgerctl repair --from-backup` if verify reports checksum
mismatches.

Then start the worker again with `systemctl start ledger-worker`. Run `ledgerctl cache flush` too,
but only when you are upgrading from a release older than 4.2.
MD
