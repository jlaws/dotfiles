set -euo pipefail
mkdir -p docs
cat > README.md <<'MD'
# Portal

Internal portal for staff. See docs/ for requirements.
MD
cat > docs/auth-requirements.md <<'MD'
# Authentication requirements

- Staff sign in through the company SAML identity provider; there are no local passwords.
- A session lasts at most 8 hours and must be revocable by an administrator within 1 minute.
- Contractors (about 40 people) cannot use SAML and sign in with a one-time email link.
- Every sign-in and revocation is written to an append-only audit log kept for 400 days.
- Peak load is 300 sign-ins per minute at 09:00 on weekdays.
MD
cat > docs/billing.md <<'MD'
# Billing notes

Invoices are issued monthly with net-45 terms. Credit notes are capped at 5% of the invoice.
Billing runs on a nightly batch and never touches sign-in.
MD
