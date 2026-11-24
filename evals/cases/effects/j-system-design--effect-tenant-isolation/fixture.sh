set -euo pipefail
mkdir -p docs
cat > README.md <<'MD'
# Clinic scheduling

Appointment booking for clinics. See docs/ for requirements.
MD
cat > docs/requirements.md <<'MD'
# Tenancy requirements

- 3,400 tenants today: solo practitioners and small practices on the Free and Team plans. Each
  holds under 50 MB of appointment data. About 60 new tenants sign up self-serve each week.
- 14 tenants are hospital groups on the Enterprise plan. Their contracts require HIPAA compliance
  for patient data.
- Finance needs a monthly report of appointments booked per tenant, across all tenants.
- One PostgreSQL 16 cluster runs everything today, with every tenant in the same tables, keyed by
  tenant_id.
MD
