set -euo pipefail
mkdir -p invoices
cat > invoices/views.py <<'PY'
"""Invoice endpoints. `request.user` is set by the auth middleware."""
from invoices.models import Invoice


def get_invoice(request, invoice_id):
    invoice = Invoice.objects.get(id=invoice_id)
    return {"id": invoice.id, "owner": invoice.owner_id, "total": invoice.total}


def list_my_invoices(request):
    rows = Invoice.objects.filter(owner_id=request.user.id)
    return [{"id": r.id, "total": r.total} for r in rows]
PY
cat > invoices/cli.py <<'PY'
"""Command-line tool an invoice owner runs on their own machine: python -m invoices.cli <path>."""
import sys


def download_own_export(path):
    with open(path) as handle:
        return handle.read()


if __name__ == "__main__":
    print(download_own_export(sys.argv[1]))
PY
cat > invoices/settings.py <<'PY'
"""Settings."""
ALLOWED_HOSTS = ["*"]
SESSION_COOKIE_SAMESITE = None
PY
