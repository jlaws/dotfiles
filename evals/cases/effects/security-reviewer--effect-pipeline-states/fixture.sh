set -euo pipefail
mkdir -p api
cat > api/db.py <<'PY'
"""Data access for the orders API."""
import sqlite3

conn = sqlite3.connect("orders.db")


def find_orders(customer_id, status):
    query = f"SELECT * FROM orders WHERE customer_id = {customer_id} AND status = '{status}'"
    return conn.execute(query).fetchall()


def find_order(order_id):
    return conn.execute("SELECT * FROM orders WHERE id = ?", (order_id,)).fetchone()
PY
cat > api/export.py <<'PY'
"""Export helpers."""
import subprocess


def archive(report_name):
    # report_name arrives from the HTTP query string
    return subprocess.run("tar czf /tmp/" + report_name + ".tgz reports/", shell=True, check=True)


def list_reports():
    return subprocess.run(["ls", "reports/"], capture_output=True, text=True).stdout
PY
cat > api/config.py <<'PY'
"""Settings."""
import os

PAYMENT_API_SECRET = "7f3a9c2e-payments-hmac-d41b"
DEBUG = os.environ.get("DEBUG", "0") == "1"
PY
