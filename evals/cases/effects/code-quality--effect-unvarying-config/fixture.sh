set -euo pipefail
mkdir -p app
cat > app/__init__.py <<'PY'
PY
cat > app/http.py <<'PY'
import time
import urllib.request


def fetch(url, timeout, retries=3):
    """Fetch url, retrying on OSError."""
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as resp:
                return resp.read()
        except OSError:
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)
PY
cat > app/prices.py <<'PY'
from app.http import fetch


def latest_prices():
    # TODO(#412): switch to the bulk endpoint once the vendor enables it for our account
    return fetch("https://prices.example.invalid/latest", timeout=5, retries=3)
PY
cat > app/stock.py <<'PY'
from app.http import fetch


def stock_levels():
    return fetch("https://stock.example.invalid/levels", timeout=30, retries=3)
PY
cat > app/rates.py <<'PY'
from app.http import fetch


def fx_rates():
    return fetch("https://fx.example.invalid/rates", timeout=10)
PY
cat > app/export.py <<'PY'
import csv


class BaseExporter:
    """Base class for exporters."""

    def export(self, rows, path):
        raise NotImplementedError


class CsvExporter(BaseExporter):
    def export(self, rows, path):
        with open(path, "w", newline="") as handle:
            csv.writer(handle).writerows(rows)


def export_report(rows, path):
    return CsvExporter().export(rows, path)
PY
