set -euo pipefail
mkdir -p reports
cat > reports/__init__.py <<'PY'
PY
cat > reports/db.py <<'PY'
"""Database connection for the reports service."""
import sqlite3

conn = sqlite3.connect("reports.db")
PY
cat > reports/search.py <<'PY'
"""Report search endpoint. `request.args` holds the HTTP query string."""
from legacy_sql import legacy_escape  # from the acme-legacy wheel installed on the servers; its source is not in this repository
from reports.db import conn


def search_reports(request):
    term = legacy_escape(request.args["q"])
    sql = "SELECT id, title FROM reports WHERE title LIKE '%" + term + "%'"
    return conn.execute(sql).fetchall()
PY
cat > README.md <<'MD'
# reports

Report search service. Production runs at https://reports.example.internal and reads the
production reports database; there is no staging copy.
MD
