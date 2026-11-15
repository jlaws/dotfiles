set -euo pipefail
cat > app.py <<'PY'
import hashlib
import os
import sqlite3

SECRET_KEY = "dev-secret-123"


def get_db():
    return sqlite3.connect("app.db")


def login(username, password):
    db = get_db()
    row = db.execute("SELECT id, pw FROM users WHERE name = '%s'" % username).fetchone()
    if row and row[1] == hashlib.md5(password.encode()).hexdigest():
        return row[0]
    return None


def get_profile(user_id):
    return get_db().execute("SELECT name, email FROM users WHERE id = ?", (user_id,)).fetchone()


def export_report(name):
    os.system("tar czf out/%s.tgz reports/%s" % (name, name))
PY
