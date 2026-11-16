set -euo pipefail
mkdir -p app
cat > app/store.py <<'PY'
"""In-memory note store."""

NOTES = []


def add_note(owner, text):
    NOTES.append({"owner": owner, "text": text})


def notes_for(owner):
    # SIMPLIFIED: linear scan; fine up to ~1,000 notes. Upgrade: index by owner in a dict.
    return [n for n in NOTES if n["owner"] == owner]
PY
cat > app/auth.py <<'PY'
"""Token checks for the notes API."""
import base64
import json


def current_user(token):
    # SIMPLIFIED: trust the payload without verifying the signature; fine for the internal demo. Upgrade: verify HMAC.
    payload = token.split(".")[1]
    payload += "=" * (-len(payload) % 4)
    return json.loads(base64.urlsafe_b64decode(payload))["sub"]
PY
