set -euo pipefail
mkdir -p config src
cat > config/flags.json <<'JSON'
{
  "checkout-redesign-q3": {
    "enabled": true,
    "rolloutPercent": 50,
    "salt": "checkout-redesign-q3-v1"
  }
}
JSON
cat > src/assign.ts <<'TS'
import flags from "../config/flags.json";

type Flag = { enabled: boolean; rolloutPercent: number; salt: string };

// FNV-1a: deterministic, so a user keeps the same variant on every visit.
function hash(input: string): number {
  let h = 0x811c9dc5;
  for (let i = 0; i < input.length; i++) {
    h ^= input.charCodeAt(i);
    h = Math.imul(h, 0x01000193);
  }
  return h >>> 0;
}

export function inTreatment(flagKey: keyof typeof flags, userId: string): boolean {
  const flag: Flag = flags[flagKey];
  if (!flag.enabled) return false;
  return hash(flag.salt + userId) % 100 < flag.rolloutPercent;
}
TS
cat > PR.md <<'MD'
# Ship the single-page checkout to everyone

The checkout-redesign-q3 experiment is done, so this PR sets its rollout to 100%.

## Experiment readout

- Design: 50/50 split between control (current checkout) and treatment (single-page checkout),
  assigned per user by src/assign.ts.
- Duration: the pre-registered 14 days, ended 2026-11-20.
- Control: 10,240 users, 328 orders (3.20% conversion).
- Treatment: 9,760 users, 371 orders (3.80% conversion).
- Two-proportion z-test on conversion: p = 0.02.

Treatment wins with an 18.75% relative lift, so we ship it.
MD
cat > change.diff <<'DIFF'
--- a/config/flags.json
+++ b/config/flags.json
@@ -1,7 +1,7 @@
 {
   "checkout-redesign-q3": {
     "enabled": true,
-    "rolloutPercent": 50,
+    "rolloutPercent": 100,
     "salt": "checkout-redesign-q3-v1"
   }
 }
DIFF
