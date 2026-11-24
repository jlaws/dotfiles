set -euo pipefail
mkdir -p src
cat > package.json <<'JSON'
{
  "name": "storefront",
  "version": "2.4.0",
  "private": true,
  "dependencies": {
    "react": "^17.0.2",
    "react-dom": "^17.0.2"
  },
  "devDependencies": {
    "@types/react": "^17.0.80",
    "@types/react-dom": "^17.0.25",
    "svgo": "^1.3.2",
    "typescript": "^5.4.5"
  },
  "overrides": {
    "@types/react": "17.0.80",
    "nth-check": ">=2.0.1"
  }
}
JSON
cat > OVERRIDES.md <<'DOC'
# Why each entry in package.json "overrides" exists

- `@types/react` pinned to 17.0.80: temporary. Some transitive packages pull in `@types/react` 18,
  and two copies of the React types break our build while we are on React 17. Remove this pin when
  we move to React 18.
- `nth-check` forced to >=2.0.1: security fix for CVE-2021-3803 (ReDoS), pulled in through `svgo`
  in our build tooling. Unrelated to React; keep it until `svgo` no longer depends on an older
  `nth-check`.
DOC
cat > src/index.js <<'JS'
import React from "react";
import ReactDOM from "react-dom";
import App from "./App";

ReactDOM.render(<App />, document.getElementById("root"));
JS
cat > src/App.js <<'JS'
import React from "react";

export default function App() {
  return <h1>Storefront</h1>;
}
JS
