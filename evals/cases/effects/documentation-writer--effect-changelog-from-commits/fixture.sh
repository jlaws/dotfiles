set -euo pipefail
cat > changes.txt <<'TXT'
a1f3c2e feat: add client.widgets.stream() for live widget updates (#310)
b72d9a0 fix: race when creating several widgets at once (#298)
c0e18b4 chore: bump eslint to 9.1
d93f6a7 ci: cache node_modules in the build workflow
e4a02c1 Merge pull request #312 from widgetkit/stream
f5b7d38 refactor: split http.js into two files
a8c61e9 feat!: rename Config to WidgetConfig; update imports
b3d0f57 feat!: drop /v1 endpoints; base URL must end in /v2/
c9e2a14 deprecate: client.widgets.poll() is replaced by stream(); removal in 3.0 (#315)
d1a5b66 fix: default timeout was 10s, now 30s as documented (#301)
e7f4c93 docs: fix typo in contributing guide
TXT
