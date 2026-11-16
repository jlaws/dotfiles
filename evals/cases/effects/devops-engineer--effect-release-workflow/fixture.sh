set -euo pipefail
mkdir -p .github/workflows
cat > .github/workflows/release.yml <<'YAML'
name: release
on:
  push:
    tags: ['v*']
concurrency:
  group: release
  cancel-in-progress: true
jobs:
  changelog:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Build changelog
        run: |
          PREV=$(git describe --tags --abbrev=0 HEAD^)
          git log --format='- %s' "$PREV"..HEAD > CHANGES.md
      - uses: actions/upload-artifact@v4
        with:
          name: changes
          path: CHANGES.md
  deploy:
    needs: changelog
    uses: ./.github/workflows/deploy.yml
    secrets: inherit
YAML
cat > .github/workflows/deploy.yml <<'YAML'
name: deploy
on:
  workflow_call:
    secrets:
      DEPLOY_TOKEN:
        required: true
jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - run: ./scripts/deploy.sh
        env:
          DEPLOY_TOKEN: ${{ secrets.DEPLOY_TOKEN }}
YAML
