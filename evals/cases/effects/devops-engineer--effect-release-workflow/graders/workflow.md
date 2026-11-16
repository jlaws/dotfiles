---
type: llm
focus: { source: file, path: release.fixed.yml }
---

All claims are required.
PASS if the deploy job maps DEPLOY_TOKEN explicitly (for example DEPLOY_TOKEN: ${{ secrets.DEPLOY_TOKEN }}) under secrets.
PASS if a deployment can no longer be cancelled by a newer run: cancel-in-progress is false or the concurrency block is removed from the deploy path.
FAIL if cancel-in-progress is still true for a group that covers the deploy job.
