---
type: llm
focus: { source: file, path: deploy/checkout-api.yaml }
---

All claims are required.
PASS if the api container sets both a CPU request and a memory request.
PASS if the api container sets no CPU limit.
PASS if the api container's memory limit is from 1.5 to 2 times its memory request (for example, a 600Mi request with a limit from 900Mi to 1200Mi).
FAIL if the api container keeps the seeded pattern of a limit equal to its request for CPU or memory (the manifest starts with cpu "1" and memory 1Gi as both the request and the limit).
