---
type: llm
focus: { source: file, path: Dockerfile }
---

All claims are required.
PASS if requirements.txt is copied and installed in a layer before the application source is copied.
PASS if the final stage copies only what it needs (installed packages or a built virtualenv) from the build stage and does not install compilers or pip caches into the final image.
PASS if the container starts uvicorn for app.main:app and exposes or binds port 8000.
FAIL if the USER directive appears only in an earlier stage and not in the final stage.
