---
type: llm
focus: { source: file, path: app/main.py }
---

All claims are required.
PASS if app/main.py passes a lifespan context manager to FastAPI(lifespan=...) that creates the tables before its yield and disposes of the engine after it.
FAIL if app/main.py registers startup or shutdown work with @app.on_event.
