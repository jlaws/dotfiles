---
type: llm
focus: { source: file, path: migration-order.md }
---

All claims are required.
PASS if the order ports shop/util.py before shop/models.py, shop/models.py before shop/pricing.py, and shop/pricing.py before both shop/plugins/discount.py and shop/handlers.py.
PASS if shop/plugins/discount.py is ported before shop/handlers.py, with the reason that handlers.py loads it by name at run time through importlib, a dependency its import statements do not show.
PASS if the risky patterns include the by-name plugin loading in shop/handlers.py (importlib.import_module with a computed name) as dynamic dispatch that a static import scan misses.
FAIL if the getattr with a default in shop/util.py is reported as a dynamic-dispatch risk, or shop/plugins/discount.py is left out of the order.
