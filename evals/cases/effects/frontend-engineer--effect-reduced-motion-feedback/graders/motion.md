---
type: llm
focus: { source: file, path: src/motion.fixed.css }
---

All claims are required.
PASS if the reduced-motion rules (prefers-reduced-motion: reduce) make the hero drift and the carousel scroll stop looping, and make the card hover and the drawer no longer animate a movement (a position change may happen at once).
PASS if under the reduced-motion rules the toast still fades in through an opacity animation and the primary button still transitions its background color, each with a visible duration rather than one cut to near zero (exempting the toast by its class or by a markup attribute such as data-motion="feedback" both count).
FAIL if a user without reduced motion turned on would see any animation or transition from src/motion.css removed or changed in src/motion.fixed.css.
