---
type: llm
focus: { source: file, path: src/motion.fixed.css }
---

All claims are required.
PASS if the reduced-motion rules (prefers-reduced-motion: reduce) make the hero drift and the carousel scroll stop looping, and make the card hover and the drawer no longer animate a movement (a position change may happen at once).
PASS if under the reduced-motion rules the toast still fades in through an opacity animation and the primary button still transitions its background color, each with a visible duration rather than one cut to near zero (exempting the toast by its class or by a markup attribute such as data-motion="feedback" both count).
FAIL if, outside the prefers-reduced-motion: reduce block, src/motion.fixed.css is missing or changes any of these: the .hero-bg drift animation (20s ease-in-out infinite alternate), the .card transform and box-shadow transitions (200ms ease-out) with the .card:hover lift and scale (translateY(-4px) scale(1.03)), the .drawer transform transition (300ms ease-out) that slides it in, the .carousel-track carousel animation (30s linear infinite), the .toast toast-in opacity animation (200ms ease-out), or the .btn-primary background-color transition (150ms ease-out).
