---
name: accessibility-testing
description: "Audit and test web accessibility with automated scanning, manual screen reader testing, and WCAG 2.2 remediation. Use when auditing accessibility, fixing WCAG violations, testing with screen readers, or implementing accessible components."
---

# Accessibility Testing

## Automated Testing Setup

### axe-core with Puppeteer

```javascript
const { AxePuppeteer } = require('@axe-core/puppeteer');
const puppeteer = require('puppeteer');

class AccessibilityAuditor {
    constructor(options = {}) {
        this.wcagLevel = options.wcagLevel || 'AA';
        this.viewport = options.viewport || { width: 1920, height: 1080 };
    }

    async runFullAudit(url) {
        const browser = await puppeteer.launch();
        const page = await browser.newPage();
        await page.setViewport(this.viewport);
        await page.goto(url, { waitUntil: 'networkidle2' });

        const results = await new AxePuppeteer(page)
            .withTags(['wcag2a', 'wcag2aa', 'wcag21a', 'wcag21aa'])
            .exclude('.no-a11y-check')
            .analyze();

        await browser.close();

        return {
            url,
            timestamp: new Date().toISOString(),
            violations: results.violations.map(v => ({
                id: v.id, impact: v.impact,
                description: v.description, help: v.help, helpUrl: v.helpUrl,
                nodes: v.nodes.map(n => ({
                    html: n.html, target: n.target, failureSummary: n.failureSummary
                }))
            })),
            score: this.calculateScore(results)
        };
    }

    calculateScore(results) {
        const weights = { critical: 10, serious: 5, moderate: 2, minor: 1 };
        let totalWeight = 0;
        results.violations.forEach(v => { totalWeight += weights[v.impact] || 0; });
        return Math.max(0, 100 - totalWeight);
    }
}
```

### Component Testing with jest-axe

```javascript
import { render } from '@testing-library/react';
import { axe, toHaveNoViolations } from 'jest-axe';

expect.extend(toHaveNoViolations);

it('should have no violations', async () => {
    const { container } = render(<MyComponent />);
    const results = await axe(container);
    expect(results).toHaveNoViolations();
});
```

### CLI Tools

```bash
npx @axe-core/cli https://example.com
npx pa11y https://example.com --standard WCAG2AA --threshold 0
lighthouse https://example.com --only-categories=accessibility
```

### CI/CD Integration

```yaml
# .github/workflows/accessibility.yml
name: Accessibility Tests
on: [push, pull_request]
jobs:
  a11y-tests:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    - uses: actions/setup-node@v3
      with: { node-version: '18' }
    - run: npm ci && npm run build
    - run: npm start & npx wait-on http://localhost:3000
    - run: npm run test:a11y
    - run: npx pa11y http://localhost:3000 --standard WCAG2AA --threshold 0
```

## Manual Screen Reader Testing

### Testing Priority

```
Minimum: NVDA + Firefox (Win), VoiceOver + Safari (Mac), VoiceOver + Safari (iOS)
Full:    + JAWS + Chrome (Win), TalkBack + Chrome (Android)
```

### VoiceOver (macOS) Commands

```
VO = Ctrl + Option
Toggle:            Cmd + F5
Next/Prev element: VO + Right/Left Arrow
Enter/Exit group:  VO + Shift + Down/Up
Read all:          VO + A
Activate:          VO + Space
Rotor:             VO + U (navigate by headings, links, forms, landmarks)
Next heading:      VO + Cmd + H
Next form control: VO + Cmd + J
Next link:         VO + Cmd + L
```

### NVDA (Windows) Commands

```
NVDA modifier = Insert
Say all:           NVDA + Down Arrow
Elements list:     NVDA + F7
Quick keys (browse mode):
  H/Shift+H  Heading next/prev    D/Shift+D  Landmark next/prev
  F           Next form field      B           Next button
  K           Next link            T           Next table
Table nav:         Ctrl + Alt + Arrows
Mode switch:       NVDA + Space (browse <-> focus)
```

### JAWS Quick Keys

```
H  Next heading    T  Next table      F  Next form field
B  Next button     ;  Next landmark   G  Next graphic
Insert + F7  Link list    Insert + F6  Heading list
```

### Test Script (All Screen Readers)

1. **Page load** - Title announced? Main landmark found? Skip link works?
2. **Landmark nav** - All main areas reachable? Properly labeled?
3. **Heading nav** - Logical structure? All sections discoverable?
4. **Form testing** - Labels read? Required fields announced? Errors announced? Focus moved to error?
5. **Interactive elements** - Each announces role + state? Activates with Enter/Space?
6. **Dynamic content** - Updates announced? Modal traps focus? Focus returns on close?

### Common Screen Reader Fixes

```html
<!-- Button without visible text -->
<button aria-label="Close dialog"><svg aria-hidden="true">...</svg></button>

<!-- Dynamic content not announced -->
<div role="status" aria-live="polite">New results loaded</div>

<!-- Form error not read -->
<input type="email" aria-invalid="true" aria-describedby="email-error">
<span id="email-error" role="alert">Invalid email</span>
```

## WCAG 2.2 Audit Checklist

### Critical Violations (Blockers)

- [ ] All functional images have alt text; decorative images `alt=""`
- [ ] All interactive elements keyboard accessible (no traps)
- [ ] All form inputs have associated labels
- [ ] Color contrast: 4.5:1 text, 3:1 large text/UI components
- [ ] No auto-playing media without controls

### Serious Violations

- [ ] Skip to main content link present
- [ ] Page titles unique and descriptive
- [ ] Heading hierarchy logical (no skipped levels)
- [ ] ARIA landmarks defined (main, nav, etc.)
- [ ] Focus indicator visible on all elements (3:1 contrast)
- [ ] `<html lang="en">` attribute set

### Forms & Interaction

- [ ] Error messages identify field and describe problem
- [ ] Required fields indicated (not color-only)
- [ ] `aria-invalid="true"` + `aria-describedby` on error fields
- [ ] Live regions: `role="status"` (polite) / `role="alert"` (assertive)
- [ ] Modal dialogs: `role="dialog"`, `aria-modal="true"`, focus trap, Esc to close

### Responsive & Motion

- [ ] Content reflows at 320px (no horizontal scroll)
- [ ] Text resizes to 200% without loss
- [ ] `@media (prefers-reduced-motion: reduce)` disables animations
- [ ] `@media (prefers-contrast: high)` increases contrast

## Remediation Patterns

### Keyboard Navigation for Custom Widgets

```javascript
class AccessibleDropdown extends HTMLElement {
  connectedCallback() {
    this.setAttribute('tabindex', '0');
    this.setAttribute('role', 'combobox');
    this.setAttribute('aria-expanded', 'false');

    this.addEventListener('keydown', (e) => {
      switch (e.key) {
        case 'Enter': case ' ': this.toggle(); e.preventDefault(); break;
        case 'Escape': this.close(); break;
        case 'ArrowDown': this.focusNext(); e.preventDefault(); break;
        case 'ArrowUp': this.focusPrevious(); e.preventDefault(); break;
      }
    });
  }
}
```

### Focus Management for Modals

```javascript
function openModal(modal) {
  const lastFocus = document.activeElement;
  modal.querySelector('h2').focus();

  modal.addEventListener('keydown', (e) => {
    if (e.key === 'Tab') {
      const focusable = modal.querySelectorAll(
        'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])'
      );
      const first = focusable[0], last = focusable[focusable.length - 1];
      if (e.shiftKey && document.activeElement === first) { last.focus(); e.preventDefault(); }
      else if (!e.shiftKey && document.activeElement === last) { first.focus(); e.preventDefault(); }
    }
    if (e.key === 'Escape') { modal.hidden = true; lastFocus.focus(); }
  });
}
```

### Tab Interface Pattern

```html
<div role="tablist" aria-label="Product info">
  <button role="tab" id="tab-1" aria-selected="true" aria-controls="panel-1">Description</button>
  <button role="tab" id="tab-2" aria-selected="false" aria-controls="panel-2" tabindex="-1">Reviews</button>
</div>
<div role="tabpanel" id="panel-1" aria-labelledby="tab-1">...</div>
<div role="tabpanel" id="panel-2" aria-labelledby="tab-2" hidden>...</div>
```

```javascript
tablist.addEventListener('keydown', (e) => {
  const tabs = [...tablist.querySelectorAll('[role="tab"]')];
  const index = tabs.indexOf(document.activeElement);
  let newIndex;
  switch (e.key) {
    case 'ArrowRight': newIndex = (index + 1) % tabs.length; break;
    case 'ArrowLeft': newIndex = (index - 1 + tabs.length) % tabs.length; break;
    case 'Home': newIndex = 0; break;
    case 'End': newIndex = tabs.length - 1; break;
    default: return;
  }
  tabs[newIndex].focus();
  activateTab(tabs[newIndex]);
  e.preventDefault();
});
```

### Color Contrast Fix

```css
/* High contrast mode support */
@media (prefers-contrast: high) {
    :root { --text-primary: #000; --bg-primary: #fff; --border-color: #000; }
    a { text-decoration: underline !important; }
    button, input { border: 2px solid var(--border-color) !important; }
}

/* Visible focus indicators */
:focus {
  outline: 3px solid #005fcc;
  outline-offset: 2px;
}
```

## Cross-References

- **frontend:design-system-patterns** -- accessible headless components, ARIA defaults
- **frontend:form-patterns** -- accessible forms, labels, error messages, aria-describedby
- **frontend:web-animation-patterns** -- reduced-motion preferences, prefers-reduced-motion
