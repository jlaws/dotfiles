---
name: browser-extension-development
description: "Use when building browser extensions, working with Manifest V3, content scripts, service workers, or cross-browser WebExtension APIs. Covers messaging, storage, permissions, and popup/options pages."
---

# Browser Extension Development

## Manifest V3 Structure

```json
{
  "manifest_version": 3,
  "name": "My Extension",
  "version": "1.0.0",
  "permissions": ["storage", "alarms"],
  "host_permissions": ["https://api.example.com/*"],
  "background": {
    "service_worker": "background.js",
    "type": "module"
  },
  "content_scripts": [{
    "matches": ["https://example.com/*"],
    "js": ["content.js"],
    "run_at": "document_idle"
  }],
  "action": {
    "default_popup": "popup.html",
    "default_icon": { "16": "icon16.png", "48": "icon48.png", "128": "icon128.png" }
  },
  "options_page": "options.html"
}
```

**V3 breaking changes from V2**: `background.scripts` replaced by `service_worker`, `browser_action`/`page_action` unified to `action`, remote code execution banned, `host_permissions` separated from `permissions`.

## Architecture Overview

| Component | Context | DOM Access | Lifecycle |
|-----------|---------|------------|-----------|
| **Content script** | Page (isolated world) | Yes | Per-page load |
| **Service worker** | Extension background | No | Event-driven, terminates after ~30s idle |
| **Popup** | Extension UI | Own DOM only | Open/close with click |
| **Options page** | Extension UI | Own DOM only | Tab lifecycle |

## Content Scripts

```typescript
// content.js -- runs in isolated world (separate JS context, shared DOM)
const data = document.querySelector("#target")?.textContent;

// Send to background
chrome.runtime.sendMessage({ type: "SCRAPED", data }, (response) => {
  console.log("Background replied:", response);
});

// Inject into page context (when you need access to page JS globals)
const script = document.createElement("script");
script.src = chrome.runtime.getURL("injected.js");
document.head.appendChild(script);
// Requires "web_accessible_resources" in manifest
```

## Service Worker (Background)

```typescript
// background.js -- event-driven, NO persistent state in memory
chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {
  if (msg.type === "SCRAPED") {
    // Persist state to storage, not variables (worker terminates)
    chrome.storage.local.set({ lastData: msg.data });
    sendResponse({ status: "ok" });
  }
  return true; // keep sendResponse channel open for async
});

// Use alarms instead of setInterval (worker dies before interval fires)
chrome.alarms.create("refresh", { periodInMinutes: 5 });
chrome.alarms.onAlarm.addListener((alarm) => {
  if (alarm.name === "refresh") fetchLatestData();
});
```

**Offscreen documents**: For DOM APIs unavailable in service workers (audio playback, clipboard, DOMParser), create with `chrome.offscreen.createDocument()`.

## Storage APIs

| API | Capacity | Sync | Use Case |
|-----|----------|------|----------|
| `chrome.storage.local` | ~10MB | No | Large data, user preferences |
| `chrome.storage.sync` | 100KB (8KB/item) | Cross-device | Settings synced across browsers |
| `chrome.storage.session` | ~10MB | No | Ephemeral data, cleared on restart |

```typescript
// Always async
const { theme } = await chrome.storage.sync.get({ theme: "dark" }); // with default
await chrome.storage.local.set({ cache: largeObject });

// React to changes from any context
chrome.storage.onChanged.addListener((changes, area) => {
  if (area === "sync" && changes.theme) {
    applyTheme(changes.theme.newValue);
  }
});
```

## Message Passing

**One-shot** (simple request/response):
```typescript
// From content script or popup
const response = await chrome.runtime.sendMessage({ type: "GET_DATA" });

// From background to specific tab
chrome.tabs.sendMessage(tabId, { type: "UPDATE_UI", data });
```

**Long-lived connections** (streaming, persistent channel):
```typescript
// Content script
const port = chrome.runtime.connect({ name: "stream" });
port.postMessage({ subscribe: "updates" });
port.onMessage.addListener((msg) => updateUI(msg));
port.onDisconnect.addListener(() => reconnect());
```

## Cross-Browser Compatibility

```typescript
// Use WebExtension polyfill for Promise-based API + browser namespace
// npm install webextension-polyfill
import browser from "webextension-polyfill";

// browser.* returns Promises (Chrome chrome.* uses callbacks)
const tabs = await browser.tabs.query({ active: true, currentWindow: true });
```

**Key differences**: Firefox uses `browser.*` (Promise-based) natively. Chrome uses `chrome.*` (callback-based, Promises added in MV3). Safari supports WebExtension API with `browser.*`. The polyfill normalizes all three.

## Gotchas

- **Service worker termination**: Dies after ~30s idle. Never store state in global variables -- use `chrome.storage`. Use `chrome.alarms` not `setInterval`/`setTimeout` for recurring tasks
- **CSP restrictions**: MV3 bans inline scripts in extension pages, `eval()`, and remote code. Bundle everything
- **Host permissions in V3**: Moved out of `permissions` into `host_permissions` -- users can selectively revoke per-site access
- **CORS in content scripts**: Requests from content scripts are subject to page's CORS policy. Route through service worker for cross-origin fetches
- **`return true` in onMessage**: Required to keep `sendResponse` channel open for async work; forgetting it causes "message port closed" errors
- **Popup lifecycle**: Popup HTML/JS reloads every time it opens -- persist state in storage, not popup variables
- **`web_accessible_resources`**: Must explicitly declare files accessible to web pages (V3 requires `matches` array to scope access)

## Cross-References

- **languages:js-ts-patterns** -- TypeScript configuration, module patterns, async/await
- **security:auth-implementation-patterns** -- OAuth flows for extensions, token storage
