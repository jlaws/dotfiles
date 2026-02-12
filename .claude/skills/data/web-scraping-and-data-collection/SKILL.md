---
name: web-scraping-and-data-collection
description: Extract web data at scale with Scrapy, Playwright, and respectful crawling patterns
---

# Web Scraping and Data Collection

Scrapy spiders, Playwright browser automation, httpx async extraction, rate limiting, pagination, and data cleaning pipelines.

## Scraping Tool Decision Table

| Site Type | Tool | Why |
|-----------|------|-----|
| Static HTML, many pages | **Scrapy** | Async engine, built-in pipelines, middleware, autothrottle |
| JS-rendered SPA | **Playwright** | Full browser, handles dynamic content, stealth plugins |
| REST/GraphQL API | **httpx** | Async HTTP client, no browser overhead |
| Simple one-off extraction | **BeautifulSoup** + httpx | Lightweight, no framework overhead |
| Login-gated + JS-heavy | Playwright + stealth | Cookie handling, human-like interaction |
| Large-scale distributed | Scrapy + Scrapy-Redis | Distributed queue, deduplication, resume |

## Scrapy Spider Patterns

### CrawlSpider with Rules

```python
import scrapy
from scrapy.linkextractors import LinkExtractor
from scrapy.spiders import CrawlSpider, Rule

class ProductSpider(CrawlSpider):
    name = "products"
    allowed_domains = ["example.com"]
    start_urls = ["https://example.com/catalog"]

    rules = (
        Rule(LinkExtractor(allow=r"/category/"), follow=True),
        Rule(LinkExtractor(allow=r"/product/\d+"), callback="parse_product"),
    )

    def parse_product(self, response):
        yield {
            "url": response.url,
            "title": response.css("h1.product-title::text").get("").strip(),
            "price": response.css("span.price::text").re_first(r"[\d.]+"),
            "description": response.css("div.description p::text").getall(),
            "sku": response.xpath('//meta[@itemprop="sku"]/@content').get(),
        }
```

### Item Pipeline

```python
import json
from itemadapter import ItemAdapter

class CleaningPipeline:
    def process_item(self, item, spider):
        adapter = ItemAdapter(item)
        for field in adapter.field_names():
            val = adapter.get(field)
            if isinstance(val, str):
                adapter[field] = val.strip()
        if not adapter.get("title"):
            raise scrapy.exceptions.DropItem(f"Missing title: {item}")
        return item

class JsonLinesPipeline:
    def open_spider(self, spider):
        self.file = open(f"{spider.name}.jsonl", "w")

    def close_spider(self, spider):
        self.file.close()

    def process_item(self, item, spider):
        self.file.write(json.dumps(ItemAdapter(item).asdict()) + "\n")
        return item
```

### Rate Limiting Settings

```python
# settings.py
AUTOTHROTTLE_ENABLED = True
AUTOTHROTTLE_START_DELAY = 2
AUTOTHROTTLE_MAX_DELAY = 30
AUTOTHROTTLE_TARGET_CONCURRENCY = 2.0
DOWNLOAD_DELAY = 1
CONCURRENT_REQUESTS_PER_DOMAIN = 4
ROBOTSTXT_OBEY = True
USER_AGENT = "MyBot/1.0 (+https://example.com/bot)"
ITEM_PIPELINES = {
    "myproject.pipelines.CleaningPipeline": 100,
    "myproject.pipelines.JsonLinesPipeline": 200,
}
```

## Playwright for JS-Rendered Pages

```python
from playwright.async_api import async_playwright

async def scrape_spa(url: str) -> list[dict]:
    async with async_playwright() as p:
        browser = await p.chromium.launch(
            headless=True,
            args=["--disable-blink-features=AutomationControlled"],
        )
        context = await browser.new_context(
            user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) ...",
            viewport={"width": 1280, "height": 720},
        )
        page = await context.new_page()
        # Block images/fonts for speed
        await page.route("**/*.{png,jpg,gif,svg,woff2}", lambda r: r.abort())
        await page.goto(url, wait_until="networkidle")
        await page.wait_for_selector("div.results", timeout=10000)

        items = await page.evaluate("""
            () => Array.from(document.querySelectorAll('.result-card')).map(el => ({
                title: el.querySelector('h3')?.textContent?.trim(),
                link: el.querySelector('a')?.href,
            }))
        """)
        await browser.close()
        return items
```

### Infinite Scroll Handling

```python
async def scrape_infinite_scroll(page, max_items: int = 500) -> list[dict]:
    items, prev_count, stale_rounds = [], 0, 0
    while len(items) < max_items and stale_rounds < 3:
        await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
        await page.wait_for_timeout(2000)
        cards = await page.query_selector_all(".item-card")
        items = [await extract_card(c) for c in cards]
        stale_rounds = stale_rounds + 1 if len(items) == prev_count else 0
        prev_count = len(items)
    return items[:max_items]
```

## httpx Async Client for APIs

```python
import httpx
import asyncio

async def fetch_paginated_api(base_url: str, max_pages: int = 50) -> list[dict]:
    results = []
    async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
        cursor = None
        for _ in range(max_pages):
            params = {"limit": 100}
            if cursor:
                params["cursor"] = cursor
            resp = await client.get(f"{base_url}/items", params=params)
            resp.raise_for_status()
            data = resp.json()
            results.extend(data["items"])
            cursor = data.get("next_cursor")
            if not cursor:
                break
            await asyncio.sleep(0.5)  # politeness delay
    return results
```

## Pagination Patterns

### Next-Page Link (Scrapy)

```python
def parse_listing(self, response):
    for card in response.css("div.item"):
        yield {"title": card.css("h3::text").get()}
    next_page = response.css("a.next-page::attr(href)").get()
    if next_page:
        yield scrapy.Request(response.urljoin(next_page), callback=self.parse_listing)
```

## Data Cleaning Pipeline

```python
import re
from bs4 import BeautifulSoup

def clean_html_text(raw_html: str) -> str:
    soup = BeautifulSoup(raw_html, "html.parser")
    for tag in soup(["script", "style"]):
        tag.decompose()
    return re.sub(r"\s+", " ", soup.get_text(separator=" ", strip=True)).strip()

def extract_price(text: str) -> float | None:
    match = re.search(r"[\d,]+\.?\d*", text.replace(",", ""))
    return float(match.group()) if match else None

def normalize_record(raw: dict) -> dict:
    return {
        "title": raw.get("title", "").strip()[:500],
        "price": extract_price(raw.get("price_text", "")),
        "description": clean_html_text(raw.get("desc_html", "")),
    }
```

## Legal and Ethical Considerations

| Aspect | Guidance |
|--------|----------|
| robots.txt | Always obey; `ROBOTSTXT_OBEY = True` in Scrapy |
| Rate limiting | 1-2 req/sec per domain minimum; use AutoThrottle |
| Terms of Service | Check ToS for scraping prohibitions before starting |
| Personal data | GDPR/CCPA apply; avoid scraping PII without legal basis |
| Caching | Cache responses locally to avoid redundant requests |
| Identification | Set a descriptive `User-Agent` with contact URL |

## Gotchas

- **Scrapy `response.css()` returns `SelectorList`**: use `.get()` for first match, `.getall()` for all. `.extract()` is deprecated.
- **`robots.txt` caching**: Scrapy caches per domain. Stale rules apply on long crawls; consider periodic re-fetch.
- **Playwright `wait_until="networkidle"`** can hang on heavy sites. Use `"domcontentloaded"` + explicit `wait_for_selector`.
- **Memory on large crawls**: enable `JOBDIR` for pause/resume, `CLOSESPIDER_ITEMCOUNT` for bounded runs.
- **Cloudflare / anti-bot**: `playwright-stealth` patches navigator properties. Rotate user agents and use proxies.
- **httpx vs aiohttp**: httpx has simpler API and HTTP/2; aiohttp is faster for high-concurrency websockets. Default to httpx.
- **Selector fragility**: prefer `data-testid` or semantic attributes over nested CSS. XPath `contains(@class, "x")` is more resilient.
- **Character encoding**: Scrapy auto-detects. For httpx, check `resp.encoding`; use `resp.apparent_encoding` as fallback.
