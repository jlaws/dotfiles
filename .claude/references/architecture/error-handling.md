# Error Handling Catalog

Typed error hierarchy for API client SDKs with status-code-to-exception mapping.

## Python Error Hierarchy

```python
class APIError(Exception):
    """Base SDK error."""
    def __init__(self, status: int, message: str, code: str | None = None):
        self.status = status
        self.message = message
        self.code = code
        super().__init__(f"[{status}] {code or 'unknown'}: {message}")

    @classmethod
    def from_response(cls, resp: httpx.Response) -> "APIError":
        try:
            body = resp.json()
            msg = body.get("message", resp.text)
            code = body.get("code")
        except Exception:
            msg = resp.text
            code = None

        status_map = {
            401: AuthenticationError,
            403: PermissionError_,
            404: NotFoundError,
            422: ValidationError_,
            429: RateLimitError,
        }
        klass = status_map.get(resp.status_code, cls)
        return klass(resp.status_code, msg, code)

class AuthenticationError(APIError): pass
class PermissionError_(APIError): pass
class NotFoundError(APIError): pass
class ValidationError_(APIError): pass
class RateLimitError(APIError):
    @property
    def retry_after(self) -> float | None:
        # Parsed from response headers during construction
        return getattr(self, "_retry_after", None)
```

## Status Code Mapping

| Status | Exception Class | User Action |
|--------|----------------|-------------|
| 401 | `AuthenticationError` | Check API key / refresh token |
| 403 | `PermissionError_` | Check scopes / permissions |
| 404 | `NotFoundError` | Verify resource ID |
| 422 | `ValidationError_` | Fix request payload |
| 429 | `RateLimitError` | Backoff, check `retry_after` |
| 5xx | `APIError` (base) | Retry with backoff |

## Design Principles

- **Typed exceptions** -- users catch specific errors: `except NotFoundError` not `except Exception`
- **Preserve API context** -- include status code, error code, and message from API response
- **Factory pattern** -- `from_response()` maps status codes to subclasses automatically
- **Never swallow** -- always surface errors to SDK consumers; log internally only if adding value
- **Rate limit metadata** -- `RateLimitError` should expose `retry_after` for caller-driven backoff
