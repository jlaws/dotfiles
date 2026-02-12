---
name: go-concurrency-patterns
description: Go concurrency decisions and non-obvious patterns. Use when building concurrent Go applications or debugging race conditions.
---

# Go Concurrency Patterns

Decision frameworks and gotchas for Go concurrency.

## Style Guide

Source: Google Go Style Guide. Only rules linters/formatters cannot enforce.

### Naming
- Names must not repeat context: `db.UserStore` not `db.DBUserStore`
- Shorter names preferred when context is clear: `i` in a loop, `r` for reader in small scope
- Exported names carry the package name — `http.Server` not `http.HTTPServer`
- Getters: `Owner()` not `GetOwner()`; setters: `SetOwner()`
- Interfaces: single-method uses method name + `er` suffix: `Reader`, `Writer`
- Package names: single lowercase word, no underscores, no `mixedCaps`
- Avoid stuttering: `user.User` is fine, `user.UserService` is not
- Constants use `MixedCaps` (not `SCREAMING_SNAKE`)

### Practices
- Comments explain *why* not *what*; don't restate obvious code
- Simplicity: fewest concepts needed; standard library first, then internal, then external
- No fixed line length — refactor long functions instead of splitting lines

## Primitive Selection

| Need | Use |
|------|-----|
| Fan-out work, collect results | `errgroup.Group` (with `SetLimit` for bounded) |
| Background workers with lifecycle | `context.Context` + `sync.WaitGroup` |
| Communicate between goroutines | channels |
| Protect shared state (write-heavy) | `sync.Mutex` |
| Protect shared state (read-heavy) | `sync.RWMutex` or `sync.Map` |
| Rate limit concurrency | `semaphore.Weighted` or buffered channel |

## errgroup (preferred for most concurrent work)

```go
g, ctx := errgroup.WithContext(ctx)
g.SetLimit(10) // bounded concurrency

for _, url := range urls {
    url := url
    g.Go(func() error {
        return fetch(ctx, url)
    })
}

if err := g.Wait(); err != nil {
    return err // first error cancels all others via ctx
}
```

## Worker Pool

```go
func WorkerPool(ctx context.Context, workers int, jobs <-chan Job) <-chan Result {
    results := make(chan Result)
    var wg sync.WaitGroup

    for i := 0; i < workers; i++ {
        wg.Add(1)
        go func() {
            defer wg.Done()
            for job := range jobs {
                select {
                case <-ctx.Done(): return
                case results <- process(job):
                }
            }
        }()
    }

    go func() { wg.Wait(); close(results) }()
    return results
}
```

## Graceful Shutdown Pattern

```go
ctx, cancel := context.WithCancel(context.Background())
sigCh := make(chan os.Signal, 1)
signal.Notify(sigCh, syscall.SIGINT, syscall.SIGTERM)

go func() { <-sigCh; cancel() }()

// All workers check ctx.Done()
```

## Channel Gotchas

- **Close from sender only** -- closing from receiver causes panic
- **Buffer channels when count is known** -- avoids goroutine leak on early return
- **`select` with `default`** for non-blocking send/receive
- **Priority select** -- nest two selects (check high-priority channel in default case)

## Race Detection

```bash
go test -race ./...    # always run in CI
go build -race .       # for development builds
```

## sync.Map vs Regular Map+Mutex

- `sync.Map`: optimized for keys written once, read many (cache-like)
- Regular `map` + `RWMutex`: better for write-heavy or small maps
- Sharded map: best for high-contention write-heavy workloads

## Key Don'ts

- **Don't leak goroutines** -- every goroutine needs an exit path (context, done channel)
- **Don't `time.Sleep` for synchronization** -- use proper primitives
- **Don't share memory without synchronization** -- even "read-only" maps aren't safe during concurrent writes
- **Don't ignore `ctx.Done()`** -- check it in long-running loops
- **Don't forget loop variable capture** -- `url := url` before goroutine (fixed in Go 1.22+ but be explicit)
