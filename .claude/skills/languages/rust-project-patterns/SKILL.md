---
name: rust-project-patterns
description: Rust project scaffolding, async patterns, and tooling opinions. Use when creating Rust projects, choosing between project types, or working with async Rust.
---

# Rust Project Patterns

Project scaffolding, async patterns, and opinionated tooling for Rust.

## Style Guide

Source: Rust Style Guide. Only rules linters/formatters cannot enforce.

### Naming
- Types/traits: `UpperCamelCase`
- Functions/methods/locals: `snake_case`
- Constants/statics: `SCREAMING_SNAKE_CASE`
- Modules: `snake_case`
- Lifetimes: short lowercase (`'a`, `'de`), descriptive when multiple
- Type parameters: single uppercase (`T`, `E`) or descriptive `CamelCase` (`Item`, `Error`)
- Avoid abbreviations: `connection` not `conn`, except well-known (`ctx`, `cfg`)

### Practices
- `///` line doc comments; `//!` for module/crate-level only
- Doc comments before attributes, not after
- Single `#[derive(...)]` — don't split into multiple
- Comments: complete sentences, capital letter, period

## Tooling Defaults

| Concern | Use |
|---------|-----|
| Error handling (apps) | `anyhow` |
| Error handling (libs) | `thiserror` |
| Serialization | `serde` + `serde_json` |
| CLI | `clap` (derive) |
| HTTP client | `reqwest` |
| Web framework | `axum` (prefer over actix-web for new projects) |
| Async runtime | `tokio` (full features) |
| Logging | `tracing` + `tracing-subscriber` |
| Benchmarking | `criterion` |

## Project Type Selection

| Type | When to Use |
|------|-------------|
| **Binary** | CLI tools, applications, services |
| **Library** | Reusable crates |
| **Workspace** | Multi-crate projects, monorepos |
| **Web API** | Axum services, REST APIs |

## Binary Project

```
src/main.rs, cli.rs, config.rs, error.rs, lib.rs
src/commands/mod.rs, init.rs, run.rs
tests/integration_test.rs
benches/benchmark.rs
```

### Cargo.toml essentials
```toml
[package]
name = "project-name"
version = "0.1.0"
edition = "2021"
rust-version = "1.75"

[dependencies]
clap = { version = "4.5", features = ["derive"] }
tokio = { version = "1", features = ["full"] }
anyhow = "1.0"
serde = { version = "1.0", features = ["derive"] }
tracing = "0.1"
tracing-subscriber = "0.3"

[dev-dependencies]
criterion = "0.5"

[[bench]]
name = "benchmark"
harness = false

[profile.release]
opt-level = 3
lto = true
codegen-units = 1
```

## Workspace Structure
```toml
[workspace]
members = ["crates/api", "crates/core", "crates/cli"]
resolver = "2"

[workspace.package]
version = "0.1.0"
edition = "2021"

[workspace.dependencies]
tokio = { version = "1", features = ["full"] }
serde = { version = "1.0", features = ["derive"] }
```

## Web API (Axum)
```rust
use axum::{Router, routing::get};
use tower_http::cors::CorsLayer;

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt::init();
    let app = Router::new()
        .route("/health", get(|| async { "ok" }))
        .layer(CorsLayer::permissive());

    let listener = tokio::net::TcpListener::bind("0.0.0.0:3000").await.unwrap();
    axum::serve(listener, app).await.unwrap();
}
```

## Async Patterns

### Concurrency with JoinSet
```rust
use tokio::task::JoinSet;

async fn fetch_all(urls: Vec<String>) -> Vec<String> {
    let mut set = JoinSet::new();
    for url in urls {
        set.spawn(async move { fetch(&url).await });
    }
    let mut results = Vec::new();
    while let Some(res) = set.join_next().await {
        if let Ok(Ok(data)) = res { results.push(data); }
    }
    results
}
```

### Bounded concurrency with streams
```rust
use futures::stream::{self, StreamExt};

async fn fetch_bounded(urls: Vec<String>, limit: usize) -> Vec<String> {
    stream::iter(urls)
        .map(|url| async move { fetch(&url).await })
        .buffer_unordered(limit)
        .filter_map(|r| async { r.ok() })
        .collect()
        .await
}
```

### Channel selection guide

| Channel | When |
|---------|------|
| `mpsc` | Multiple producers, single consumer (most common) |
| `broadcast` | Multiple consumers all get every message |
| `oneshot` | Single value response (request/reply) |
| `watch` | Latest-value broadcast (config changes) |

### Graceful shutdown
```rust
use tokio_util::sync::CancellationToken;

let token = CancellationToken::new();
let t = token.clone();

tokio::spawn(async move {
    loop {
        tokio::select! {
            _ = t.cancelled() => break,
            _ = do_work() => {}
        }
    }
});

tokio::signal::ctrl_c().await?;
token.cancel();
```

### Async gotchas
- **Never `std::thread::sleep` in async** -- blocks the entire runtime
- **Don't hold `MutexGuard` across `.await`** -- causes deadlocks
- **Spawned futures must be `Send`** -- no `Rc`, non-Send types
- **Use `tokio::select!`** for racing futures, not manual polling
- **`async_trait`** still needed for trait objects (RPITIT stabilized but limited)

## Dev Tool Config

**rustfmt.toml**: `edition = "2021"`, `max_width = 100`, `use_small_heuristics = "Max"`

**clippy.toml**: `cognitive-complexity-threshold = 30`

```makefile
build: ; cargo build
test:  ; cargo test
lint:  ; cargo clippy -- -D warnings
fmt:   ; cargo fmt --check
bench: ; cargo bench
```
