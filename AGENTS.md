# Project Guidelines

## Start Here
- Read [README.md](README.md) for the workspace overview and quick-start commands.
- Read [DESIGN.md](DESIGN.md) for crate boundaries, dependency direction, and architecture rationale.
- Read [REQUIREMENTS.md](REQUIREMENTS.md) when a change affects framework behavior, API guarantees, or feature scope.
- Use [.github/agents/rust-expert.agent.md](.github/agents/rust-expert.agent.md) when the task needs a Rust-specialized subagent.

## Workspace Shape
- This repo is a Cargo workspace. The authoritative crate list lives in [Cargo.toml](Cargo.toml).
- Most work should stay crate-local under `crates/<name>` and validate only the touched crate first.
- Core dependency direction is: `ai-core` traits and types -> provider/tool crates -> `ai-agent` -> `ai-pipeline`; infrastructure crates such as `ai-memory`, `ai-cache`, and `ai-observability` support higher-level crates like `ai-rag`, `ai-server`, and `ai-cli`.
- Cross-crate integration tests live in `tests/`. Runnable examples live in `examples/` and some crates also have `examples/` folders.
- Do not edit generated artifacts under `target/`.

## Build And Test
- Run Cargo commands from the repository root.
- Prefer the narrowest validating command first: `cargo check -p <crate>`, `cargo test -p <crate>`, or `cargo clippy -p <crate> --all-targets -- -D warnings`.
- Use workspace-level commands only when a change crosses crate boundaries or the narrower check passes and broader validation is needed.
- Common focused validations in this repo:
  - `cargo test -p ai-cache`
  - `cargo test -p ai-memory -p ai-agent`
  - `cargo test -p ai-pipeline`
  - `cargo test -p ai-agent -p ai-pipeline -p ai-observability`
- Useful smoke checks:
  - `cargo build`
  - `cargo test`
  - `cargo run -p ai-cli -- --help`
  - `cargo run --example basic_agent`

## Rust Conventions
- Prefer extending existing traits and typed abstractions in `ai-core` over adding ad hoc cross-crate coupling.
- Keep public APIs small and idiomatic. Prefer `Result`-based errors and avoid `unwrap`/`expect` in library code unless the invariant is fully internal.
- Add or update tests with behavior changes. Keep unit tests close to the touched code; use `tests/` for cross-crate integration behavior.
- Preserve the existing async and HTTP stack unless the task requires otherwise: `tokio`, `async-trait`, `reqwest` with `rustls-tls`, `axum`, and `serde`.
- Configuration is layered through `ai_framework.toml` or `ai_framework.yaml`, `.env`, and `AI_*` environment variables with `__` separators for nesting. Keep that shape intact when changing config behavior.

## Documentation Policy
- Link to existing docs instead of copying them into code comments or new instruction files.
- If implementation diverges from [README.md](README.md), [DESIGN.md](DESIGN.md), or [REQUIREMENTS.md](REQUIREMENTS.md), update the relevant document in the same change when practical.