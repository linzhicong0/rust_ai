# rust_ai

A modular AI agent framework built in Rust. Provides traits for providers, tools, memory, and pipelines — with built-in implementations for OpenAI, Anthropic, and Ollama.

## Architecture

The framework is organized as a Cargo workspace with the following crates:

| Crate | Purpose |
|---|---|
| `ai-core` | Core traits (`Provider`, `Tool`, `Memory`, `Embedder`, `Guardrail`), types, config, error enums |
| `ai-provider-openai` | OpenAI provider implementation via reqwest |
| `ai-provider-anthropic` | Anthropic provider implementation via reqwest |
| `ai-provider-ollama` | Ollama/local provider implementation via reqwest |
| `ai-agent` | Agent system with ReAct loop, type-state builder, delegation, planning |
| `ai-tools` | Built-in tools: file I/O, HTTP fetch, web search, shell, code execution |
| `ai-tools-macros` | Procedural macros (`#[tool]`, `#[agent]`) |
| `ai-memory` | Memory backends: in-memory, SQL (sqlx), Redis |
| `ai-rag` | RAG pipeline: chunking, ingestion, retrieval, vector store |
| `ai-pipeline` | Workflow orchestration: sequential, parallel, conditional, DAG pipelines |
| `ai-observability` | Structured logging, distributed tracing, Prometheus metrics |
| `ai-cache` | Response cache (moka) and semantic cache (vector similarity) |
| `ai-server` | REST API server (Axum) with auth and rate limiting |
| `ai-cli` | CLI tool for scaffolding, running, and deploying |

## Build Phases

```
Phase 1 (core):
  ai-core → ai-provider-* / ai-tools / ai-tools-macros → ai-agent → ai-pipeline

Phase 2 (infrastructure):
  ai-memory / ai-cache / ai-observability → ai-rag → ai-server

Phase 3:
  ai-cli (depends on everything)
```

## Quick Start

```bash
# Build the entire workspace
cargo build

# Run the CLI
cargo run -p ai-cli -- --help

# Run an example
cargo run --example basic_agent
```

## Key Dependencies

- **Async**: tokio, futures, async-trait
- **HTTP**: reqwest (client), axum (server)
- **Serialization**: serde, serde_json
- **Error handling**: thiserror, anyhow
- **Tracing**: tracing, tracing-subscriber, opentelemetry
- **Database**: sqlx (async, compile-time checked)
- **Caching**: moka (concurrent async cache)
- **CLI**: clap
- **Templates**: tera

## License

MIT
