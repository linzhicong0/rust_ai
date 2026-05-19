# AI Framework — Rust Technical Design

---

## 1. Crate Selection

| Domain | Crate | Purpose |
|--------|-------|---------|
| Async Runtime | `tokio` | Async I/O, task spawning, channels, timers |
| HTTP Client | `reqwest` | Call LLM provider APIs (OpenAI, Anthropic, Google) |
| HTTP Server | `axum` | Expose framework as REST API |
| Serialization | `serde` + `serde_json` | JSON serialize/deserialize everywhere |
| Error Handling | `thiserror` | Typed, ergonomic error enums for the library |
| Logging/Tracing | `tracing` + `tracing-subscriber` | Structured logging, spans, distributed tracing |
| Configuration | `config` + `dotenvy` | Layered config: defaults < file < env < code |
| CLI | `clap` | CLI tools: scaffold, run, test, deploy |
| Template | `tera` | Prompt templates with `{{var}}`, conditionals, inheritance |
| Schema | `schemars` + `jsonschema` | JSON Schema generation and validation |
| Caching | `moka` | Concurrent async LRU/TTL cache |
| Streams | `futures` + `tokio-stream` | Stream trait, async iterators for streaming responses |
| Concurrency | `tokio::sync` | Semaphore, Mutex, RwLock, broadcast, mpsc |
| Database | `sqlx` | PostgreSQL, MySQL, SQLite (async, compile-time checked) |
| Vector Search | `qdrant-client` | Vector database client for RAG |
| Embeddings (local) | `ort` + `ndarray` | ONNX Runtime for local embedding inference |
| Process Isolation | `wasmtime` | WASM sandbox for untrusted tool execution |
| Async Trait | `async-trait` | Async methods in trait definitions |
| Builder | `derive_builder` or typed-builder | Builder pattern for config structs |
| Macros | `proc-macro2` + `quote` + `syn` | Custom derive macros for Tool, Agent |
| Container | `docker-api` or Bollard | Docker integration for sandboxed execution |
| Observability | `opentelemetry` + `tracing-opentelemetry` | OpenTelemetry export (traces + metrics) |
| Testing | `tokio::test`, `proptest` | Async tests + property-based testing |

---

## 2. Core Trait Hierarchy

The framework is built around a small set of core traits. Every major component is a trait with multiple implementations.

```
Provider                    Tool                     Memory
├── OpenAiProvider          ├── FnTool (closure)     ├── InMemoryMemory
├── AnthropicProvider       ├── FileReadTool         ├── SqlMemory
├── GoogleProvider          ├── HttpFetchTool        ├── VectorMemory
└── OllamaProvider          └── ShellExecTool        └── RedisMemory

Agent                       Pipeline                 Embedder
├── Agent<Provider, Mem>    ├── SequentialPipeline   ├── OpenAiEmbedder
└── (generic over           ├── ParallelPipeline     ├── CohereEmbedder
    Provider + Memory)      └── DagPipeline          └── LocalEmbedder (ort)
```

### 2.1 Provider Trait (LLM Access)

```rust
use async_trait::async_trait;
use futures::stream::BoxStream;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    pub model: String,
    pub temperature: Option<f64>,
    pub max_tokens: Option<u32>,
    pub top_p: Option<f64>,
    pub frequency_penalty: Option<f64>,
    pub presence_penalty: Option<f64>,
    pub stop_sequences: Option<Vec<String>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub role: Role,
    pub content: Content,
}

pub enum Role {
    System,
    User,
    Assistant,
    Tool,
}

pub enum Content {
    Text(String),
    MultiPart(Vec<ContentPart>),  // for vision: text + image
}

pub enum ContentPart {
    Text(String),
    Image { url: String, media_type: String },
}

#[derive(Debug)]
pub struct Usage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
}

#[derive(Debug)]
pub struct CompletionResponse {
    pub content: String,
    pub tool_calls: Vec<ToolCall>,
    pub usage: Usage,
    pub finish_reason: FinishReason,
}

#[derive(Debug)]
pub struct StreamChunk {
    pub delta: Option<String>,
    pub tool_call_delta: Option<ToolCallDelta>,
    pub finish_reason: Option<FinishReason>,
    pub usage: Option<Usage>,
}

#[async_trait]
pub trait Provider: Send + Sync {
    async fn complete(
        &self,
        messages: Vec<Message>,
        config: &ModelConfig,
        tools: &[ToolDescriptor],
    ) -> Result<CompletionResponse, ProviderError>;

    fn stream(
        &self,
        messages: Vec<Message>,
        config: &ModelConfig,
        tools: &[ToolDescriptor],
    ) -> BoxStream<'static, Result<StreamChunk, ProviderError>>;

    async fn embed(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>, ProviderError>;

    fn name(&self) -> &str;
}
```

**Rust features used:**
- `async_trait` — async methods in traits (until RPITIT stabilizes)
- `BoxStream` — trait-object streaming return (avoids naming the concrete stream type)
- `enum` — sum types for `Content`, `Role`, `FinishReason`
- `#[derive(Serialize, Deserialize)]` — serde derives for config types
- `Result<T, E>` — typed error propagation

**Reqwest usage in Provider implementations:**

```rust
pub struct OpenAiProvider {
    client: reqwest::Client,       // connection pooling built-in
    api_key: String,
    base_url: String,
}

#[async_trait]
impl Provider for OpenAiProvider {
    async fn complete(
        &self,
        messages: Vec<Message>,
        config: &ModelConfig,
        tools: &[ToolDescriptor],
    ) -> Result<CompletionResponse, ProviderError> {
        let body = self.build_request_body(messages, config, tools);

        let response = self.client
            .post(format!("{}/chat/completions", self.base_url))
            .header("Authorization", format!("Bearer {}", self.api_key))
            .json(&body)
            .send()
            .await
            .map_err(ProviderError::Http)?;

        let status = response.status();
        let body = response.text().await.map_err(ProviderError::Http)?;

        if !status.is_success() {
            return Err(ProviderError::Api { status, body });
        }

        serde_json::from_str(&body).map_err(ProviderError::Deserialize)
    }

    fn stream(
        &self,
        messages: Vec<Message>,
        config: &ModelConfig,
        tools: &[ToolDescriptor],
    ) -> BoxStream<'static, Result<StreamChunk, ProviderError>> {
        // reqwest streaming: response.bytes_stream()
        // converted to BoxStream via try_stream! macro from futures
        todo!()
    }
}
```

### 2.2 Tool Trait

```rust
use schemars::JsonSchema;
use serde_json::Value;

pub struct ToolDescriptor {
    pub name: String,
    pub description: String,
    pub input_schema: Value,  // JSON Schema
}

pub struct ToolOutput {
    pub content: String,
    pub is_error: bool,
}

#[async_trait]
pub trait Tool: Send + Sync {
    fn descriptor(&self) -> ToolDescriptor;
    async fn execute(&self, input: Value) -> Result<ToolOutput, ToolError>;
}
```

**Ergonomic macro for defining tools:**

```rust
// Usage:
#[tool(description = "Search the web for information")]
async fn web_search(
    #[tool(desc = "Search query")] query: String,
    #[tool(desc = "Max results")] max_results: Option<u32>,
) -> Result<String, ToolError> {
    // implementation
}

// The macro generates:
// 1. JSON Schema from parameter types (via schemars)
// 2. ToolDescriptor with name, description, schema
// 3. Tool trait implementation
// 4. Input deserialization from Value
```

### 2.3 Memory Trait

```rust
#[derive(Debug, Clone)]
pub struct MemoryEntry {
    pub role: Role,
    pub content: String,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub metadata: HashMap<String, Value>,
}

#[async_trait]
pub trait Memory: Send + Sync {
    async fn add(&self, entry: MemoryEntry) -> Result<(), MemoryError>;
    async fn get(&self, limit: Option<usize>) -> Result<Vec<MemoryEntry>, MemoryError>;
    async fn search(&self, query: &str, limit: usize) -> Result<Vec<MemoryEntry>, MemoryError>;
    async fn clear(&self) -> Result<(), MemoryError>;
}

// In-memory implementation
pub struct InMemoryMemory {
    entries: tokio::sync::RwLock<Vec<MemoryEntry>>,
    max_entries: usize,
}
```

### 2.4 Agent

```rust
pub struct AgentBuilder<P, M>
where
    P: Provider,
    M: Memory,
{
    provider: P,
    memory: M,
    role: Option<String>,
    goal: Option<String>,
    backstory: Option<String>,
    tools: Vec<Box<dyn Tool>>,
    model_config: ModelConfig,
    max_iterations: u32,
}

pub struct Agent<P, M>
where
    P: Provider,
    M: Memory,
{
    inner: Arc<AgentInner<P, M>>,
}

impl<P, M> Agent<P, M>
where
    P: Provider,
    M: Memory,
{
    pub async fn run(&self, input: impl Into<String>) -> Result<AgentOutput, AgentError> {
        // ReAct loop: Reason → Act (tool call) → Observe → repeat
        let mut messages = self.build_messages(input).await?;

        for _ in 0..self.inner.max_iterations {
            let response = self.inner.provider
                .complete(messages.clone(), &self.inner.model_config, &self.tool_descriptors())
                .await?;

            if response.tool_calls.is_empty() {
                // No tool calls — agent is done
                return Ok(AgentOutput { content: response.content });
            }

            // Execute tool calls
            for tool_call in response.tool_calls {
                let tool = self.find_tool(&tool_call.name)?;
                let result = tool.execute(tool_call.arguments).await?;
                messages.push(Message::tool_result(tool_call.id, result.content));
            }
        }

        Err(AgentError::MaxIterationsExceeded)
    }

    pub fn stream(&self, input: impl Into<String>)
        -> BoxStream<'static, Result<AgentEvent, AgentError>>
    {
        // Streaming version of the ReAct loop
        // Yields AgentEvent::Text, AgentEvent::ToolCall, AgentEvent::ToolResult
        todo!()
    }
}
```

**Rust features used:**
- **Generic type parameters** `P: Provider, M: Memory` — swap implementations without touching agent logic
- **`Arc<AgentInner>`** — cheap clone, shared state across async tasks
- **Builder pattern** — `AgentBuilder` enforces required fields at compile time
- **`impl Into<String>`** — accept `&str`, `String`, or any string-like type
- **`Box<dyn Tool>`** — heterogeneous tool collection via trait objects

### 2.5 Pipeline

```rust
pub struct Pipeline {
    steps: Vec<Step>,
}

pub struct Step {
    pub name: String,
    pub kind: StepKind,
}

pub enum StepKind {
    // Single task run by an agent
    Task {
        agent: Arc<dyn Runnable>,
        input_mapper: Box<dyn Fn(&PipelineContext) -> String + Send + Sync>,
        output_key: String,
    },
    // Run multiple steps concurrently
    Parallel(Vec<Step>),
    // Conditional branching
    Conditional {
        condition: Box<dyn Fn(&PipelineContext) -> bool + Send + Sync>,
        then_step: Box<Step>,
        else_step: Option<Box<Step>>,
    },
    // Loop until condition met
    Loop {
        body: Box<Step>,
        condition: Box<dyn Fn(&PipelineContext) -> bool + Send + Sync>,
        max_iterations: u32,
    },
}

pub struct PipelineContext {
    data: HashMap<String, Value>,
}

impl Pipeline {
    pub async fn execute(&self, input: Value) -> Result<PipelineContext, PipelineError> {
        let mut ctx = PipelineContext::new(input);
        for step in &self.steps {
            self.execute_step(step, &mut ctx).await?;
        }
        Ok(ctx)
    }
}
```

---

## 3. Reqwest — HTTP Client Architecture

Reqwest is the backbone for all LLM provider communication. Here is how it maps to framework requirements.

### 3.1 Connection Pooling (REQ-12.5)

Reqwest's `Client` internally uses `hyper` with a connection pool. A single `Client` instance per provider:

```rust
pub struct ProviderPool {
    clients: HashMap<String, reqwest::Client>,  // keyed by provider name
}

impl ProviderPool {
    pub fn client_for(&self, provider: &str) -> &reqwest::Client {
        self.clients.get(provider).expect("unknown provider")
    }
}
```

- One `reqwest::Client` per provider = one connection pool
- Reuse across all requests to the same provider
- Configurable: `pool_max_idle_per_host`, `pool_idle_timeout`, `connect_timeout`

```rust
let client = reqwest::Client::builder()
    .pool_max_idle_per_host(20)
    .pool_idle_timeout(Duration::from_secs(90))
    .connect_timeout(Duration::from_secs(10))
    .timeout(Duration::from_secs(120))
    .build()?;
```

### 3.2 Streaming (REQ-1.4)

```rust
fn stream_completion(
    client: &reqwest::Client,
    url: &str,
    body: &Value,
    api_key: &str,
) -> BoxStream<'static, Result<StreamChunk, ProviderError>> {
    let client = client.clone();
    let url = url.to_string();
    let body = body.clone();
    let api_key = api_key.to_string();

    try_stream! {
        let response = client
            .post(&url)
            .header("Authorization", format!("Bearer {}", api_key))
            .json(&body)
            .send()
            .await?;

        let mut stream = response.bytes_stream();
        while let Some(chunk) = stream.next().await {
            let chunk = chunk?;
            // Parse SSE format: "data: {...}\n\n"
            for line in parse_sse_lines(&chunk) {
                if line == "data: [DONE]" { break; }
                let parsed: StreamChunk = serde_json::from_str(&line)?;
                yield parsed;
            }
        }
    }.boxed()
}
```

### 3.3 Retry and Error Handling (REQ-7.4)

```rust
async fn request_with_retry(
    client: &reqwest::Client,
    request: reqwest::RequestBuilder,
    policy: &RetryPolicy,
) -> Result<reqwest::Response, ProviderError> {
    let mut attempt = 0;
    loop {
        let req = request.try_clone().ok_or(ProviderError::NotRetryable)?;
        let result = req.send().await;

        match result {
            Ok(resp) if resp.status().is_success() => return Ok(resp),
            Ok(resp) if is_retryable(resp.status()) && attempt < policy.max_retries => {
                attempt += 1;
                let delay = policy.backoff.delay(attempt);
                tokio::time::sleep(delay).await;
            }
            Ok(resp) => {
                let status = resp.status();
                let body = resp.text().await.unwrap_or_default();
                return Err(ProviderError::Api { status, body });
            }
            Err(e) if e.is_connect() && attempt < policy.max_retries => {
                attempt += 1;
                let delay = policy.backoff.delay(attempt);
                tokio::time::sleep(delay).await;
            }
            Err(e) => return Err(ProviderError::Http(e)),
        }
    }
}

fn is_retryable(status: reqwest::StatusCode) -> bool {
    matches!(status.as_u16(), 429 | 500 | 502 | 503 | 504)
}
```

### 3.4 Cancellation (REQ-1.4)

```rust
use tokio_util::sync::CancellationToken;

async fn stream_with_cancellation(
    client: &reqwest::Client,
    url: &str,
    body: &Value,
    token: CancellationToken,
) -> Result<String, ProviderError> {
    tokio::select! {
        result = do_stream(client, url, body) => result,
        _ = token.cancelled() => Err(ProviderError::Cancelled),
    }
}
```

---

## 4. Rust Language Features Used

| Rust Feature | Framework Usage | Requirements Served |
|-------------|----------------|-------------------|
| **Traits** | `Provider`, `Tool`, `Memory`, `Embedder`, `Guardrail` — plugin architecture | 15.4, 3.1, 1.1 |
| **Generics** | `Agent<P, M>` — swap provider/memory without touching agent logic | 15.6, 1.1 |
| **Trait objects (`dyn`)** | Heterogeneous collections: `Vec<Box<dyn Tool>>` | 3.1, 15.4 |
| **Enums** | Sum types: `Content`, `Role`, `StepKind`, `AgentEvent`, `FinishReason` | Throughout |
| **Pattern matching** | Match on response types, error variants, tool call decisions | Throughout |
| **`Arc<Mutex<T>>`** | Shared mutable state across async tasks | 14.1, 14.2 |
| **`async/await` + tokio** | All I/O is async: HTTP calls, DB queries, tool execution | Throughout |
| **`async_trait`** | Async methods in traits (bridge until RPITIT stable) | 2.1, 3.1, 5.1 |
| **Closures + `Fn` traits** | Pipeline conditions, mappers, guardrails as closures | 7.1, 9.3 |
| **Procedural macros** | `#[tool]`, `#[agent]` — derive `Tool`/`Agent` boilerplate | 15.1, 3.1 |
| **Declarative macros** | Internal DSLs for pipeline definition | 7.1 |
| **Type-state pattern** | Builder that encodes state in type parameter: `AgentBuilder<NoProvider>` won't compile `.build()` | 15.1, 15.6 |
| **`Drop`** | Automatic cleanup: cancel streams, release connections, flush buffers | 1.4, 12.5 |
| **`std::any::Any`** | Dynamic downcast for pipeline context when types vary per step | 7.1, 7.2 |
| **Associated types** | `type Output` on traits when implementations differ | 15.6 |
| **`impl Trait`** | Return opaque types from functions without naming them | 1.4 |
| **`Cow<str>`** | Borrow or own strings — avoid allocation in prompt templates | 4.1 |
| **`PhantomData`** | Zero-cost type markers on generic structs | 15.6 |

---

## 5. Project Structure

```
rust_ai/
├── Cargo.toml                    # workspace root
├── crates/
│   ├── ai-core/                  # Core traits, types, config
│   │   ├── src/
│   │   │   ├── lib.rs
│   │   │   ├── config.rs         # FrameworkConfig, layered config
│   │   │   ├── types.rs          # Message, Role, Content, Usage, etc.
│   │   │   ├── provider.rs       # Provider trait
│   │   │   ├── tool.rs           # Tool trait + ToolDescriptor
│   │   │   ├── memory.rs         # Memory trait
│   │   │   ├── embedder.rs       # Embedder trait
│   │   │   ├── guardrail.rs      # Guardrail trait
│   │   │   ├── error.rs          # Unified error types
│   │   │   └── template.rs       # Prompt template engine (tera)
│   │   └── Cargo.toml
│   │
│   ├── ai-provider-openai/       # OpenAI provider impl
│   │   ├── src/lib.rs            # OpenAiProvider (uses reqwest)
│   │   └── Cargo.toml
│   │
│   ├── ai-provider-anthropic/    # Anthropic provider impl
│   │   ├── src/lib.rs            # AnthropicProvider (uses reqwest)
│   │   └── Cargo.toml
│   │
│   ├── ai-provider-ollama/       # Ollama/local provider impl
│   │   ├── src/lib.rs
│   │   └── Cargo.toml
│   │
│   ├── ai-agent/                 # Agent system
│   │   ├── src/
│   │   │   ├── lib.rs
│   │   │   ├── agent.rs          # Agent struct + ReAct loop
│   │   │   ├── builder.rs        # AgentBuilder (type-state)
│   │   │   ├── delegate.rs       # Agent delegation
│   │   │   └── planner.rs        # Agent planning
│   │   └── Cargo.toml
│   │
│   ├── ai-tools/                 # Built-in tools
│   │   ├── src/
│   │   │   ├── lib.rs
│   │   │   ├── file.rs           # FileRead, FileWrite
│   │   │   ├── http.rs           # HttpFetch (uses reqwest)
│   │   │   ├── search.rs         # WebSearch
│   │   │   ├── shell.rs          # ShellExec
│   │   │   └── code.rs           # Code execution
│   │   └── Cargo.toml
│   │
│   ├── ai-tools-macros/          # Procedural macros
│   │   ├── src/
│   │   │   └── lib.rs            # #[tool], #[agent] derive macros
│   │   └── Cargo.toml
│   │
│   ├── ai-memory/                # Memory implementations
│   │   ├── src/
│   │   │   ├── lib.rs
│   │   │   ├── in_memory.rs      # InMemoryMemory
│   │   │   ├── sql.rs            # SqlMemory (sqlx)
│   │   │   └── redis.rs          # RedisMemory
│   │   └── Cargo.toml
│   │
│   ├── ai-rag/                   # RAG pipeline
│   │   ├── src/
│   │   │   ├── lib.rs
│   │   │   ├── chunker.rs        # Chunking strategies
│   │   │   ├── ingestion.rs      # Document ingestion
│   │   │   ├── retrieval.rs      # Retrieval strategies
│   │   │   └── vector_store.rs   # VectorStore trait + backends
│   │   └── Cargo.toml
│   │
│   ├── ai-pipeline/              # Workflow orchestration
│   │   ├── src/
│   │   │   ├── lib.rs
│   │   │   ├── pipeline.rs       # Pipeline struct + executor
│   │   │   ├── step.rs           # Step types: Task, Parallel, Conditional, Loop
│   │   │   ├── context.rs        # PipelineContext
│   │   │   └── dag.rs            # DAG resolution for task dependencies
│   │   └── Cargo.toml
│   │
│   ├── ai-observability/         # Logging, tracing, metrics
│   │   ├── src/
│   │   │   ├── lib.rs
│   │   │   ├── logging.rs        # Structured logging setup
│   │   │   ├── tracing.rs        # Distributed tracing spans
│   │   │   └── metrics.rs        # Prometheus metrics
│   │   └── Cargo.toml
│   │
│   ├── ai-cache/                 # Caching layer
│   │   ├── src/
│   │   │   ├── lib.rs
│   │   │   ├── response.rs       # Response cache (moka)
│   │   │   └── semantic.rs       # Semantic cache (vector similarity)
│   │   └── Cargo.toml
│   │
│   ├── ai-server/                # REST API server
│   │   ├── src/
│   │   │   ├── lib.rs
│   │   │   ├── app.rs            # Axum router setup
│   │   │   ├── routes/
│   │   │   │   ├── agents.rs     # POST /agents/{id}/run
│   │   │   │   ├── pipelines.rs  # POST /pipelines/{id}/execute
│   │   │   │   └── health.rs     # GET /health
│   │   │   └── middleware/
│   │   │       ├── auth.rs       # API key auth
│   │   │       └── rate_limit.rs # Rate limiting
│   │   └── Cargo.toml
│   │
│   └── ai-cli/                   # CLI tool
│       ├── src/
│       │   └── main.rs           # clap-based CLI
│       └── Cargo.toml
│
├── examples/                     # Example programs
│   ├── basic_agent.rs
│   ├── multi_agent.rs
│   ├── rag_pipeline.rs
│   └── streaming.rs
│
└── tests/                        # Integration tests
    ├── provider_tests.rs
    ├── agent_tests.rs
    └── pipeline_tests.rs
```

---

## 6. Workspace Cargo.toml

```toml
[workspace]
members = [
    "crates/ai-core",
    "crates/ai-provider-openai",
    "crates/ai-provider-anthropic",
    "crates/ai-provider-ollama",
    "crates/ai-agent",
    "crates/ai-tools",
    "crates/ai-tools-macros",
    "crates/ai-memory",
    "crates/ai-rag",
    "crates/ai-pipeline",
    "crates/ai-observability",
    "crates/ai-cache",
    "crates/ai-server",
    "crates/ai-cli",
]
resolver = "2"

[workspace.dependencies]
# Async
tokio = { version = "1", features = ["full"] }
futures = "0.3"
async-trait = "0.1"
tokio-stream = "0.1"

# HTTP
reqwest = { version = "0.12", features = ["json", "stream", "rustls-tls"] }
axum = { version = "0.8", features = ["macros"] }
tower = "0.5"
tower-http = { version = "0.6", features = ["cors", "trace"] }

# Serialization
serde = { version = "1", features = ["derive"] }
serde_json = "1"

# Error handling
thiserror = "2"
anyhow = "1"

# Logging/Tracing
tracing = "0.1"
tracing-subscriber = { version = "0.3", features = ["json", "env-filter"] }
tracing-opentelemetry = "0.28"
opentelemetry = "0.27"

# Configuration
config = "0.14"
dotenvy = "0.15"

# CLI
clap = { version = "4", features = ["derive"] }

# Templates
tera = "1"

# Schema
schemars = "0.8"
jsonschema = "0.26"

# Caching
moka = { version = "0.12", features = ["future"] }

# Database
sqlx = { version = "0.8", features = ["runtime-tokio", "postgres", "sqlite", "json"] }

# Vector
qdrant-client = "1.12"

# Local inference
ort = { version = "2", features = ["load-dynamic"] }
ndarray = "0.16"

# Time
chrono = { version = "0.4", features = ["serde"] }

# Utils
uuid = { version = "1", features = ["v4", "serde"] }
dashmap = "6"
```

---

## 7. Build Phases — Crate Dependency Order

Crate dependencies form a DAG. Build from the bottom up.

```
Phase 1 — Build Order:

  ai-core              (no internal deps — pure traits + types)
       ↓
  ai-provider-openai   (depends on ai-core + reqwest)
  ai-provider-anthropic (depends on ai-core + reqwest)
  ai-provider-ollama   (depends on ai-core + reqwest)
       ↓
  ai-tools             (depends on ai-core + reqwest)
  ai-tools-macros      (depends on ai-core, proc-macro crate)
       ↓
  ai-agent             (depends on ai-core)
       ↓
  ai-pipeline          (depends on ai-core, ai-agent)

Phase 2 — Build Order:

  ai-memory            (depends on ai-core)
  ai-cache             (depends on ai-core)
  ai-observability     (depends on ai-core, tracing)
       ↓
  ai-rag               (depends on ai-core, ai-memory)
       ↓
  ai-server            (depends on ai-core, ai-agent, ai-pipeline, axum)

Phase 3:

  ai-cli               (depends on everything)
```

---

## 8. Key Design Decisions

### 8.1 Why Reqwest

- **Connection pooling**: Built-in via hyper. One `Client` per provider, reused across all requests. No manual pool management.
- **Streaming**: `bytes_stream()` returns a `Stream<Item = Result<Bytes>>` — maps directly to `BoxStream` for SSE parsing.
- **TLS**: `rustls-tls` feature avoids OpenSSL dependency. Pure Rust TLS stack.
- **JSON**: `json()` method serializes any `Serialize` type directly into the request body.
- **Middleware**: `reqwest-middleware` crate extends reqwest with retry, logging, tracing — matches REQ-7.4 and REQ-11.1.

### 8.2 Why Traits Over Enums for Extensibility

Users add their own providers, tools, and memory backends by implementing traits — no need to modify framework code or match on enum variants. The framework ships with built-in implementations; users add theirs via the plugin system.

### 8.3 Why Workspace Over Single Crate

- **Compile time**: Only rebuild what changed. Provider crates rarely change together.
- **Dependency isolation**: `ai-rag` pulls in heavy deps (qdrant, ort) that `ai-agent` doesn't need.
- **Optional features**: Users depend on only the crates they need.
- **Publishing**: Each crate can be versioned and published independently.

### 8.4 Why async-trait (for now)

Native async fn in traits (RPITIT) landed in Rust 1.75, but with limitations:
- No `Send` bounds on returned futures (needed for tokio::spawn)
- No `dyn` dispatch (needed for `Box<dyn Provider>`)

Once these stabilize, `async_trait` can be removed with a find-and-replace. The trait signatures stay the same.

### 8.5 Error Handling Strategy

```rust
// Library errors — typed, exhaustive, thiserror
#[derive(Debug, thiserror::Error)]
pub enum ProviderError {
    #[error("HTTP request failed: {0}")]
    Http(#[from] reqwest::Error),

    #[error("API error {status}: {body}")]
    Api { status: reqwest::StatusCode, body: String },

    #[error("Failed to deserialize response: {0}")]
    Deserialize(#[from] serde_json::Error),

    #[error("Request cancelled")]
    Cancelled,
}

// Application errors — flexible, anyhow
// Used in CLI, examples, and application code that doesn't need typed matching
```

### 8.6 Type-State Builder for Agent

Prevents runtime errors at compile time:

```rust
// These marker types cost zero bytes at runtime
pub struct NoProvider;
pub struct HasProvider(P); // where P: Provider

pub struct AgentBuilder<P, M, State> { /* ... */ }

// build() only exists when State = HasProvider
impl<P, M> AgentBuilder<P, M, HasProvider>
where
    P: Provider,
    M: Memory,
{
    pub fn build(self) -> Result<Agent<P, M>, AgentError> { /* ... */ }
}

// provider() transitions from NoProvider → HasProvider
impl<M> AgentBuilder<NoProvider, M, NoProvider> {
    pub fn provider<P: Provider>(self, provider: P) -> AgentBuilder<P, M, HasProvider> {
        AgentBuilder { provider, ..self }
    }
}
```

Calling `.build()` without setting a provider is a compile error — not a runtime panic.
