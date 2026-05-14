#[derive(Debug, thiserror::Error)]
pub enum ProviderError {
    #[error("HTTP request failed: {0}")]
    Http(#[from] reqwest::Error),

    #[error("API error {status}: {body}")]
    Api {
        status: reqwest::StatusCode,
        body: String,
    },

    #[error("Failed to deserialize response: {0}")]
    Deserialize(#[from] serde_json::Error),

    #[error("Request cancelled")]
    Cancelled,
}

#[derive(Debug, thiserror::Error)]
pub enum ToolError {
    #[error("Tool execution failed: {0}")]
    Execution(String),

    #[error("Invalid input: {0}")]
    InvalidInput(String),

    #[error("Deserialization error: {0}")]
    Deserialize(#[from] serde_json::Error),
}

#[derive(Debug, thiserror::Error)]
pub enum MemoryError {
    #[error("Storage error: {0}")]
    Storage(String),

    #[error("Entry not found: {0}")]
    NotFound(String),
}

#[derive(Debug, thiserror::Error)]
pub enum AgentError {
    #[error("Provider error: {0}")]
    Provider(#[from] ProviderError),

    #[error("Tool error: {0}")]
    Tool(#[from] ToolError),

    #[error("Memory error: {0}")]
    Memory(#[from] MemoryError),

    #[error("Max iterations exceeded")]
    MaxIterationsExceeded,

    #[error("Tool not found: {0}")]
    ToolNotFound(String),
}

#[derive(Debug, thiserror::Error)]
pub enum PipelineError {
    #[error("Step failed '{name}': {source}")]
    StepFailed {
        name: String,
        #[source]
        source: Box<dyn std::error::Error + Send + Sync>,
    },

    #[error("Agent error: {0}")]
    Agent(#[from] AgentError),

    #[error("Context error: {0}")]
    Context(String),
}

#[derive(Debug, thiserror::Error)]
pub enum EmbedderError {
    #[error("Embedding failed: {0}")]
    Embedding(String),

    #[error("Model error: {0}")]
    Model(String),
}

#[derive(Debug, thiserror::Error)]
pub enum GuardrailError {
    #[error("Guardrail check failed: {0}")]
    Check(String),
}
