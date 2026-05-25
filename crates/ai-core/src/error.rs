//! Unified error types for the AI framework.
//!
//! This module defines typed, hierarchical errors using [`thiserror`].
//! Each major component has its own error type with specific variants.
//!
//! ## Error Hierarchy
//!
//! ```text
//! AgentError (top-level)
//!   ├─ ProviderError (LLM API issues)
//!   ├─ ToolError (Tool execution issues)
//!   ├─ MemoryError (Storage issues)
//!   └─ PipelineError (Workflow issues)
//!
//! EmbedderError (Embedding generation)
//! GuardrailError (Validation failures)
//! ```

/// Errors from LLM provider interactions.
#[derive(Debug, thiserror::Error)]
pub enum ProviderError {
    /// HTTP request failed (network, timeout, etc.).
    #[error("HTTP request failed: {0}")]
    Http(#[from] reqwest::Error),

    /// API returned an error response.
    #[error("API error {status}: {body}")]
    Api {
        /// HTTP status code
        status: reqwest::StatusCode,
        /// Response body (may contain error details)
        body: String,
    },

    /// Failed to deserialize API response.
    #[error("Failed to deserialize response: {0}")]
    Deserialize(#[from] serde_json::Error),

    /// Request was cancelled by the client.
    #[error("Request cancelled")]
    Cancelled,
}

/// Errors from tool execution.
#[derive(Debug, thiserror::Error)]
pub enum ToolError {
    /// Tool execution returned an error.
    #[error("Tool execution failed: {0}")]
    Execution(String),

    /// Tool input doesn't match the schema.
    #[error("Invalid input: {0}")]
    InvalidInput(String),

    /// Failed to deserialize tool input.
    #[error("Deserialization error: {0}")]
    Deserialize(#[from] serde_json::Error),
}

/// Errors from memory operations.
#[derive(Debug, thiserror::Error)]
pub enum MemoryError {
    /// Storage backend error (database, file, etc.).
    #[error("Storage error: {0}")]
    Storage(String),

    /// Requested entry not found.
    #[error("Entry not found: {0}")]
    NotFound(String),
}

/// Errors from agent execution.
#[derive(Debug, thiserror::Error)]
pub enum AgentError {
    /// Error from the underlying provider.
    #[error("Provider error: {0}")]
    Provider(#[from] ProviderError),

    /// Error from tool execution.
    #[error("Tool error: {0}")]
    Tool(#[from] ToolError),

    /// Error from memory operations.
    #[error("Memory error: {0}")]
    Memory(#[from] MemoryError),

    /// Agent exceeded maximum iterations without finishing.
    #[error("Max iterations exceeded")]
    MaxIterationsExceeded,

    /// Agent tried to call a tool that wasn't registered.
    #[error("Tool not found: {0}")]
    ToolNotFound(String),

    /// Error during plan execution.
    #[error("Plan error: {0}")]
    PlanError(String),
}

/// Errors from pipeline execution.
#[derive(Debug, thiserror::Error)]
pub enum PipelineError {
    /// A pipeline step failed.
    #[error("Step failed '{name}': {source}")]
    StepFailed {
        /// Name of the failed step
        name: String,
        /// Underlying error
        #[source]
        source: Box<dyn std::error::Error + Send + Sync>,
    },

    /// Agent error in pipeline step.
    #[error("Agent error: {0}")]
    Agent(#[from] AgentError),

    /// Pipeline context error (missing data, type mismatch, etc.).
    #[error("Context error: {0}")]
    Context(String),
}

/// Errors from embedding generation.
#[derive(Debug, thiserror::Error)]
pub enum EmbedderError {
    /// Embedding generation failed.
    #[error("Embedding failed: {0}")]
    Embedding(String),

    /// Model-related error (not found, invalid parameters, etc.).
    #[error("Model error: {0}")]
    Model(String),
}

/// Errors from guardrail validation.
#[derive(Debug, thiserror::Error)]
pub enum GuardrailError {
    /// Guardrail check failed.
    #[error("Guardrail check failed: {0}")]
    Check(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_messages() {
        let err = ToolError::Execution("something failed".to_string());
        assert_eq!(err.to_string(), "Tool execution failed: something failed");

        let agent_err = AgentError::Tool(err);
        assert_eq!(
            agent_err.to_string(),
            "Tool error: Tool execution failed: something failed"
        );
    }
}
