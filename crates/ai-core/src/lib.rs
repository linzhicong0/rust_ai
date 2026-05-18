// Copyright 2026 AI Framework Contributors. SPDX-License-Identifier: Apache-2.0

//! # AI Framework — Core Library
//!
//! The core library provides the fundamental traits, types, and abstractions
//! for building AI-powered applications with LLMs, agents, tools, and workflows.
//!
//! ## Quick Start
//!
//! ```rust,no_run
//! use ai_core::{Provider, Agent, Tool, Memory};
//! use ai_core::types::{Message, Role, Content};
//!
//! // Create a provider (e.g., OpenAI, Anthropic)
//! let provider = MyProvider::new("api-key");
//!
//! // Define an agent with tools and memory
//! let agent = Agent::builder()
//!     .provider(provider)
//!     .role("You are a helpful assistant.")
//!     .tool(MyTool::new())
//!     .build()?;
//!
//! // Run the agent
//! let response = agent.run("Hello, world!").await?;
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! ## Key Concepts
//!
//! - **Provider**: Abstraction over LLM APIs (OpenAI, Anthropic, etc.)
//! - **Agent**: AI entity with role, tools, memory that can reason and act
//! - **Tool**: Function that agents can call (web search, file I/O, etc.)
//! - **Memory**: Storage for conversation history and context
//! - **Pipeline**: Multi-step workflow orchestration
//!
//! ## Module Organization
//!
//! - [`types`] — Core data types (Message, ModelConfig, Usage, etc.)
//! - [`provider`] — LLM provider trait and related types
//! - [`tool`] — Tool trait and execution
//! - [`memory`] — Memory storage and retrieval
//! - [`agent`] — Agent definition and ReAct loop
//! - [`pipeline`] — Workflow orchestration
//! - [`config`] — Framework configuration
//! - [`error`] — Unified error types

// Re-export core types at the crate root for convenience
pub use crate::types::{
    Content, ContentPart, CompletionResponse, FinishReason, Message,
    ModelConfig, Role, StreamChunk, ToolCall, ToolCallDelta, Usage,
    AgentEvent, AgentOutput,
};

// Re-export core traits
pub use crate::provider::Provider;
pub use crate::tool::{Tool, ToolDescriptor, ToolOutput};
pub use crate::memory::{Memory, MemoryEntry, ScopedMemory};
pub use crate::embedder::Embedder;
pub use crate::guardrail::Guardrail;

// Re-export configuration
pub use crate::config::FrameworkConfig;

// Re-export errors — users typically import these via `use ai_core::Error`
// or use specific error variants like `ProviderError`
pub use crate::error::{
    AgentError, EmbedderError, GuardrailError, MemoryError,
    PipelineError, ProviderError, ToolError,
};

// Re-export template engine
pub use crate::template::TemplateEngine;

// Re-export structured output
pub use crate::structured::{StructuredOutputConfig, StructuredOutputValidator, extract_json, StructuredOutputError};

// Module declarations
pub mod config;
pub mod embedder;
pub mod error;
pub mod guardrail;
pub mod memory;
pub mod provider;
pub mod structured;
pub mod template;
pub mod tool;
pub mod types;

// Prelude module for common imports
pub mod prelude {
    //! The "prelude" — common imports that most users will want.
    //!
    //! ```rust
    //! use ai_core::prelude::*;
    //! ```

    // Core types
    pub use crate::types::{
        AgentEvent, AgentOutput, Content, ContentPart, Message,
        ModelConfig, Role, CompletionResponse, Usage,
    };

    // Core traits
    pub use crate::provider::Provider;
    pub use crate::tool::{Tool, ToolDescriptor};
    pub use crate::memory::{Memory, MemoryEntry, ScopedMemory};
    pub use crate::embedder::Embedder;

    // Configuration
    pub use crate::config::FrameworkConfig;
}

/// Result type alias for convenience.
///
/// Using this result type is optional but recommended for consistency.
pub type Result<T, E = error::AgentError> = std::result::Result<T, E>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_re_exports_are_accessible() {
        // Verify that re-exports compile and are accessible

        // Types
        let _role = Role::User;
        let _content = Content::Text("test".to_string());

        // Error types can be instantiated
        let _err = ToolError::Execution("test".to_string());
    }

    #[test]
    fn test_prelude_imports() {
        // Verify prelude can be used as expected
        use crate::prelude::*;

        let _msg = Message {
            role: Role::User,
            content: Content::Text("hello".to_string()),
        };
    }
}
