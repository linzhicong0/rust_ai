// Copyright 2026 AI Framework Contributors. SPDX-License-Identifier: Apache-2.0

//! # AI Framework — Core Library
//!
//! The core library provides the fundamental traits, types, and abstractions
//! for building AI-powered applications with LLMs, agents, tools, and workflows.
//!
//! ## Quick Start
//!
//! The framework provides traits for providers, agents, tools, and memory.
//! See the [`prelude`](prelude) module for a convenient set of imports.
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
    Content, ContentPart, CompletionResponse, FinishReason, ImageDetail, Message,
    ModelConfig, Role, StreamChunk, ToolCall, ToolCallDelta, Usage,
    AgentEvent, AgentOutput,
};

// Re-export context management types
pub use crate::context::{
    ContextConfig, ContextManager, ContextResult, ContextUsage, TruncationStrategy, estimate_tokens,
};

// Re-export cost tracking types
pub use crate::cost::{
    agent_scope, new_request_id, project_scope, request_scope, workflow_scope,
    CostAccumulator, CostSnapshot, CostTracker, ModelPricing, PricingTable,
    GLOBAL_SCOPE,
};

// Re-export model registry types
pub use crate::model_registry::{
    ModelCapability, ModelCost, ModelInfo, ModelRegistry, ModelRegistryError,
};

// Re-export prompt injection defense types
pub use crate::prompt_injection::{
    InjectionPattern, InjectionScanResult, LeakDetectionResult,
    PromptInjectionDefender, global_defender, has_injection_attempts, has_prompt_leaks,
};

// Re-export core traits
pub use crate::provider::Provider;
pub use crate::tool::{Tool, ToolDescriptor, ToolOutput};
pub use crate::memory::{Memory, MemoryEntry, ScopedMemory};
pub use crate::embedder::Embedder;
pub use crate::guardrail::{
    Guardrail, GuardrailAction, GuardrailChain, RegexPiiDetector,
    LengthLimiter, CustomGuardrail, PiiAction
};

// Re-export configuration
pub use crate::config::FrameworkConfig;

// Re-export Client — primary entry point per REQ-15.1
pub use crate::client::{Client, TrackedCompletionResponse};

// Re-export errors — users typically import these via `use ai_core::Error`
// or use specific error variants like `ProviderError`
pub use crate::error::{
    AgentError, EmbedderError, GuardrailError, MemoryError,
    PipelineError, ProviderError, ToolError,
};

// Re-export template engine
pub use crate::template::TemplateEngine;

// Re-export structured output
pub use crate::structured::{StructuredOutputConfig, StructuredOutputValidator, complete_structured, extract_json, StructuredOutputError};

// Module declarations
pub mod client;
pub mod config;
pub mod context;
pub mod cost;
pub mod embedder;
pub mod error;
pub mod guardrail;
pub mod memory;
pub mod model_registry;
pub mod prompt_injection;
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

    // Client entry point
    pub use crate::client::Client;
}

/// Result type alias for convenience.
///
/// Using this result type is optional but recommended for consistency.
pub type Result<T, E = error::AgentError> = std::result::Result<T, E>;

#[cfg(test)]
mod tests {
    use super::*;

    // REQ-15.1: SDK/API - Test that all public re-exports compile correctly

    #[test]
    fn test_re_exports_are_accessible() {
        // Verify that re-exports compile and are accessible

        // Core types from types.rs
        let _role = Role::User;
        let _content = Content::Text("test".to_string());
        let _content_part = ContentPart::Text("test".to_string());

        let _config = ModelConfig::new("gpt-4");
        let _message = Message::user("test");
        let _finish_reason = FinishReason::Stop;

        // Struct types
        let _tool_call = ToolCall {
            id: "test".to_string(),
            name: "test_tool".to_string(),
            arguments: serde_json::json!({}),
        };

        let _usage = Usage {
            prompt_tokens: 100,
            completion_tokens: 50,
            total_tokens: 150,
        };

        // Error types can be instantiated
        let _tool_err = ToolError::Execution("test".to_string());
        let _agent_err = AgentError::MaxIterationsExceeded;
        let _provider_err = ProviderError::Cancelled;
        let _memory_err = MemoryError::NotFound("test".to_string());
        let _embedder_err = EmbedderError::Model("test".to_string());
        let _guardrail_err = GuardrailError::Check("test".to_string());
        let _pipeline_err = PipelineError::Context("test".to_string());
    }

    #[test]
    fn test_prelude_imports() {
        // Verify prelude can be used as expected
        use crate::prelude::*;

        let _msg = Message {
            role: Role::User,
            content: Content::Text("hello".to_string()),
        };

        let _config = ModelConfig::new("gpt-4");
        let _event = AgentEvent::Text("test".to_string());
        let _output = AgentOutput {
            content: "test".to_string(),
            usage: Usage::default(),
            estimated_cost: 0.0,
            tracked_scopes: vec![],
        };

        // Verify usage is accessible
        let _usage = Usage {
            prompt_tokens: 100,
            completion_tokens: 50,
            total_tokens: 150,
        };
    }

    #[test]
    fn test_config_re_export() {
        // Verify FrameworkConfig is accessible
        let _config = FrameworkConfig::default();
    }

    #[test]
    fn test_template_engine_re_export() {
        // Verify TemplateEngine is accessible
        let _engine = TemplateEngine::new().unwrap();
    }

    #[test]
    fn test_structured_output_re_export() {
        // Verify structured output types are accessible
        let _config = StructuredOutputConfig::new(serde_json::json!({"type": "object"}));
        let _err = StructuredOutputError::ValidationError("test".to_string());
    }

    #[test]
    fn test_result_type_alias() {
        // Verify the Result type alias works
        fn returns_result() -> Result<()> {
            Ok(())
        }
        let _ = returns_result();

        fn returns_error() -> Result<String> {
            Err(AgentError::MaxIterationsExceeded)
        }
        let _ = returns_error();
    }

    #[test]
    fn test_all_prelude_types_exist() {
        // This test ensures all types declared in prelude actually exist
        use crate::prelude::*;

        // These should all compile without errors
        let msg = Message::user("test");
        match msg.role {
            Role::User => {} // Verify User variant exists
            _ => {}
        }

        let _ = ModelConfig::gpt4();
        let _ = Role::User;
        let _ = Content::Text("test".to_string());
        let _ = AgentEvent::Text("test".to_string());
        let _ = AgentOutput {
            content: "test".to_string(),
            usage: Usage::default(),
            estimated_cost: 0.0,
            tracked_scopes: vec![],
        };
        let _ = Usage {
            prompt_tokens: 0,
            completion_tokens: 0,
            total_tokens: 0,
        };

        // Traits are object-safe
        let _tool: Option<Box<dyn Tool>> = None;
        let _memory: Option<Box<dyn Memory>> = None;
        let _embedder: Option<Box<dyn Embedder>> = None;

        let _config = FrameworkConfig::default();
    }
}
