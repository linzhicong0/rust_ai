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
//! - [`plugin`] — Plugin lifecycle and registry
//! - [`config`] — Framework configuration
//! - [`error`] — Unified error types

// Re-export core types at the crate root for convenience
pub use crate::types::{
    AgentEvent, AgentOutput, CompletionResponse, Content, ContentPart, FinishReason, ImageDetail,
    Message, ModelConfig, Role, StreamChunk, ToolCall, ToolCallDelta, Usage,
};

// Re-export context management types
pub use crate::context::{
    estimate_tokens, ContextConfig, ContextManager, ContextResult, ContextUsage, TruncationStrategy,
};

// Re-export cost tracking types
pub use crate::cost::{
    agent_scope, new_request_id, project_scope, request_scope, workflow_scope, CostAccumulator,
    CostSnapshot, CostTracker, ModelPricing, PricingTable, GLOBAL_SCOPE,
};

// Re-export model registry types
pub use crate::model_registry::{
    ModelCapability, ModelCost, ModelInfo, ModelRegistry, ModelRegistryError,
};

// Re-export prompt injection defense types
pub use crate::prompt_injection::{
    global_defender, has_injection_attempts, has_prompt_leaks, InjectionPattern,
    InjectionScanResult, LeakDetectionResult, PromptInjectionDefender,
};

// Re-export core traits
pub use crate::embedder::Embedder;
pub use crate::guardrail::{
    CustomGuardrail, Guardrail, GuardrailAction, GuardrailChain, LengthLimiter, PiiAction,
    PromptInjectionGuard, RegexPiiDetector,
};
pub use crate::memory::{Memory, MemoryEntry, ScopedMemory};
pub use crate::plugin::{Plugin, PluginError, PluginRegistry, PluginRequest, PluginResponse};
pub use crate::provider::Provider;
pub use crate::tool::{Tool, ToolDescriptor, ToolOutput};

// Re-export tool validation (REQ-3.4)
pub use crate::tool::{coerce_input, validate_tool_input, ValidationError};

// Re-export configuration
pub use crate::config::FrameworkConfig;

// Re-export Client — primary entry point per REQ-15.1
pub use crate::client::{Client, TrackedCompletionResponse};

// Re-export errors — users typically import these via `use ai_core::Error`
// or use specific error variants like `ProviderError`
pub use crate::error::{
    AgentError, EmbedderError, GuardrailError, MemoryError, PipelineError, ProviderError, ToolError,
};

// Re-export template engine
pub use crate::template::TemplateEngine;

// Re-export structured output
pub use crate::structured::{
    complete_structured, extract_and_fix_json, extract_json, fix_json, StructuredOutputConfig,
    StructuredOutputError, StructuredOutputValidator,
};

// Re-export structured data (REQ-8.5)
pub use crate::structured_data::{
    convert, parse_csv, parse_json, parse_tsv, to_csv, to_json, ColumnType, DataFormat, DataRow,
    DataSchema, StructuredData, StructuredDataError,
};

// Re-export output formatting (REQ-9.4)
pub use crate::output_format::{format_output, OutputFormat, OutputFormatConfig};

// Re-export rate limiting (REQ-12.4)
pub use crate::rate_limit::{RateLimitConfig, RateLimitExceeded, TokenBucketRateLimiter};

// Re-export prompt caching (REQ-12.3)
pub use crate::prompt_cache::{
    detect_caching_support, mark_system_prompt_cacheable, parse_anthropic_cache_metadata,
    CacheMarker, CacheMetadata, CacheableContent, CachingProvider, PromptCacheConfig,
};

// Re-export hot reload (REQ-15.5)
pub use crate::hot_reload::{ConfigRegistry, FileWatcherConfig, HotReloadError, HotReloadable};

// Re-export typed interfaces (REQ-15.6)
pub use crate::typed::{
    AgentStep, FnStep, ToolStep, TypedAgent, TypedChain, TypedError, TypedStep, TypedStepExt,
    TypedTool,
};

// Re-export system prompt builder (REQ-4.4)
pub use crate::system_prompt::SystemPromptBuilder;

// Re-export tool composition (REQ-3.5)
pub use crate::tool_compose::{ComposedTool, IntermediateResult, ToolPipeExt, ToolPipeline};

// Re-export image generation (REQ-8.2)
pub use crate::image_gen::{
    GeneratedImage, ImageData, ImageEditConfig, ImageFormat, ImageGenConfig, ImageGenError,
    ImageGenerator, ImageMessage, ImageQuality, ImageSize, ImageStyle,
};

// Re-export audio processing (REQ-8.3)
pub use crate::audio::{
    AudioError, AudioFormat, Language, SynthesisResult, SynthesizeConfig, Synthesizer, TimedWord,
    TranscribeConfig, Transcriber, TranscriptResult, TranscriptSegment, Voice, VoiceGender,
};

// Re-export benchmarking (REQ-10.1)
pub use crate::benchmark::{
    BenchmarkConfig, BenchmarkError, BenchmarkMetrics, BenchmarkRun, BenchmarkRunner, CaseResult,
    EvalCase, EvalDataset, RunComparison,
};

// Re-export A/B testing (REQ-10.2)
pub use crate::ab_testing::{
    AbTestCollector, AbTestConfig as AbTestExperimentConfig, AbTestError, Observation,
    SignificanceResult, TestVariant, VariantAssigner, VariantMetrics,
};

// Re-export regression testing (REQ-10.3)
pub use crate::regression::{
    CaseTestResult, GoldenCase, GoldenDataset, RegressionConfig, RegressionError, RegressionRunner,
    RegressionTestResult, SimilarityStrategy,
};

// Re-export prompt registry (REQ-4.2)
pub use crate::prompt_registry::{
    AbTestConfig, AbVariant, PromptDiff, PromptRegistry, PromptVersion,
};

// Module declarations
pub mod ab_testing;
pub mod audio;
pub mod benchmark;
pub mod client;
pub mod config;
pub mod context;
pub mod cost;
pub mod embedder;
pub mod error;
pub mod guardrail;
pub mod hot_reload;
pub mod image_gen;
pub mod memory;
pub mod model_registry;
pub mod output_format;
pub mod plugin;
pub mod prompt_cache;
pub mod prompt_injection;
pub mod prompt_registry;
pub mod provider;
pub mod rate_limit;
pub mod regression;
pub mod structured;
pub mod structured_data;
pub mod system_prompt;
pub mod template;
pub mod tool;
pub mod tool_compose;
pub mod typed;
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
        AgentEvent, AgentOutput, CompletionResponse, Content, ContentPart, Message, ModelConfig,
        Role, Usage,
    };

    // Core traits
    pub use crate::embedder::Embedder;
    pub use crate::memory::{Memory, MemoryEntry, ScopedMemory};
    pub use crate::provider::Provider;
    pub use crate::tool::{Tool, ToolDescriptor};

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
        let _plugin_request = PluginRequest::new("gpt-4", vec![Message::user("test")]);
        let _plugin_response = PluginResponse::new("ok", Usage::default());
        let _plugin_error = PluginError::Lifecycle("test".to_string());
        let _plugin: Option<Box<dyn Plugin>> = None;

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
