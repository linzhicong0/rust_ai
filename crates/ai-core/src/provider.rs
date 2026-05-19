//! LLM provider abstraction.
//!
//! The [`Provider`] trait defines a unified interface for interacting with
//! different LLM providers (OpenAI, Anthropic, Google, Ollama, etc.).
//!
//! ## Example
//!
//! ```rust,no_run
//! # use ai_core::{Provider, ModelConfig, ToolDescriptor};
//! # use ai_core::types::{Message, CompletionResponse, StreamChunk};
//! # use ai_core::error::ProviderError;
//! # use futures::stream::BoxStream;
//! struct MyProvider;
//!
//! # #[async_trait::async_trait]
//! impl Provider for MyProvider {
//! #     async fn complete(
//! #         &self,
//! #         messages: Vec<Message>,
//! #         config: &ModelConfig,
//! #         tools: &[ToolDescriptor],
//! #     ) -> Result<CompletionResponse, ProviderError> {
//!         // Implementation...
//!         # todo!()
//! #     }
//!
//! #     fn stream(
//! #         &self,
//! #         messages: Vec<Message>,
//! #         config: &ModelConfig,
//! #         tools: &[ToolDescriptor],
//! #     ) -> BoxStream<'static, Result<StreamChunk, ProviderError>> {
//!         // Implementation...
//!         # Box::pin(futures::stream::empty())
//! #     }
//!
//! #     async fn embed(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>, ProviderError> {
//!         // Implementation...
//!         # todo!()
//! #     }
//!
//! #     fn name(&self) -> &str {
//!         "my-provider"
//! #     }
//! }
//! ```

use async_trait::async_trait;
use futures::stream::BoxStream;

use crate::error::ProviderError;
use crate::tool::ToolDescriptor;
use crate::types::{CompletionResponse, ModelConfig, StreamChunk};

/// Unified interface for LLM providers.
///
/// This trait abstracts over different LLM APIs, allowing applications
/// to switch providers without code changes. Implementations handle
/// authentication, request formatting, and response parsing.
#[async_trait]
pub trait Provider: Send + Sync {
    /// Generate a completion for the given messages.
    ///
    /// # Arguments
    ///
    /// * `messages` — Conversation history including the new prompt
    /// * `config` — Model parameters (temperature, max tokens, etc.)
    /// * `tools` — Available tools for function calling (may be empty)
    ///
    /// # Returns
    ///
    /// A [`CompletionResponse`] containing the generated content, tool calls,
    /// usage statistics, and finish reason.
    async fn complete(
        &self,
        messages: Vec<crate::types::Message>,
        config: &ModelConfig,
        tools: &[ToolDescriptor],
    ) -> Result<CompletionResponse, ProviderError>;

    /// Stream a completion token-by-token.
    ///
    /// Returns a stream of [`StreamChunk`] values, each containing a delta
    /// of the generated content. The stream may yield tool call deltas
    /// and eventually includes usage statistics.
    ///
    /// # Cancellation
    ///
    /// Dropping the stream cancels the request. Use a [`CancellationToken`]
    /// for cooperative cancellation.
    ///
    /// [`CancellationToken`]: tokio_util::sync::CancellationToken
    fn stream(
        &self,
        messages: Vec<crate::types::Message>,
        config: &ModelConfig,
        tools: &[ToolDescriptor],
    ) -> BoxStream<'static, Result<StreamChunk, ProviderError>>;

    /// Generate embeddings for the given texts.
    ///
    /// # Arguments
    ///
    /// * `texts` — Strings to embed (typically batched)
    ///
    /// # Returns
    ///
    /// A vector of embeddings, one per input text. Each embedding is
    /// a vector of f32 values with dimensionality dependent on the model.
    async fn embed(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>, ProviderError>;

    /// Returns the name of this provider (e.g., "openai", "anthropic").
    fn name(&self) -> &str;
}
