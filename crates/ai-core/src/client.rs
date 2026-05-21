// Copyright 2026 AI Framework Contributors. SPDX-License-Identifier: Apache-2.0

//! High-level `Client` entry point for the AI framework.
//!
//! [`Client`] is the primary user-facing type for interacting with LLM providers.
//! It owns a [`Provider`] and a default [`ModelConfig`], and exposes ergonomic
//! methods for completions, streaming, and embeddings.
//!
//! # Example
//!
//! ```rust,no_run
//! # use ai_core::client::Client;
//! # use ai_core::types::{Message, Role, Content, ModelConfig};
//! # // Provider import omitted for brevity
//! # async fn example() -> Result<(), Box<dyn std::error::Error>> {
//! // Construct a client around any Provider implementation
//! // let client = Client::new(openai_provider);
//! //
//! // One-shot completion
//! // let response = client
//! //     .complete(vec![Message::user("What is Rust?")])
//! //     .await?;
//! // println!("{}", response.content_text().unwrap_or_default());
//! # Ok(())
//! # }
//! ```

use std::sync::Arc;

use futures::stream::BoxStream;

use crate::error::ProviderError;
use crate::provider::Provider;
use crate::tool::ToolDescriptor;
use crate::types::{CompletionResponse, Message, ModelConfig, StreamChunk};

/// The top-level client for LLM interaction.
///
/// `Client` wraps any [`Provider`] implementation and adds:
/// - A default [`ModelConfig`] applied to every request (overridable per-call)
/// - Ergonomic helper methods (`complete`, `stream`, `embed`)
/// - A consistent public API surface per REQ-15.1
///
/// # Thread Safety
///
/// `Client` is `Clone + Send + Sync`. The inner provider is wrapped in an
/// `Arc` so that multiple clones share the same connection pool.
#[derive(Clone)]
pub struct Client<P: Provider> {
    provider: Arc<P>,
    default_config: ModelConfig,
}

impl<P: Provider> Client<P> {
    /// Create a new client wrapping the given provider.
    ///
    /// Uses default [`ModelConfig`] settings. Call [`with_config`](Self::with_config)
    /// to override defaults.
    pub fn new(provider: P) -> Self {
        Self {
            provider: Arc::new(provider),
            default_config: ModelConfig::default(),
        }
    }

    /// Override the default model configuration.
    ///
    /// This config is merged into every request. Per-request configs passed to
    /// [`complete_with_config`](Self::complete_with_config) take higher precedence.
    pub fn with_config(mut self, config: ModelConfig) -> Self {
        self.default_config = config;
        self
    }

    /// Return a reference to the underlying provider.
    pub fn provider(&self) -> &P {
        &self.provider
    }

    /// Return the default model configuration.
    pub fn default_config(&self) -> &ModelConfig {
        &self.default_config
    }

    /// Send a completion request with the default model configuration.
    ///
    /// # Errors
    ///
    /// Returns [`ProviderError`] if the provider call fails.
    pub async fn complete(
        &self,
        messages: Vec<Message>,
    ) -> Result<CompletionResponse, ProviderError> {
        self.provider
            .complete(messages, &self.default_config, &[])
            .await
    }

    /// Send a completion request with a custom model configuration.
    ///
    /// The provided `config` is used as-is (the client's default config is
    /// *not* merged). Use [`ModelConfig::merge_with`] if you want layered overrides.
    pub async fn complete_with_config(
        &self,
        messages: Vec<Message>,
        config: &ModelConfig,
    ) -> Result<CompletionResponse, ProviderError> {
        self.provider.complete(messages, config, &[]).await
    }

    /// Send a completion request with tool definitions for function calling.
    pub async fn complete_with_tools(
        &self,
        messages: Vec<Message>,
        tools: &[ToolDescriptor],
    ) -> Result<CompletionResponse, ProviderError> {
        self.provider
            .complete(messages, &self.default_config, tools)
            .await
    }

    /// Open a streaming completion with the default model configuration.
    ///
    /// Returns a `BoxStream` that yields [`StreamChunk`] tokens as they arrive.
    /// Drop the stream to cancel.
    pub fn stream(
        &self,
        messages: Vec<Message>,
    ) -> BoxStream<'static, Result<StreamChunk, ProviderError>> {
        self.provider.stream(messages, &self.default_config, &[])
    }

    /// Open a streaming completion with a custom model configuration.
    pub fn stream_with_config(
        &self,
        messages: Vec<Message>,
        config: &ModelConfig,
    ) -> BoxStream<'static, Result<StreamChunk, ProviderError>> {
        self.provider.stream(messages, config, &[])
    }

    /// Generate embeddings for a list of texts.
    ///
    /// Returns one embedding vector per input string.
    pub async fn embed(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>, ProviderError> {
        self.provider.embed(texts).await
    }

    /// Return the provider's name (e.g., `"openai"`, `"anthropic"`).
    pub fn provider_name(&self) -> &str {
        self.provider.name()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{CompletionResponse, FinishReason, Usage};
    use async_trait::async_trait;
    use futures::stream;

    struct MockProvider;

    #[async_trait]
    impl Provider for MockProvider {
        async fn complete(
            &self,
            _messages: Vec<Message>,
            _config: &ModelConfig,
            _tools: &[ToolDescriptor],
        ) -> Result<CompletionResponse, ProviderError> {
            Ok(CompletionResponse {
                content: "hello".to_string(),
                tool_calls: vec![],
                usage: Usage {
                    prompt_tokens: 5,
                    completion_tokens: 2,
                    total_tokens: 7,
                },
                finish_reason: FinishReason::Stop,
            })
        }

        fn stream(
            &self,
            _messages: Vec<Message>,
            _config: &ModelConfig,
            _tools: &[ToolDescriptor],
        ) -> BoxStream<'static, Result<StreamChunk, ProviderError>> {
            Box::pin(stream::empty())
        }

        async fn embed(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>, ProviderError> {
            Ok(texts.iter().map(|_| vec![0.0f32; 3]).collect())
        }

        fn name(&self) -> &str {
            "mock"
        }
    }

    #[test]
    fn test_client_new() {
        let client = Client::new(MockProvider);
        assert_eq!(client.provider_name(), "mock");
    }

    #[test]
    fn test_client_with_config() {
        let config = ModelConfig::default().with_temperature(0.5);
        let client = Client::new(MockProvider).with_config(config);
        assert_eq!(client.default_config().temperature, Some(0.5));
    }

    #[tokio::test]
    async fn test_client_complete() {
        let client = Client::new(MockProvider);
        let resp = client.complete(vec![Message::user("hi")]).await.unwrap();
        assert_eq!(resp.content, "hello");
    }

    #[tokio::test]
    async fn test_client_embed() {
        let client = Client::new(MockProvider);
        let embeddings = client
            .embed(vec!["hello".to_string(), "world".to_string()])
            .await
            .unwrap();
        assert_eq!(embeddings.len(), 2);
        assert_eq!(embeddings[0].len(), 3);
    }
}
