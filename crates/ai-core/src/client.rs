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

use crate::cost::{new_request_id, request_scope, CostTracker, GLOBAL_SCOPE};
use crate::error::ProviderError;
use crate::provider::Provider;
use crate::tool::ToolDescriptor;
use crate::types::{CompletionResponse, Message, ModelConfig, StreamChunk};

/// Completion response enriched with cost-tracking metadata.
#[derive(Debug)]
pub struct TrackedCompletionResponse {
    /// The provider response.
    pub response: CompletionResponse,

    /// Generated request identifier used for request-scoped accounting.
    pub request_id: String,

    /// All scopes updated for this completion.
    pub tracked_scopes: Vec<String>,

    /// Estimated cost in USD for this completion.
    pub estimated_cost: f64,
}

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
    cost_tracker: CostTracker,
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
            cost_tracker: CostTracker::new(),
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

    /// Override the cost tracker used by this client.
    pub fn with_cost_tracker(mut self, cost_tracker: CostTracker) -> Self {
        self.cost_tracker = cost_tracker;
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

    /// Return the cost tracker associated with this client.
    pub fn cost_tracker(&self) -> &CostTracker {
        &self.cost_tracker
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
        Ok(self.complete_tracked(messages).await?.response)
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
        Ok(self
            .complete_with_tracking(messages, config, &[], &[])
            .await?
            .response)
    }

    /// Send a completion request with tool definitions for function calling.
    pub async fn complete_with_tools(
        &self,
        messages: Vec<Message>,
        tools: &[ToolDescriptor],
    ) -> Result<CompletionResponse, ProviderError> {
        Ok(self
            .complete_with_tracking(messages, &self.default_config, tools, &[])
            .await?
            .response)
    }

    /// Send a completion request and expose the request scope that was tracked.
    pub async fn complete_tracked(
        &self,
        messages: Vec<Message>,
    ) -> Result<TrackedCompletionResponse, ProviderError> {
        self.complete_with_tracking(messages, &self.default_config, &[], &[])
            .await
    }

    /// Send a completion request and record usage for the generated request scope,
    /// the global scope, and any additional caller-provided scopes.
    pub async fn complete_with_tracking(
        &self,
        messages: Vec<Message>,
        config: &ModelConfig,
        tools: &[ToolDescriptor],
        additional_scopes: &[String],
    ) -> Result<TrackedCompletionResponse, ProviderError> {
        let request_id = new_request_id();
        let request_scope_name = request_scope(&request_id);
        let mut tracked_scopes = vec![request_scope_name.clone(), GLOBAL_SCOPE.to_string()];

        for scope in additional_scopes {
            if !tracked_scopes.iter().any(|existing| existing == scope) {
                tracked_scopes.push(scope.clone());
            }
        }

        let response = self.provider.complete(messages, config, tools).await?;
        let estimated_cost = self
            .cost_tracker
            .estimate_cost(&config.model, &response.usage);

        self.cost_tracker
            .record_many(
                tracked_scopes.iter().cloned(),
                &config.model,
                &response.usage,
            )
            .await;

        Ok(TrackedCompletionResponse {
            response,
            request_id,
            tracked_scopes,
            estimated_cost,
        })
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
        let client = Client::new(MockProvider).with_config(ModelConfig::new("gpt-4"));
        let resp = client.complete(vec![Message::user("hi")]).await.unwrap();
        assert_eq!(resp.content, "hello");
    }

    #[tokio::test]
    async fn test_client_complete_tracked_records_request_and_global_scopes() {
        let client = Client::new(MockProvider).with_config(ModelConfig::new("gpt-4"));

        let tracked = client
            .complete_tracked(vec![Message::user("hi")])
            .await
            .unwrap();

        assert!(tracked.request_id.len() > 10);
        assert!(tracked
            .tracked_scopes
            .iter()
            .any(|scope| scope == GLOBAL_SCOPE));

        let request_snapshot = client
            .cost_tracker()
            .get(&request_scope(&tracked.request_id))
            .await;
        assert_eq!(request_snapshot.request_count, 1);

        let global_snapshot = client.cost_tracker().get(GLOBAL_SCOPE).await;
        assert_eq!(global_snapshot.request_count, 1);
        assert!(tracked.estimated_cost > 0.0);
    }

    #[tokio::test]
    async fn test_client_complete_with_tracking_records_custom_scope() {
        let client = Client::new(MockProvider).with_config(ModelConfig::new("gpt-4"));
        let custom_scope = "agent:researcher".to_string();

        let tracked = client
            .complete_with_tracking(
                vec![Message::user("hi")],
                client.default_config(),
                &[],
                &[custom_scope.clone()],
            )
            .await
            .unwrap();

        assert!(tracked.tracked_scopes.contains(&custom_scope));
        assert_eq!(
            client.cost_tracker().get(&custom_scope).await.request_count,
            1
        );
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
