// Copyright 2026 AI Framework Contributors. SPDX-License-Identifier: Apache-2.0

//! # Embedding Generation (REQ-1.5)
//!
//! Extended embedding generation with batch support, rate limiting,
//! and multi-provider support through a provider-agnostic interface.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::Mutex;

use crate::error::EmbedderError;

/// Configuration for batch embedding generation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchEmbeddingConfig {
    /// Maximum number of texts per batch.
    pub batch_size: usize,
    /// Maximum requests per minute for rate limiting.
    pub max_rpm: Option<u32>,
    /// Maximum tokens per minute for rate limiting.
    pub max_tpm: Option<u32>,
    /// Whether to retry failed batches.
    pub retry_on_failure: bool,
    /// Maximum retry attempts.
    pub max_retries: u32,
}

impl Default for BatchEmbeddingConfig {
    fn default() -> Self {
        Self {
            batch_size: 100,
            max_rpm: Some(60),
            max_tpm: None,
            retry_on_failure: true,
            max_retries: 3,
        }
    }
}

impl BatchEmbeddingConfig {
    /// Create with custom batch size.
    pub fn with_batch_size(mut self, size: usize) -> Self {
        self.batch_size = size;
        self
    }

    /// Set rate limit (requests per minute).
    pub fn with_max_rpm(mut self, rpm: u32) -> Self {
        self.max_rpm = Some(rpm);
        self
    }

    /// Set token rate limit (tokens per minute).
    pub fn with_max_tpm(mut self, tpm: u32) -> Self {
        self.max_tpm = Some(tpm);
        self
    }

    /// Disable retries.
    pub fn without_retries(mut self) -> Self {
        self.retry_on_failure = false;
        self
    }
}

/// Result of a batch embedding operation.
#[derive(Debug, Clone)]
pub struct BatchEmbeddingResult {
    /// The generated embeddings (one per input text).
    pub embeddings: Vec<Vec<f32>>,
    /// Total tokens consumed.
    pub total_tokens: usize,
    /// Number of batches processed.
    pub batches_processed: usize,
    /// Number of failed batches (retried or skipped).
    pub failed_batches: usize,
}

/// Embedding model information.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingModelInfo {
    /// Model identifier.
    pub model: String,
    /// Provider name.
    pub provider: String,
    /// Embedding dimension.
    pub dimension: usize,
    /// Maximum input tokens supported.
    pub max_input_tokens: usize,
    /// Whether the model supports batching natively.
    pub supports_batching: bool,
}

/// Extended embedder trait with batch support and rate limiting.
#[async_trait]
pub trait BatchEmbedder: Send + Sync {
    /// Generate embeddings for a batch of texts with rate limiting.
    async fn embed_batch(
        &self,
        texts: Vec<String>,
        config: &BatchEmbeddingConfig,
    ) -> Result<BatchEmbeddingResult, EmbedderError>;

    /// Get model information.
    fn model_info(&self) -> EmbeddingModelInfo;

    /// Get the embedding dimension.
    fn dimension(&self) -> usize;
}

/// A multi-provider embedding manager that can route to different providers.
pub struct EmbeddingManager {
    providers: HashMap<String, Arc<dyn BatchEmbedder>>,
    default_provider: Option<String>,
    config: BatchEmbeddingConfig,
}

impl EmbeddingManager {
    /// Create a new embedding manager.
    pub fn new(config: BatchEmbeddingConfig) -> Self {
        Self {
            providers: HashMap::new(),
            default_provider: None,
            config,
        }
    }

    /// Register an embedding provider.
    pub fn register_provider(&mut self, name: impl Into<String>, provider: Arc<dyn BatchEmbedder>) {
        let name = name.into();
        if self.default_provider.is_none() {
            self.default_provider = Some(name.clone());
        }
        self.providers.insert(name, provider);
    }

    /// Set the default provider.
    pub fn set_default(&mut self, name: impl Into<String>) {
        self.default_provider = Some(name.into());
    }

    /// Get a provider by name.
    pub fn get_provider(&self, name: &str) -> Option<&Arc<dyn BatchEmbedder>> {
        self.providers.get(name)
    }

    /// List all registered providers.
    pub fn list_providers(&self) -> Vec<&str> {
        self.providers.keys().map(|k| k.as_str()).collect()
    }

    /// Generate embeddings using the default provider.
    pub async fn embed(&self, texts: Vec<String>) -> Result<BatchEmbeddingResult, EmbedderError> {
        let provider_name = self
            .default_provider
            .as_ref()
            .ok_or_else(|| EmbedderError::Model("No default provider set".to_string()))?;
        self.embed_with(provider_name, texts).await
    }

    /// Generate embeddings using a specific provider.
    pub async fn embed_with(
        &self,
        provider: &str,
        texts: Vec<String>,
    ) -> Result<BatchEmbeddingResult, EmbedderError> {
        let embedder = self
            .providers
            .get(provider)
            .ok_or_else(|| EmbedderError::Model(format!("Provider not found: {provider}")))?;
        embedder.embed_batch(texts, &self.config).await
    }
}

/// In-memory batch embedder for testing purposes.
pub struct InMemoryBatchEmbedder {
    dimension: usize,
    model: String,
    provider: String,
    call_count: Arc<Mutex<usize>>,
}

impl InMemoryBatchEmbedder {
    /// Create a new in-memory batch embedder.
    pub fn new(dimension: usize, model: impl Into<String>, provider: impl Into<String>) -> Self {
        Self {
            dimension,
            model: model.into(),
            provider: provider.into(),
            call_count: Arc::new(Mutex::new(0)),
        }
    }

    /// Get the number of embed_batch calls made.
    pub async fn call_count(&self) -> usize {
        *self.call_count.lock().await
    }
}

#[async_trait]
impl BatchEmbedder for InMemoryBatchEmbedder {
    async fn embed_batch(
        &self,
        texts: Vec<String>,
        config: &BatchEmbeddingConfig,
    ) -> Result<BatchEmbeddingResult, EmbedderError> {
        let mut count = self.call_count.lock().await;
        *count += 1;

        let total_texts = texts.len();
        let batches_processed = if total_texts == 0 {
            0
        } else {
            (total_texts + config.batch_size - 1) / config.batch_size
        };

        // Generate deterministic embeddings based on text content
        let embeddings: Vec<Vec<f32>> = texts
            .iter()
            .map(|text| {
                let seed = text.len() as f32 / 100.0;
                (0..self.dimension)
                    .map(|i| (seed + i as f32 * 0.1).sin())
                    .collect()
            })
            .collect();

        let total_tokens = texts.iter().map(|t| t.len() / 4).sum();

        Ok(BatchEmbeddingResult {
            embeddings,
            total_tokens,
            batches_processed,
            failed_batches: 0,
        })
    }

    fn model_info(&self) -> EmbeddingModelInfo {
        EmbeddingModelInfo {
            model: self.model.clone(),
            provider: self.provider.clone(),
            dimension: self.dimension,
            max_input_tokens: 8192,
            supports_batching: true,
        }
    }

    fn dimension(&self) -> usize {
        self.dimension
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_batch_embedder_basic() {
        let embedder = InMemoryBatchEmbedder::new(384, "text-embedding-3-small", "openai");
        let config = BatchEmbeddingConfig::default();

        let texts = vec!["Hello world".to_string(), "Goodbye world".to_string()];
        let result = embedder.embed_batch(texts, &config).await.unwrap();

        assert_eq!(result.embeddings.len(), 2);
        assert_eq!(result.embeddings[0].len(), 384);
        assert_eq!(result.embeddings[1].len(), 384);
        assert_eq!(result.failed_batches, 0);
    }

    #[tokio::test]
    async fn test_batch_embedder_large_batch() {
        let embedder = InMemoryBatchEmbedder::new(128, "model", "provider");
        let config = BatchEmbeddingConfig::default().with_batch_size(10);

        let texts: Vec<String> = (0..25).map(|i| format!("Text number {i}")).collect();
        let result = embedder.embed_batch(texts, &config).await.unwrap();

        assert_eq!(result.embeddings.len(), 25);
        assert_eq!(result.batches_processed, 3); // ceil(25/10) = 3
    }

    #[tokio::test]
    async fn test_batch_embedder_empty_input() {
        let embedder = InMemoryBatchEmbedder::new(256, "model", "provider");
        let config = BatchEmbeddingConfig::default();

        let result = embedder.embed_batch(vec![], &config).await.unwrap();
        assert_eq!(result.embeddings.len(), 0);
        assert_eq!(result.batches_processed, 0);
    }

    #[tokio::test]
    async fn test_batch_embedder_model_info() {
        let embedder = InMemoryBatchEmbedder::new(1536, "text-embedding-ada-002", "openai");
        let info = embedder.model_info();

        assert_eq!(info.model, "text-embedding-ada-002");
        assert_eq!(info.provider, "openai");
        assert_eq!(info.dimension, 1536);
        assert!(info.supports_batching);
    }

    #[tokio::test]
    async fn test_batch_embedder_dimension() {
        let embedder = InMemoryBatchEmbedder::new(768, "model", "provider");
        assert_eq!(embedder.dimension(), 768);
    }

    #[tokio::test]
    async fn test_embedding_manager_register_and_embed() {
        let config = BatchEmbeddingConfig::default();
        let mut manager = EmbeddingManager::new(config);

        let provider = Arc::new(InMemoryBatchEmbedder::new(384, "model-a", "provider-a"));
        manager.register_provider("provider-a", provider);

        let result = manager.embed(vec!["test".to_string()]).await.unwrap();
        assert_eq!(result.embeddings.len(), 1);
        assert_eq!(result.embeddings[0].len(), 384);
    }

    #[tokio::test]
    async fn test_embedding_manager_multiple_providers() {
        let config = BatchEmbeddingConfig::default();
        let mut manager = EmbeddingManager::new(config);

        let provider_a = Arc::new(InMemoryBatchEmbedder::new(384, "small", "openai"));
        let provider_b = Arc::new(InMemoryBatchEmbedder::new(1536, "large", "openai"));

        manager.register_provider("small", provider_a);
        manager.register_provider("large", provider_b);

        // Default should be the first registered
        let result = manager.embed(vec!["test".to_string()]).await.unwrap();
        assert_eq!(result.embeddings[0].len(), 384);

        // Specific provider
        let result = manager
            .embed_with("large", vec!["test".to_string()])
            .await
            .unwrap();
        assert_eq!(result.embeddings[0].len(), 1536);
    }

    #[tokio::test]
    async fn test_embedding_manager_no_default() {
        let config = BatchEmbeddingConfig::default();
        let manager = EmbeddingManager::new(config);

        let result = manager.embed(vec!["test".to_string()]).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_embedding_manager_provider_not_found() {
        let config = BatchEmbeddingConfig::default();
        let manager = EmbeddingManager::new(config);

        let result = manager
            .embed_with("nonexistent", vec!["test".to_string()])
            .await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_embedding_manager_list_providers() {
        let config = BatchEmbeddingConfig::default();
        let mut manager = EmbeddingManager::new(config);

        let p1 = Arc::new(InMemoryBatchEmbedder::new(384, "m1", "p1"));
        let p2 = Arc::new(InMemoryBatchEmbedder::new(768, "m2", "p2"));

        manager.register_provider("alpha", p1);
        manager.register_provider("beta", p2);

        let providers = manager.list_providers();
        assert_eq!(providers.len(), 2);
        assert!(providers.contains(&"alpha"));
        assert!(providers.contains(&"beta"));
    }

    #[tokio::test]
    async fn test_embedding_manager_set_default() {
        let config = BatchEmbeddingConfig::default();
        let mut manager = EmbeddingManager::new(config);

        let p1 = Arc::new(InMemoryBatchEmbedder::new(384, "m1", "p1"));
        let p2 = Arc::new(InMemoryBatchEmbedder::new(768, "m2", "p2"));

        manager.register_provider("small", p1);
        manager.register_provider("large", p2);
        manager.set_default("large");

        let result = manager.embed(vec!["test".to_string()]).await.unwrap();
        assert_eq!(result.embeddings[0].len(), 768);
    }

    #[tokio::test]
    async fn test_batch_embedding_config_builder() {
        let config = BatchEmbeddingConfig::default()
            .with_batch_size(50)
            .with_max_rpm(120)
            .with_max_tpm(1_000_000)
            .without_retries();

        assert_eq!(config.batch_size, 50);
        assert_eq!(config.max_rpm, Some(120));
        assert_eq!(config.max_tpm, Some(1_000_000));
        assert!(!config.retry_on_failure);
    }

    #[tokio::test]
    async fn test_batch_embedder_call_count() {
        let embedder = InMemoryBatchEmbedder::new(128, "model", "provider");
        let config = BatchEmbeddingConfig::default();

        assert_eq!(embedder.call_count().await, 0);

        embedder
            .embed_batch(vec!["a".to_string()], &config)
            .await
            .unwrap();
        assert_eq!(embedder.call_count().await, 1);

        embedder
            .embed_batch(vec!["b".to_string()], &config)
            .await
            .unwrap();
        assert_eq!(embedder.call_count().await, 2);
    }

    #[tokio::test]
    async fn test_deterministic_embeddings() {
        let embedder = InMemoryBatchEmbedder::new(64, "model", "provider");
        let config = BatchEmbeddingConfig::default();

        let result1 = embedder
            .embed_batch(vec!["same text".to_string()], &config)
            .await
            .unwrap();
        let result2 = embedder
            .embed_batch(vec!["same text".to_string()], &config)
            .await
            .unwrap();

        assert_eq!(result1.embeddings[0], result2.embeddings[0]);
    }
}
