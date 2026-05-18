//! Text embedding generation.
//!
//! The [`Embedder`] trait defines a unified interface for generating
//! text embeddings from various providers (OpenAI, Cohere, local models).
//!
//! ## Example
//!
//! ```rust,no_run
//! struct MyEmbedder;
//!
//! #[async_trait::async_trait]
//! impl ai_core::Embedder for MyEmbedder {
//!     async fn embed(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>, ai_core::EmbedderError> {
//!         // Generate embeddings...
//!         # Ok(vec![])
//!     }
//!
//!     fn name(&self) -> &str {
//!         "my-embedder"
//!     }
//! }
//! ```

use async_trait::async_trait;

use crate::error::EmbedderError;

/// Unified interface for text embedding generation.
///
/// Embedders convert text into fixed-size vector representations (embeddings)
/// that capture semantic meaning. These are used for:
/// - Semantic search
/// - RAG (retrieval-augmented generation)
/// - Clustering and classification
#[async_trait]
pub trait Embedder: Send + Sync {
    /// Generate embeddings for the given texts.
    ///
    /// # Arguments
    ///
    /// * `texts` — Strings to embed (typically batched for efficiency)
    ///
    /// # Returns
    ///
    /// A vector of embeddings, one per input text. Each embedding is
    /// a vector of f32 values. The dimensionality depends on the model.
    async fn embed(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>, EmbedderError>;

    /// Returns the name of this embedder.
    fn name(&self) -> &str;
}

#[cfg(test)]
mod tests {
    use super::*;

    // Mock embedder for testing
    struct MockEmbedder {
        embedding_size: usize,
    }

    #[async_trait::async_trait]
    impl Embedder for MockEmbedder {
        async fn embed(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>, EmbedderError> {
            Ok(texts
                .iter()
                .map(|_| vec![0.0; self.embedding_size])
                .collect())
        }

        fn name(&self) -> &str {
            "mock"
        }
    }

    #[tokio::test]
    async fn test_mock_embedder() {
        let embedder = MockEmbedder { embedding_size: 10 };

        let texts = vec!["hello".to_string(), "world".to_string()];
        let embeddings = embedder.embed(texts).await.unwrap();

        assert_eq!(embeddings.len(), 2);
        assert_eq!(embeddings[0].len(), 10);
        assert_eq!(embeddings[1].len(), 10);
    }

    #[tokio::test]
    async fn test_mock_embedder_name() {
        let embedder = MockEmbedder { embedding_size: 5 };
        assert_eq!(embedder.name(), "mock");
    }

    #[tokio::test]
    async fn test_embedder_empty_input() {
        let embedder = MockEmbedder { embedding_size: 10 };

        let embeddings = embedder.embed(vec![]).await.unwrap();

        assert_eq!(embeddings.len(), 0);
    }

    #[tokio::test]
    async fn test_embedder_single_text() {
        let embedder = MockEmbedder { embedding_size: 3 };

        let embeddings = embedder.embed(vec!["test".to_string()]).await.unwrap();

        assert_eq!(embeddings.len(), 1);
        assert_eq!(embeddings[0], vec![0.0, 0.0, 0.0]);
    }
}
