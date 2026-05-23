//! Long-term memory with embedding-based semantic search.

use std::cmp::Ordering;
use std::collections::HashMap;
use std::sync::Arc;

use ai_core::{Embedder, MemoryError};
use async_trait::async_trait;
use chrono::{DateTime, Utc};
use serde_json::Value;
use tokio::sync::RwLock;
use tracing::debug;

/// A long-term memory entry with its generated embedding.
#[derive(Debug, Clone, PartialEq)]
pub struct LongTermMemoryEntry {
    pub id: String,
    pub content: String,
    pub embedding: Vec<f32>,
    pub metadata: HashMap<String, Value>,
    pub timestamp: DateTime<Utc>,
}

/// Configuration for long-term memory search behavior.
#[derive(Debug, Clone, PartialEq)]
pub struct LongTermMemoryConfig {
    pub similarity_threshold: f32,
    pub max_results: usize,
    pub embedding_dimension: usize,
}

impl Default for LongTermMemoryConfig {
    fn default() -> Self {
        Self {
            similarity_threshold: 0.7,
            max_results: 10,
            embedding_dimension: 1536,
        }
    }
}

/// Long-term memory abstraction for storing and retrieving embedded content.
#[async_trait]
pub trait LongTermMemory: Send + Sync {
    async fn store(
        &self,
        id: &str,
        content: &str,
        metadata: HashMap<String, Value>,
    ) -> Result<(), MemoryError>;

    async fn search(
        &self,
        query: &str,
        limit: usize,
    ) -> Result<Vec<LongTermMemoryEntry>, MemoryError>;

    async fn delete(&self, id: &str) -> Result<bool, MemoryError>;

    async fn get(&self, id: &str) -> Result<Option<LongTermMemoryEntry>, MemoryError>;
}

/// In-memory long-term store backed by embeddings for semantic search.
#[derive(Clone)]
pub struct InMemoryLongTermStore {
    entries: Arc<RwLock<HashMap<String, LongTermMemoryEntry>>>,
    embedder: Arc<dyn Embedder>,
    config: LongTermMemoryConfig,
}

impl InMemoryLongTermStore {
    /// Create a new in-memory long-term memory store.
    pub fn new(embedder: Arc<dyn Embedder>, config: LongTermMemoryConfig) -> Self {
        Self {
            entries: Arc::new(RwLock::new(HashMap::new())),
            embedder,
            config,
        }
    }

    /// Return the configured search behavior.
    pub fn config(&self) -> &LongTermMemoryConfig {
        &self.config
    }

    async fn embed_single(&self, text: &str) -> Result<Vec<f32>, MemoryError> {
        let mut embeddings = self
            .embedder
            .embed(vec![text.to_string()])
            .await
            .map_err(|err| MemoryError::Storage(format!("embedding generation failed: {err}")))?;

        if embeddings.len() != 1 {
            return Err(MemoryError::Storage(format!(
                "expected 1 embedding, got {}",
                embeddings.len()
            )));
        }

        let embedding = embeddings
            .pop()
            .ok_or_else(|| MemoryError::Storage("embedder returned no embeddings".to_string()))?;

        self.validate_embedding(&embedding)?;
        Ok(embedding)
    }

    fn validate_embedding(&self, embedding: &[f32]) -> Result<(), MemoryError> {
        if embedding.len() != self.config.embedding_dimension {
            return Err(MemoryError::Storage(format!(
                "invalid embedding dimension: expected {}, got {}",
                self.config.embedding_dimension,
                embedding.len()
            )));
        }

        Ok(())
    }

    fn cosine_similarity(&self, left: &[f32], right: &[f32]) -> Result<f32, MemoryError> {
        if left.len() != right.len() {
            return Err(MemoryError::Storage(format!(
                "embedding dimension mismatch: {} != {}",
                left.len(),
                right.len()
            )));
        }

        let dot_product: f32 = left.iter().zip(right.iter()).map(|(a, b)| a * b).sum();
        let left_norm: f32 = left.iter().map(|value| value * value).sum::<f32>().sqrt();
        let right_norm: f32 = right.iter().map(|value| value * value).sum::<f32>().sqrt();

        if left_norm == 0.0 || right_norm == 0.0 {
            return Ok(0.0);
        }

        Ok(dot_product / (left_norm * right_norm))
    }
}

#[async_trait]
impl LongTermMemory for InMemoryLongTermStore {
    async fn store(
        &self,
        id: &str,
        content: &str,
        metadata: HashMap<String, Value>,
    ) -> Result<(), MemoryError> {
        let embedding = self.embed_single(content).await?;
        let entry = LongTermMemoryEntry {
            id: id.to_string(),
            content: content.to_string(),
            embedding,
            metadata,
            timestamp: Utc::now(),
        };

        debug!(id, "storing long-term memory entry");
        self.entries.write().await.insert(id.to_string(), entry);
        Ok(())
    }

    async fn search(
        &self,
        query: &str,
        limit: usize,
    ) -> Result<Vec<LongTermMemoryEntry>, MemoryError> {
        let query_embedding = self.embed_single(query).await?;
        let effective_limit = limit.min(self.config.max_results);

        let entries = self.entries.read().await;
        let mut scored_entries = Vec::new();

        for entry in entries.values() {
            let similarity = self.cosine_similarity(&query_embedding, &entry.embedding)?;
            if similarity >= self.config.similarity_threshold {
                scored_entries.push((similarity, entry.timestamp, entry.clone()));
            }
        }

        scored_entries.sort_by(|left, right| {
            right
                .0
                .partial_cmp(&left.0)
                .unwrap_or(Ordering::Equal)
                .then_with(|| right.1.cmp(&left.1))
        });

        Ok(scored_entries
            .into_iter()
            .take(effective_limit)
            .map(|(_, _, entry)| entry)
            .collect())
    }

    async fn delete(&self, id: &str) -> Result<bool, MemoryError> {
        Ok(self.entries.write().await.remove(id).is_some())
    }

    async fn get(&self, id: &str) -> Result<Option<LongTermMemoryEntry>, MemoryError> {
        Ok(self.entries.read().await.get(id).cloned())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ai_core::EmbedderError;
    use serde_json::json;
    use std::collections::{HashMap, HashSet};

    struct MockEmbedder {
        embeddings: HashMap<String, Vec<f32>>,
        fail_on: HashSet<String>,
    }

    impl MockEmbedder {
        fn new(embeddings: HashMap<String, Vec<f32>>) -> Self {
            Self {
                embeddings,
                fail_on: HashSet::new(),
            }
        }

        fn with_failure(mut self, text: &str) -> Self {
            self.fail_on.insert(text.to_string());
            self
        }
    }

    #[async_trait]
    impl Embedder for MockEmbedder {
        async fn embed(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>, EmbedderError> {
            texts
                .into_iter()
                .map(|text| {
                    if self.fail_on.contains(&text) {
                        return Err(EmbedderError::Embedding(format!("failed for {text}")));
                    }

                    self.embeddings.get(&text).cloned().ok_or_else(|| {
                        EmbedderError::Embedding(format!("missing embedding for {text}"))
                    })
                })
                .collect()
        }

        fn name(&self) -> &str {
            "mock-long-term-embedder"
        }
    }

    fn config() -> LongTermMemoryConfig {
        LongTermMemoryConfig {
            similarity_threshold: 0.8,
            max_results: 3,
            embedding_dimension: 3,
        }
    }

    fn store_with_embeddings(embeddings: HashMap<String, Vec<f32>>) -> InMemoryLongTermStore {
        InMemoryLongTermStore::new(Arc::new(MockEmbedder::new(embeddings)), config())
    }

    #[tokio::test]
    async fn store_and_get_round_trip() {
        let store = store_with_embeddings(HashMap::from([(
            "rust ownership".to_string(),
            vec![1.0, 0.0, 0.0],
        )]));
        let metadata = HashMap::from([("topic".to_string(), json!("rust"))]);

        store
            .store("entry-1", "rust ownership", metadata.clone())
            .await
            .unwrap();

        let entry = store.get("entry-1").await.unwrap().unwrap();
        assert_eq!(entry.id, "entry-1");
        assert_eq!(entry.content, "rust ownership");
        assert_eq!(entry.embedding, vec![1.0, 0.0, 0.0]);
        assert_eq!(entry.metadata, metadata);
    }

    #[tokio::test]
    async fn store_overwrites_existing_entry() {
        let store = store_with_embeddings(HashMap::from([
            ("first version".to_string(), vec![1.0, 0.0, 0.0]),
            ("second version".to_string(), vec![0.0, 1.0, 0.0]),
        ]));

        store
            .store("shared-id", "first version", HashMap::new())
            .await
            .unwrap();
        let first_timestamp = store.get("shared-id").await.unwrap().unwrap().timestamp;

        store
            .store("shared-id", "second version", HashMap::new())
            .await
            .unwrap();

        let entry = store.get("shared-id").await.unwrap().unwrap();
        assert_eq!(entry.content, "second version");
        assert_eq!(entry.embedding, vec![0.0, 1.0, 0.0]);
        assert!(entry.timestamp >= first_timestamp);
    }

    #[tokio::test]
    async fn search_returns_ranked_matches_above_threshold() {
        let store = store_with_embeddings(HashMap::from([
            ("rust memory safety".to_string(), vec![1.0, 0.0, 0.0]),
            ("python scripting".to_string(), vec![0.0, 1.0, 0.0]),
            ("rust borrow checker".to_string(), vec![0.9, 0.1, 0.0]),
            ("rust query".to_string(), vec![1.0, 0.0, 0.0]),
        ]));

        store
            .store("a", "rust memory safety", HashMap::new())
            .await
            .unwrap();
        store
            .store("b", "python scripting", HashMap::new())
            .await
            .unwrap();
        store
            .store("c", "rust borrow checker", HashMap::new())
            .await
            .unwrap();

        let results = store.search("rust query", 10).await.unwrap();

        assert_eq!(results.len(), 2);
        assert_eq!(results[0].id, "a");
        assert_eq!(results[1].id, "c");
    }

    #[tokio::test]
    async fn search_applies_requested_limit_and_config_cap() {
        let store = store_with_embeddings(HashMap::from([
            ("entry one".to_string(), vec![1.0, 0.0, 0.0]),
            ("entry two".to_string(), vec![0.95, 0.05, 0.0]),
            ("entry three".to_string(), vec![0.9, 0.1, 0.0]),
            ("entry four".to_string(), vec![0.85, 0.15, 0.0]),
            ("query".to_string(), vec![1.0, 0.0, 0.0]),
        ]));

        for id in ["1", "2", "3", "4"] {
            let content = match id {
                "1" => "entry one",
                "2" => "entry two",
                "3" => "entry three",
                _ => "entry four",
            };
            store.store(id, content, HashMap::new()).await.unwrap();
        }

        let results = store.search("query", 10).await.unwrap();
        assert_eq!(results.len(), 3);

        let limited_results = store.search("query", 2).await.unwrap();
        assert_eq!(limited_results.len(), 2);
    }

    #[tokio::test]
    async fn search_returns_empty_when_nothing_meets_threshold() {
        let store = store_with_embeddings(HashMap::from([
            ("distant topic".to_string(), vec![0.0, 1.0, 0.0]),
            ("query".to_string(), vec![1.0, 0.0, 0.0]),
        ]));

        store
            .store("entry", "distant topic", HashMap::new())
            .await
            .unwrap();

        let results = store.search("query", 5).await.unwrap();
        assert!(results.is_empty());
    }

    #[tokio::test]
    async fn delete_removes_entry() {
        let store = store_with_embeddings(HashMap::from([(
            "knowledge".to_string(),
            vec![1.0, 0.0, 0.0],
        )]));

        store
            .store("entry", "knowledge", HashMap::new())
            .await
            .unwrap();

        assert!(store.delete("entry").await.unwrap());
        assert!(!store.delete("entry").await.unwrap());
        assert!(store.get("entry").await.unwrap().is_none());
    }

    #[tokio::test]
    async fn store_propagates_embedder_failures() {
        let embedder = Arc::new(MockEmbedder::new(HashMap::new()).with_failure("broken content"));
        let store = InMemoryLongTermStore::new(embedder, config());

        let error = store
            .store("entry", "broken content", HashMap::new())
            .await
            .unwrap_err();

        assert!(matches!(error, MemoryError::Storage(_)));
        assert!(error.to_string().contains("embedding generation failed"));
    }

    #[tokio::test]
    async fn store_rejects_wrong_embedding_dimension() {
        let store = store_with_embeddings(HashMap::from([(
            "short embedding".to_string(),
            vec![1.0, 0.0],
        )]));

        let error = store
            .store("entry", "short embedding", HashMap::new())
            .await
            .unwrap_err();

        assert!(matches!(error, MemoryError::Storage(_)));
        assert!(error.to_string().contains("invalid embedding dimension"));
    }

    #[tokio::test]
    async fn search_handles_zero_norm_embeddings() {
        let store = store_with_embeddings(HashMap::from([
            ("empty semantic vector".to_string(), vec![0.0, 0.0, 0.0]),
            ("query".to_string(), vec![1.0, 0.0, 0.0]),
        ]));

        store
            .store("entry", "empty semantic vector", HashMap::new())
            .await
            .unwrap();

        let results = store.search("query", 5).await.unwrap();
        assert!(results.is_empty());
    }
}
