//! Vector store abstractions and in-memory backend.
//!
//! The [`VectorStore`] trait defines a backend-agnostic API that can be backed by
//! in-memory storage, Qdrant, pgvector, or other vector databases. Networked
//! backends are expected to manage connection pooling internally and use
//! [`VectorStore::health_check`] to validate pool connectivity.

use std::{
    cmp::Ordering as CmpOrdering,
    collections::HashMap,
    sync::{
        atomic::{AtomicBool, Ordering as AtomicOrdering},
        Arc,
    },
};

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use thiserror::Error;
use tokio::sync::RwLock;
use tracing::{debug, instrument};

/// A document stored in a vector index.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct VectorDocument {
    /// Unique document identifier.
    pub id: String,
    /// Original document or chunk text.
    pub content: String,
    /// Vector embedding used for similarity search.
    pub embedding: Vec<f32>,
    /// Arbitrary document metadata used for filtering.
    pub metadata: HashMap<String, Value>,
}

/// A similarity search request.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct VectorQuery {
    /// Query embedding.
    pub embedding: Vec<f32>,
    /// Maximum number of results to return.
    pub top_k: usize,
    /// Optional exact-match metadata filter.
    pub filter: Option<HashMap<String, Value>>,
    /// Optional minimum cosine similarity threshold.
    pub min_score: Option<f32>,
}

/// A similarity search match returned by a vector store.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct VectorSearchResult {
    /// Matched document identifier.
    pub id: String,
    /// Matched document content.
    pub content: String,
    /// Cosine similarity score.
    pub score: f32,
    /// Metadata associated with the matched document.
    pub metadata: HashMap<String, Value>,
}

/// Errors produced by vector store operations.
#[derive(Debug, Error)]
pub enum VectorStoreError {
    /// Returned when a document embedding is empty.
    #[error("document `{id}` has an empty embedding")]
    EmptyDocumentEmbedding { id: String },
    /// Returned when the query embedding is empty.
    #[error("query embedding cannot be empty")]
    EmptyQueryEmbedding,
    /// Returned when embeddings with inconsistent dimensionality are used.
    #[error("embedding dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch { expected: usize, actual: usize },
    /// Returned when the backend is unavailable or unhealthy.
    #[error("vector store backend is unavailable")]
    Unavailable,
    /// Returned for backend-specific failures.
    #[error("vector store backend error: {0}")]
    Backend(String),
}

/// Backend-agnostic vector store abstraction.
#[async_trait]
pub trait VectorStore: Send + Sync {
    /// Insert or update documents in the vector store.
    async fn upsert(&self, documents: Vec<VectorDocument>) -> Result<(), VectorStoreError>;

    /// Execute a similarity query against the vector store.
    async fn query(&self, query: VectorQuery) -> Result<Vec<VectorSearchResult>, VectorStoreError>;

    /// Delete documents by identifier.
    async fn delete(&self, ids: Vec<String>) -> Result<(), VectorStoreError>;

    /// Validate that the backend and its connection pool are healthy.
    async fn health_check(&self) -> Result<(), VectorStoreError>;
}

/// In-memory [`VectorStore`] implementation backed by a Tokio [`RwLock`].
#[derive(Debug, Clone, Default)]
pub struct InMemoryVectorStore {
    documents: Arc<RwLock<HashMap<String, VectorDocument>>>,
    dimension: Arc<RwLock<Option<usize>>>,
    healthy: Arc<AtomicBool>,
}

impl InMemoryVectorStore {
    /// Create a new empty in-memory vector store.
    pub fn new() -> Self {
        Self {
            documents: Arc::new(RwLock::new(HashMap::new())),
            dimension: Arc::new(RwLock::new(None)),
            healthy: Arc::new(AtomicBool::new(true)),
        }
    }

    async fn validate_document(&self, document: &VectorDocument) -> Result<(), VectorStoreError> {
        if document.embedding.is_empty() {
            return Err(VectorStoreError::EmptyDocumentEmbedding {
                id: document.id.clone(),
            });
        }

        self.validate_dimension(document.embedding.len()).await
    }

    async fn validate_query(&self, query: &VectorQuery) -> Result<(), VectorStoreError> {
        if query.embedding.is_empty() {
            return Err(VectorStoreError::EmptyQueryEmbedding);
        }

        self.validate_dimension(query.embedding.len()).await
    }

    async fn validate_dimension(&self, dimension: usize) -> Result<(), VectorStoreError> {
        let expected = *self.dimension.read().await;

        if let Some(expected) = expected {
            if expected != dimension {
                return Err(VectorStoreError::DimensionMismatch {
                    expected,
                    actual: dimension,
                });
            }
        }

        Ok(())
    }

    fn matches_filter(
        metadata: &HashMap<String, Value>,
        filter: Option<&HashMap<String, Value>>,
    ) -> bool {
        filter.is_none_or(|filter| {
            filter
                .iter()
                .all(|(key, value)| metadata.get(key) == Some(value))
        })
    }

    fn cosine_similarity(left: &[f32], right: &[f32]) -> f32 {
        let dot_product = left
            .iter()
            .zip(right.iter())
            .map(|(a, b)| a * b)
            .sum::<f32>();
        let left_norm = left.iter().map(|value| value * value).sum::<f32>().sqrt();
        let right_norm = right.iter().map(|value| value * value).sum::<f32>().sqrt();

        if left_norm == 0.0 || right_norm == 0.0 {
            return 0.0;
        }

        dot_product / (left_norm * right_norm)
    }

    #[cfg(test)]
    fn set_healthy(&self, healthy: bool) {
        self.healthy.store(healthy, AtomicOrdering::SeqCst);
    }
}

#[async_trait]
impl VectorStore for InMemoryVectorStore {
    #[instrument(skip(self, documents), fields(document_count = documents.len()))]
    async fn upsert(&self, documents: Vec<VectorDocument>) -> Result<(), VectorStoreError> {
        self.health_check().await?;

        if documents.is_empty() {
            return Ok(());
        }

        for document in &documents {
            self.validate_document(document).await?;
        }

        let mut dimension = self.dimension.write().await;
        if dimension.is_none() {
            *dimension = Some(documents[0].embedding.len());
        }
        drop(dimension);

        let mut stored_documents = self.documents.write().await;
        for document in documents {
            debug!(document_id = %document.id, "upserting vector document");
            stored_documents.insert(document.id.clone(), document);
        }

        Ok(())
    }

    #[instrument(skip(self, query), fields(top_k = query.top_k))]
    async fn query(&self, query: VectorQuery) -> Result<Vec<VectorSearchResult>, VectorStoreError> {
        self.health_check().await?;
        self.validate_query(&query).await?;

        if query.top_k == 0 {
            return Ok(Vec::new());
        }

        let documents = self.documents.read().await;
        let mut results: Vec<_> = documents
            .values()
            .filter(|document| Self::matches_filter(&document.metadata, query.filter.as_ref()))
            .filter_map(|document| {
                let score = Self::cosine_similarity(&query.embedding, &document.embedding);
                if query.min_score.is_some_and(|min_score| score < min_score) {
                    return None;
                }

                Some(VectorSearchResult {
                    id: document.id.clone(),
                    content: document.content.clone(),
                    score,
                    metadata: document.metadata.clone(),
                })
            })
            .collect();

        results.sort_by(|left, right| {
            right
                .score
                .partial_cmp(&left.score)
                .unwrap_or(CmpOrdering::Equal)
                .then_with(|| left.id.cmp(&right.id))
        });
        results.truncate(query.top_k);

        Ok(results)
    }

    #[instrument(skip(self, ids), fields(document_count = ids.len()))]
    async fn delete(&self, ids: Vec<String>) -> Result<(), VectorStoreError> {
        self.health_check().await?;

        if ids.is_empty() {
            return Ok(());
        }

        let mut documents = self.documents.write().await;
        for id in ids {
            debug!(document_id = %id, "deleting vector document");
            documents.remove(&id);
        }

        if documents.is_empty() {
            *self.dimension.write().await = None;
        }

        Ok(())
    }

    async fn health_check(&self) -> Result<(), VectorStoreError> {
        if self.healthy.load(AtomicOrdering::SeqCst) {
            Ok(())
        } else {
            Err(VectorStoreError::Unavailable)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn document(id: &str, content: &str, embedding: Vec<f32>) -> VectorDocument {
        VectorDocument {
            id: id.to_string(),
            content: content.to_string(),
            embedding,
            metadata: HashMap::new(),
        }
    }

    #[tokio::test]
    async fn upsert_and_query_returns_ranked_results() {
        let store = InMemoryVectorStore::new();

        store
            .upsert(vec![
                document("doc-1", "Rust async runtime", vec![1.0, 0.0]),
                document("doc-2", "Vector database", vec![0.8, 0.2]),
                document("doc-3", "Relational database", vec![0.0, 1.0]),
            ])
            .await
            .unwrap();

        let results = store
            .query(VectorQuery {
                embedding: vec![1.0, 0.0],
                top_k: 2,
                filter: None,
                min_score: None,
            })
            .await
            .unwrap();

        assert_eq!(results.len(), 2);
        assert_eq!(results[0].id, "doc-1");
        assert_eq!(results[0].content, "Rust async runtime");
        assert!((results[0].score - 1.0).abs() < 1e-6);
        assert_eq!(results[1].id, "doc-2");
        assert!(results[1].score < results[0].score);
    }

    #[tokio::test]
    async fn upsert_overwrites_existing_document() {
        let store = InMemoryVectorStore::new();

        store
            .upsert(vec![document("doc-1", "old", vec![1.0, 0.0])])
            .await
            .unwrap();
        store
            .upsert(vec![document("doc-1", "new", vec![0.0, 1.0])])
            .await
            .unwrap();

        let results = store
            .query(VectorQuery {
                embedding: vec![0.0, 1.0],
                top_k: 1,
                filter: None,
                min_score: None,
            })
            .await
            .unwrap();

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, "doc-1");
        assert_eq!(results[0].content, "new");
        assert!((results[0].score - 1.0).abs() < 1e-6);
    }

    #[tokio::test]
    async fn query_applies_filter_and_min_score() {
        let store = InMemoryVectorStore::new();
        let mut rust_metadata = HashMap::new();
        rust_metadata.insert("topic".to_string(), json!("rust"));
        let mut db_metadata = HashMap::new();
        db_metadata.insert("topic".to_string(), json!("database"));

        store
            .upsert(vec![
                VectorDocument {
                    id: "doc-1".to_string(),
                    content: "Rust traits".to_string(),
                    embedding: vec![1.0, 0.0],
                    metadata: rust_metadata.clone(),
                },
                VectorDocument {
                    id: "doc-2".to_string(),
                    content: "Database indexing".to_string(),
                    embedding: vec![0.7, 0.3],
                    metadata: db_metadata,
                },
            ])
            .await
            .unwrap();

        let results = store
            .query(VectorQuery {
                embedding: vec![1.0, 0.0],
                top_k: 5,
                filter: Some(rust_metadata),
                min_score: Some(0.9),
            })
            .await
            .unwrap();

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, "doc-1");
    }

    #[tokio::test]
    async fn delete_removes_documents_and_resets_dimension_when_empty() {
        let store = InMemoryVectorStore::new();

        store
            .upsert(vec![document("doc-1", "one", vec![1.0, 0.0])])
            .await
            .unwrap();
        store.delete(vec!["doc-1".to_string()]).await.unwrap();

        let results = store
            .query(VectorQuery {
                embedding: vec![1.0, 0.0],
                top_k: 1,
                filter: None,
                min_score: None,
            })
            .await
            .unwrap();
        assert!(results.is_empty());

        store
            .upsert(vec![document("doc-2", "two", vec![1.0, 0.0, 0.0])])
            .await
            .unwrap();

        let results = store
            .query(VectorQuery {
                embedding: vec![1.0, 0.0, 0.0],
                top_k: 1,
                filter: None,
                min_score: None,
            })
            .await
            .unwrap();
        assert_eq!(results[0].id, "doc-2");
    }

    #[tokio::test]
    async fn rejects_dimension_mismatch_for_upsert_and_query() {
        let store = InMemoryVectorStore::new();

        store
            .upsert(vec![document("doc-1", "one", vec![1.0, 0.0])])
            .await
            .unwrap();

        let upsert_error = store
            .upsert(vec![document("doc-2", "two", vec![1.0, 0.0, 0.0])])
            .await
            .unwrap_err();
        assert!(matches!(
            upsert_error,
            VectorStoreError::DimensionMismatch {
                expected: 2,
                actual: 3
            }
        ));

        let query_error = store
            .query(VectorQuery {
                embedding: vec![1.0, 0.0, 0.0],
                top_k: 1,
                filter: None,
                min_score: None,
            })
            .await
            .unwrap_err();
        assert!(matches!(
            query_error,
            VectorStoreError::DimensionMismatch {
                expected: 2,
                actual: 3
            }
        ));
    }

    #[tokio::test]
    async fn rejects_empty_embeddings() {
        let store = InMemoryVectorStore::new();

        let upsert_error = store
            .upsert(vec![document("doc-1", "one", vec![])])
            .await
            .unwrap_err();
        assert!(matches!(
            upsert_error,
            VectorStoreError::EmptyDocumentEmbedding { .. }
        ));

        let query_error = store
            .query(VectorQuery {
                embedding: vec![],
                top_k: 1,
                filter: None,
                min_score: None,
            })
            .await
            .unwrap_err();
        assert!(matches!(query_error, VectorStoreError::EmptyQueryEmbedding));
    }

    #[tokio::test]
    async fn health_check_reflects_backend_state() {
        let store = InMemoryVectorStore::new();
        assert!(store.health_check().await.is_ok());

        store.set_healthy(false);
        assert!(matches!(
            store.health_check().await,
            Err(VectorStoreError::Unavailable)
        ));
    }

    #[tokio::test]
    async fn query_with_top_k_zero_returns_no_results() {
        let store = InMemoryVectorStore::new();
        store
            .upsert(vec![document("doc-1", "one", vec![1.0, 0.0])])
            .await
            .unwrap();

        let results = store
            .query(VectorQuery {
                embedding: vec![1.0, 0.0],
                top_k: 0,
                filter: None,
                min_score: None,
            })
            .await
            .unwrap();

        assert!(results.is_empty());
    }
}
