//! Knowledge-base builder and query API for RAG.

use std::{collections::HashMap, sync::Arc};

use ai_core::{Embedder, EmbedderError};
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::{
    chunker::ChunkerConfig,
    ingestion::{
        ChunkingStrategy, Document, IngestionConfig, IngestionError, IngestionPipeline,
        IngestionResult,
    },
    vector_store::{VectorQuery, VectorSearchResult, VectorStore, VectorStoreError},
};

/// Configuration for a knowledge base ingestion pipeline.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct KnowledgeBaseConfig {
    /// Chunking strategy applied during ingestion.
    #[serde(default)]
    pub chunking_strategy: ChunkingStrategy,
    /// Shared configuration passed to the configured chunker.
    #[serde(default)]
    pub chunker_config: ChunkerConfig,
    /// Number of chunks to embed per batch.
    pub batch_size: usize,
}

impl Default for KnowledgeBaseConfig {
    fn default() -> Self {
        Self {
            chunking_strategy: ChunkingStrategy::default(),
            chunker_config: ChunkerConfig::default(),
            batch_size: 16,
        }
    }
}

impl From<KnowledgeBaseConfig> for IngestionConfig {
    fn from(value: KnowledgeBaseConfig) -> Self {
        Self {
            batch_size: value.batch_size,
            chunking_strategy: value.chunking_strategy,
            chunker_config: value.chunker_config,
        }
    }
}

/// Errors produced by knowledge-base operations.
#[derive(Debug, Error)]
pub enum KnowledgeBaseError {
    /// Document ingestion failure.
    #[error(transparent)]
    Ingestion(#[from] IngestionError),
    /// Query embedding generation failure.
    #[error(transparent)]
    Embedder(#[from] EmbedderError),
    /// Vector-store failure.
    #[error(transparent)]
    VectorStore(#[from] VectorStoreError),
    /// Returned when the knowledge base has not been built.
    #[error("knowledge base has not been built")]
    NotBuilt,
    /// Returned when the query text is empty or whitespace only.
    #[error("query text cannot be empty")]
    EmptyQuery,
}

/// Builder for constructing a [`KnowledgeBase`] from source documents.
#[derive(Debug, Clone)]
pub struct KnowledgeBaseBuilder<E, V>
where
    E: Embedder + 'static,
    V: VectorStore + 'static,
{
    config: KnowledgeBaseConfig,
    embedder: Arc<E>,
    vector_store: Arc<V>,
    documents: HashMap<String, Document>,
}

impl<E, V> KnowledgeBaseBuilder<E, V>
where
    E: Embedder + 'static,
    V: VectorStore + 'static,
{
    /// Create a new knowledge-base builder.
    pub fn new(embedder: Arc<E>, vector_store: Arc<V>) -> Self {
        Self {
            config: KnowledgeBaseConfig::default(),
            embedder,
            vector_store,
            documents: HashMap::new(),
        }
    }

    /// Override the knowledge-base configuration.
    pub fn with_config(mut self, config: KnowledgeBaseConfig) -> Self {
        self.config = config;
        self
    }

    /// Queue a document for ingestion when the knowledge base is built.
    pub fn add_document(mut self, document: Document) -> Self {
        self.documents.insert(document.id.clone(), document);
        self
    }

    /// Build the knowledge base and ingest all queued documents.
    pub async fn build(self) -> Result<KnowledgeBase<E, V>, KnowledgeBaseError> {
        let ingestion_pipeline = IngestionPipeline::new(
            self.config.clone().into(),
            self.embedder.clone(),
            self.vector_store.clone(),
        );
        let mut document_chunks = HashMap::with_capacity(self.documents.len());

        for document in self.documents.into_values() {
            let result = ingestion_pipeline.ingest(document).await?;
            document_chunks.insert(result.document_id, result.chunk_ids);
        }

        Ok(KnowledgeBase {
            embedder: self.embedder,
            vector_store: self.vector_store,
            ingestion_pipeline,
            document_chunks,
        })
    }
}

/// Queryable knowledge base backed by an embedder and vector store.
#[derive(Clone)]
pub struct KnowledgeBase<E, V>
where
    E: Embedder + 'static,
    V: VectorStore + 'static,
{
    embedder: Arc<E>,
    vector_store: Arc<V>,
    ingestion_pipeline: IngestionPipeline,
    document_chunks: HashMap<String, Vec<String>>,
}

impl<E, V> std::fmt::Debug for KnowledgeBase<E, V>
where
    E: Embedder + 'static,
    V: VectorStore + 'static,
{
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("KnowledgeBase")
            .field("embedder", &self.embedder.name())
            .field("document_count", &self.document_chunks.len())
            .finish_non_exhaustive()
    }
}

impl<E, V> KnowledgeBase<E, V>
where
    E: Embedder + 'static,
    V: VectorStore + 'static,
{
    /// Embed a query and search the vector store.
    pub async fn query(
        &self,
        query_text: impl AsRef<str>,
        top_k: usize,
    ) -> Result<Vec<VectorSearchResult>, KnowledgeBaseError> {
        let query_text = query_text.as_ref().trim();
        if query_text.is_empty() {
            return Err(KnowledgeBaseError::EmptyQuery);
        }
        if top_k == 0 {
            return Ok(Vec::new());
        }

        let embeddings = self.embedder.embed(vec![query_text.to_string()]).await?;
        if embeddings.len() != 1 {
            return Err(KnowledgeBaseError::Embedder(EmbedderError::Embedding(
                format!(
                    "expected exactly one query embedding, got {}",
                    embeddings.len()
                ),
            )));
        }

        let embedding = embeddings.into_iter().next().unwrap();
        if embedding.is_empty() {
            return Err(KnowledgeBaseError::Embedder(EmbedderError::Embedding(
                "query embedding cannot be empty".to_string(),
            )));
        }

        Ok(self
            .vector_store
            .query(VectorQuery {
                embedding,
                top_k,
                filter: None,
                min_score: None,
            })
            .await?)
    }

    /// Ingest a document into the knowledge base.
    pub async fn add_document(
        &mut self,
        document: Document,
    ) -> Result<IngestionResult, KnowledgeBaseError> {
        let document_id = document.id.clone();
        self.remove_document(&document_id).await?;

        let result = self.ingestion_pipeline.ingest(document).await?;
        self.document_chunks
            .insert(result.document_id.clone(), result.chunk_ids.clone());

        Ok(result)
    }

    /// Remove an ingested document and all indexed chunks.
    pub async fn remove_document(&mut self, document_id: &str) -> Result<(), KnowledgeBaseError> {
        if let Some(chunk_ids) = self.document_chunks.remove(document_id) {
            if !chunk_ids.is_empty() {
                self.vector_store.delete(chunk_ids).await?;
            }
        }

        Ok(())
    }

    /// Return the number of tracked source documents in the knowledge base.
    pub fn document_count(&self) -> usize {
        self.document_chunks.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    use async_trait::async_trait;
    use serde_json::Value;

    use crate::{ingestion::DocumentFormat, vector_store::InMemoryVectorStore};

    #[derive(Debug, Default)]
    struct MockEmbedder;

    fn tokenize(text: &str) -> HashSet<String> {
        text.split(|ch: char| !ch.is_alphanumeric())
            .filter(|token| !token.is_empty())
            .map(|token| token.to_ascii_lowercase())
            .collect()
    }

    #[async_trait]
    impl Embedder for MockEmbedder {
        async fn embed(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>, EmbedderError> {
            Ok(texts
                .into_iter()
                .map(|text| {
                    let tokens = tokenize(&text);
                    vec![
                        if tokens.contains("rust") { 1.0 } else { 0.0 },
                        if tokens.contains("async") { 1.0 } else { 0.0 },
                        if tokens.contains("vector") { 1.0 } else { 0.0 },
                        if tokens.contains("database") {
                            1.0
                        } else {
                            0.0
                        },
                        text.len() as f32,
                    ]
                })
                .collect())
        }

        fn name(&self) -> &str {
            "mock_knowledge_base_embedder"
        }
    }

    fn document(id: &str, content: &str) -> Document {
        Document {
            id: id.to_string(),
            content: content.to_string(),
            format: DocumentFormat::PlainText,
            metadata: HashMap::new(),
        }
    }

    #[tokio::test]
    async fn builds_knowledge_base_with_multiple_documents_and_queries() {
        let embedder = Arc::new(MockEmbedder);
        let vector_store = Arc::new(InMemoryVectorStore::new());

        let knowledge_base = KnowledgeBaseBuilder::new(embedder, vector_store)
            .with_config(KnowledgeBaseConfig {
                chunking_strategy: ChunkingStrategy::FixedSize,
                chunker_config: ChunkerConfig {
                    chunk_size: 256,
                    overlap: 0,
                    min_chunk_size: 1,
                },
                batch_size: 2,
            })
            .add_document(document("doc-rust", "Rust async programming with tokio."))
            .add_document(document(
                "doc-db",
                "Vector database indexing for retrieval systems.",
            ))
            .build()
            .await
            .unwrap();

        assert_eq!(knowledge_base.document_count(), 2);

        let results = knowledge_base.query("rust async", 2).await.unwrap();

        assert_eq!(results.len(), 2);
        assert_eq!(
            results[0].metadata.get("source_document_id"),
            Some(&Value::String("doc-rust".to_string()))
        );
        assert!(results[0].content.contains("Rust async"));
    }

    #[tokio::test]
    async fn supports_incremental_add_and_remove_documents() {
        let embedder = Arc::new(MockEmbedder);
        let vector_store = Arc::new(InMemoryVectorStore::new());

        let mut knowledge_base = KnowledgeBaseBuilder::new(embedder, vector_store)
            .build()
            .await
            .unwrap();

        assert_eq!(knowledge_base.document_count(), 0);

        let ingestion_result = knowledge_base
            .add_document(document(
                "doc-rust",
                "Rust powers async services and reliable systems.",
            ))
            .await
            .unwrap();

        assert_eq!(knowledge_base.document_count(), 1);
        assert_eq!(ingestion_result.document_id, "doc-rust");
        assert!(!ingestion_result.chunk_ids.is_empty());

        let results = knowledge_base.query("rust", 1).await.unwrap();
        assert_eq!(
            results[0].metadata.get("source_document_id"),
            Some(&Value::String("doc-rust".to_string()))
        );

        knowledge_base.remove_document("doc-rust").await.unwrap();

        assert_eq!(knowledge_base.document_count(), 0);
        assert!(knowledge_base.query("rust", 5).await.unwrap().is_empty());
    }

    #[tokio::test]
    async fn query_with_empty_text_returns_error() {
        let embedder = Arc::new(MockEmbedder);
        let vector_store = Arc::new(InMemoryVectorStore::new());
        let knowledge_base = KnowledgeBaseBuilder::new(embedder, vector_store)
            .build()
            .await
            .unwrap();

        let error = knowledge_base.query("   ", 1).await.unwrap_err();
        assert!(matches!(error, KnowledgeBaseError::EmptyQuery));
    }

    #[tokio::test]
    async fn builder_without_documents_creates_empty_knowledge_base() {
        let embedder = Arc::new(MockEmbedder);
        let vector_store = Arc::new(InMemoryVectorStore::new());
        let knowledge_base = KnowledgeBaseBuilder::new(embedder, vector_store)
            .build()
            .await
            .unwrap();

        assert_eq!(knowledge_base.document_count(), 0);
        assert!(knowledge_base.query("rust", 3).await.unwrap().is_empty());
    }
}
