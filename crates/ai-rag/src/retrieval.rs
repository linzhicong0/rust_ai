//! Retrieval strategies for semantic, keyword, hybrid, and reranked search.

use std::{
    cmp::Ordering,
    collections::{HashMap, HashSet},
    sync::Arc,
};

use ai_core::{Embedder, EmbedderError};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use thiserror::Error;
use tracing::instrument;

use crate::vector_store::{VectorQuery, VectorSearchResult, VectorStore, VectorStoreError};

/// A single retrieval match returned by a retriever.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RetrievalResult {
    /// Matched document identifier.
    pub id: String,
    /// Matched document content.
    pub content: String,
    /// Retriever score for this match.
    pub score: f32,
    /// Arbitrary metadata associated with the match.
    pub metadata: HashMap<String, Value>,
}

impl From<VectorSearchResult> for RetrievalResult {
    fn from(value: VectorSearchResult) -> Self {
        Self {
            id: value.id,
            content: value.content,
            score: value.score,
            metadata: value.metadata,
        }
    }
}

/// Errors produced by retrieval strategies.
#[derive(Debug, Error)]
pub enum RetrievalError {
    /// Returned when the query is empty or whitespace only.
    #[error("retrieval query cannot be empty")]
    EmptyQuery,
    /// Returned when a retriever is configured with invalid parameters.
    #[error("invalid retrieval configuration: {0}")]
    InvalidConfiguration(String),
    /// Returned when the embedder fails.
    #[error("embedder error: {0}")]
    Embedder(#[from] EmbedderError),
    /// Returned when the vector store fails.
    #[error("vector store error: {0}")]
    VectorStore(#[from] VectorStoreError),
    /// Returned when the embedder returns an unexpected number of embeddings.
    #[error("expected exactly one query embedding, got {actual}")]
    InvalidEmbeddingCount { actual: usize },
    /// Returned when the embedder returns an empty embedding.
    #[error("query embedding cannot be empty")]
    EmptyEmbedding,
}

/// Common interface for pluggable retrieval strategies.
#[async_trait]
pub trait Retriever: Send + Sync {
    /// Retrieve the top ranked results for a query.
    async fn retrieve(
        &self,
        query: &str,
        top_k: usize,
    ) -> Result<Vec<RetrievalResult>, RetrievalError>;
}

/// Optional re-ranking step applied after retrieval.
#[async_trait]
pub trait ReRanker: Send + Sync {
    /// Re-rank a candidate set for the given query.
    async fn rerank(
        &self,
        query: &str,
        results: Vec<RetrievalResult>,
    ) -> Result<Vec<RetrievalResult>, RetrievalError>;
}

/// Semantic retriever backed by an [`Embedder`] and [`VectorStore`].
#[derive(Clone)]
pub struct SemanticRetriever {
    embedder: Arc<dyn Embedder>,
    vector_store: Arc<dyn VectorStore>,
    min_score: Option<f32>,
}

impl std::fmt::Debug for SemanticRetriever {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("SemanticRetriever")
            .field("embedder", &self.embedder.name())
            .field("min_score", &self.min_score)
            .finish_non_exhaustive()
    }
}

impl SemanticRetriever {
    /// Create a semantic retriever.
    pub fn new(embedder: Arc<dyn Embedder>, vector_store: Arc<dyn VectorStore>) -> Self {
        Self {
            embedder,
            vector_store,
            min_score: None,
        }
    }

    /// Configure a minimum similarity score.
    pub fn with_min_score(mut self, min_score: f32) -> Self {
        self.min_score = Some(min_score);
        self
    }
}

#[async_trait]
impl Retriever for SemanticRetriever {
    #[instrument(skip(self, query), fields(top_k))]
    async fn retrieve(
        &self,
        query: &str,
        top_k: usize,
    ) -> Result<Vec<RetrievalResult>, RetrievalError> {
        if top_k == 0 {
            return Ok(Vec::new());
        }
        if query.trim().is_empty() {
            return Err(RetrievalError::EmptyQuery);
        }

        let embeddings = self.embedder.embed(vec![query.to_string()]).await?;
        if embeddings.len() != 1 {
            return Err(RetrievalError::InvalidEmbeddingCount {
                actual: embeddings.len(),
            });
        }

        let embedding = embeddings
            .into_iter()
            .next()
            .ok_or(RetrievalError::InvalidEmbeddingCount { actual: 0 })?;
        if embedding.is_empty() {
            return Err(RetrievalError::EmptyEmbedding);
        }

        let results = self
            .vector_store
            .query(VectorQuery {
                embedding,
                top_k,
                filter: None,
                min_score: self.min_score,
            })
            .await?;

        Ok(results.into_iter().map(Into::into).collect())
    }
}

/// In-memory BM25 retriever for sparse keyword-based search.
#[derive(Debug, Clone)]
pub struct BM25Retriever {
    documents: Vec<RetrievalResult>,
    term_frequencies: Vec<HashMap<String, usize>>,
    document_frequencies: HashMap<String, usize>,
    document_lengths: Vec<usize>,
    average_document_length: f32,
    k1: f32,
    b: f32,
}

impl BM25Retriever {
    /// Create a BM25 retriever with standard defaults.
    pub fn new(documents: Vec<RetrievalResult>) -> Self {
        Self::with_parameters(documents, 1.5, 0.75)
    }

    /// Create a BM25 retriever with explicit `k1` and `b` parameters.
    pub fn with_parameters(documents: Vec<RetrievalResult>, k1: f32, b: f32) -> Self {
        let mut term_frequencies = Vec::with_capacity(documents.len());
        let mut document_frequencies = HashMap::new();
        let mut document_lengths = Vec::with_capacity(documents.len());

        for document in &documents {
            let tokens = tokenize(&document.content);
            let mut frequencies = HashMap::new();
            let mut unique_terms = HashSet::new();

            for token in tokens {
                *frequencies.entry(token.clone()).or_insert(0) += 1;
                unique_terms.insert(token);
            }

            document_lengths.push(frequencies.values().sum());
            for term in unique_terms {
                *document_frequencies.entry(term).or_insert(0) += 1;
            }
            term_frequencies.push(frequencies);
        }

        let average_document_length = if documents.is_empty() {
            0.0
        } else {
            document_lengths.iter().sum::<usize>() as f32 / documents.len() as f32
        };

        Self {
            documents,
            term_frequencies,
            document_frequencies,
            document_lengths,
            average_document_length,
            k1,
            b,
        }
    }

    fn score_document(&self, query_terms: &[String], document_index: usize) -> f32 {
        if self.documents.is_empty() || self.average_document_length == 0.0 {
            return 0.0;
        }

        let frequencies = &self.term_frequencies[document_index];
        let document_length = self.document_lengths[document_index] as f32;
        let document_count = self.documents.len() as f32;
        let mut score = 0.0;

        for term in query_terms {
            let term_frequency = *frequencies.get(term).unwrap_or(&0) as f32;
            if term_frequency == 0.0 {
                continue;
            }

            let document_frequency = *self.document_frequencies.get(term).unwrap_or(&0) as f32;
            let idf = ((document_count - document_frequency + 0.5) / (document_frequency + 0.5)
                + 1.0)
                .ln();
            let normalization = term_frequency
                + self.k1
                    * (1.0 - self.b + self.b * (document_length / self.average_document_length));

            score += idf * (term_frequency * (self.k1 + 1.0)) / normalization;
        }

        score
    }
}

#[async_trait]
impl Retriever for BM25Retriever {
    #[instrument(skip(self, query), fields(top_k))]
    async fn retrieve(
        &self,
        query: &str,
        top_k: usize,
    ) -> Result<Vec<RetrievalResult>, RetrievalError> {
        if top_k == 0 {
            return Ok(Vec::new());
        }
        if query.trim().is_empty() {
            return Err(RetrievalError::EmptyQuery);
        }

        let query_terms = tokenize(query);
        let mut results: Vec<_> = self
            .documents
            .iter()
            .enumerate()
            .filter_map(|(index, document)| {
                let score = self.score_document(&query_terms, index);
                (score > 0.0).then(|| RetrievalResult {
                    id: document.id.clone(),
                    content: document.content.clone(),
                    score,
                    metadata: document.metadata.clone(),
                })
            })
            .collect();

        sort_results(&mut results);
        results.truncate(top_k);
        Ok(results)
    }
}

/// Hybrid retriever that linearly combines semantic and BM25 scores.
#[derive(Debug, Clone)]
pub struct HybridRetriever {
    semantic: SemanticRetriever,
    bm25: BM25Retriever,
    alpha: f32,
}

impl HybridRetriever {
    /// Create a hybrid retriever.
    pub fn new(
        semantic: SemanticRetriever,
        bm25: BM25Retriever,
        alpha: f32,
    ) -> Result<Self, RetrievalError> {
        if !(0.0..=1.0).contains(&alpha) {
            return Err(RetrievalError::InvalidConfiguration(
                "alpha must be between 0.0 and 1.0".to_string(),
            ));
        }

        Ok(Self {
            semantic,
            bm25,
            alpha,
        })
    }
}

#[async_trait]
impl Retriever for HybridRetriever {
    #[instrument(skip(self, query), fields(top_k, alpha = self.alpha))]
    async fn retrieve(
        &self,
        query: &str,
        top_k: usize,
    ) -> Result<Vec<RetrievalResult>, RetrievalError> {
        if top_k == 0 {
            return Ok(Vec::new());
        }

        let candidate_k = top_k.saturating_mul(2).max(top_k);
        let (semantic_results, bm25_results) = tokio::try_join!(
            self.semantic.retrieve(query, candidate_k),
            self.bm25.retrieve(query, candidate_k)
        )?;

        let semantic_scores = normalize_scores(&semantic_results);
        let bm25_scores = normalize_scores(&bm25_results);
        let mut merged = HashMap::<String, RetrievalResult>::new();

        for result in semantic_results.into_iter().chain(bm25_results.into_iter()) {
            merged.entry(result.id.clone()).or_insert(result);
        }

        let mut combined: Vec<_> = merged
            .into_values()
            .map(|mut result| {
                let semantic_score = semantic_scores.get(&result.id).copied().unwrap_or(0.0);
                let bm25_score = bm25_scores.get(&result.id).copied().unwrap_or(0.0);
                result.score = self.alpha * semantic_score + (1.0 - self.alpha) * bm25_score;
                result
            })
            .collect();

        sort_results(&mut combined);
        combined.truncate(top_k);
        Ok(combined)
    }
}

/// Simple re-ranker that boosts documents with keyword overlap.
#[derive(Debug, Clone)]
pub struct ScoreBasedReRanker {
    keyword_boost: f32,
}

impl Default for ScoreBasedReRanker {
    fn default() -> Self {
        Self {
            keyword_boost: 0.25,
        }
    }
}

impl ScoreBasedReRanker {
    /// Create a score-based re-ranker.
    pub fn new(keyword_boost: f32) -> Self {
        Self { keyword_boost }
    }
}

#[async_trait]
impl ReRanker for ScoreBasedReRanker {
    #[instrument(skip(self, query, results))]
    async fn rerank(
        &self,
        query: &str,
        mut results: Vec<RetrievalResult>,
    ) -> Result<Vec<RetrievalResult>, RetrievalError> {
        if query.trim().is_empty() {
            return Err(RetrievalError::EmptyQuery);
        }

        let query_terms: HashSet<_> = tokenize(query).into_iter().collect();
        if query_terms.is_empty() {
            return Ok(results);
        }

        for result in &mut results {
            let overlap = tokenize(&result.content)
                .into_iter()
                .filter(|token| query_terms.contains(token))
                .collect::<HashSet<_>>()
                .len();
            let overlap_ratio = overlap as f32 / query_terms.len() as f32;
            result.score += self.keyword_boost * overlap_ratio;
        }

        sort_results(&mut results);
        Ok(results)
    }
}

fn tokenize(text: &str) -> Vec<String> {
    text.split(|character: char| !character.is_alphanumeric())
        .filter(|token| !token.is_empty())
        .map(|token| token.to_lowercase())
        .collect()
}

fn normalize_scores(results: &[RetrievalResult]) -> HashMap<String, f32> {
    if results.is_empty() {
        return HashMap::new();
    }

    let min_score = results
        .iter()
        .map(|result| result.score)
        .fold(f32::INFINITY, f32::min);
    let max_score = results
        .iter()
        .map(|result| result.score)
        .fold(f32::NEG_INFINITY, f32::max);

    results
        .iter()
        .map(|result| {
            let normalized = if (max_score - min_score).abs() < f32::EPSILON {
                1.0
            } else {
                (result.score - min_score) / (max_score - min_score)
            };
            (result.id.clone(), normalized)
        })
        .collect()
}

fn sort_results(results: &mut [RetrievalResult]) {
    results.sort_by(|left, right| {
        right
            .score
            .partial_cmp(&left.score)
            .unwrap_or(Ordering::Equal)
            .then_with(|| left.id.cmp(&right.id))
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vector_store::{InMemoryVectorStore, VectorDocument, VectorStore};

    struct MockEmbedder {
        embeddings: HashMap<String, Vec<f32>>,
        fail: Option<EmbedderError>,
    }

    impl MockEmbedder {
        fn new(embeddings: HashMap<String, Vec<f32>>) -> Self {
            Self {
                embeddings,
                fail: None,
            }
        }

        fn failing(error: EmbedderError) -> Self {
            Self {
                embeddings: HashMap::new(),
                fail: Some(error),
            }
        }
    }

    #[async_trait]
    impl Embedder for MockEmbedder {
        async fn embed(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>, EmbedderError> {
            if let Some(error) = &self.fail {
                return Err(match error {
                    EmbedderError::Embedding(message) => EmbedderError::Embedding(message.clone()),
                    EmbedderError::Model(message) => EmbedderError::Model(message.clone()),
                });
            }

            Ok(texts
                .into_iter()
                .map(|text| self.embeddings.get(&text).cloned().unwrap_or_default())
                .collect())
        }

        fn name(&self) -> &str {
            "mock"
        }
    }

    fn result(id: &str, content: &str) -> RetrievalResult {
        RetrievalResult {
            id: id.to_string(),
            content: content.to_string(),
            score: 0.0,
            metadata: HashMap::new(),
        }
    }

    fn vector_document(id: &str, content: &str, embedding: Vec<f32>) -> VectorDocument {
        VectorDocument {
            id: id.to_string(),
            content: content.to_string(),
            embedding,
            metadata: HashMap::new(),
        }
    }

    #[tokio::test]
    async fn semantic_retriever_returns_vector_matches() {
        let store = Arc::new(InMemoryVectorStore::new());
        store
            .upsert(vec![
                vector_document("doc-1", "Rust async runtime", vec![1.0, 0.0]),
                vector_document("doc-2", "Database indexing", vec![0.0, 1.0]),
            ])
            .await
            .unwrap();

        let embedder = Arc::new(MockEmbedder::new(HashMap::from([(
            "rust async".to_string(),
            vec![1.0, 0.0],
        )])));
        let retriever = SemanticRetriever::new(embedder, store);

        let results = retriever.retrieve("rust async", 2).await.unwrap();

        assert_eq!(results.len(), 2);
        assert_eq!(results[0].id, "doc-1");
        assert!(results[0].score > results[1].score);
    }

    #[tokio::test]
    async fn semantic_retriever_propagates_embedder_errors() {
        let store = Arc::new(InMemoryVectorStore::new());
        let embedder = Arc::new(MockEmbedder::failing(EmbedderError::Embedding(
            "boom".to_string(),
        )));
        let retriever = SemanticRetriever::new(embedder, store);

        let error = retriever.retrieve("query", 1).await.unwrap_err();
        assert!(matches!(error, RetrievalError::Embedder(_)));
    }

    #[tokio::test]
    async fn semantic_retriever_rejects_empty_embedding() {
        let store = Arc::new(InMemoryVectorStore::new());
        let embedder = Arc::new(MockEmbedder::new(HashMap::new()));
        let retriever = SemanticRetriever::new(embedder, store);

        let error = retriever.retrieve("missing", 1).await.unwrap_err();
        assert!(matches!(error, RetrievalError::EmptyEmbedding));
    }

    #[tokio::test]
    async fn bm25_retriever_ranks_keyword_matches() {
        let retriever = BM25Retriever::new(vec![
            result("doc-1", "Rust async runtime and tasks"),
            result("doc-2", "Rust ownership and borrowing"),
            result("doc-3", "Database indexing strategies"),
        ]);

        let results = retriever.retrieve("rust async", 2).await.unwrap();

        assert_eq!(results.len(), 2);
        assert_eq!(results[0].id, "doc-1");
        assert_eq!(results[1].id, "doc-2");
        assert!(results[0].score > results[1].score);
    }

    #[tokio::test]
    async fn bm25_retriever_returns_empty_when_no_terms_match() {
        let retriever = BM25Retriever::new(vec![result("doc-1", "Rust async runtime")]);

        let results = retriever.retrieve("vector database", 5).await.unwrap();

        assert!(results.is_empty());
    }

    #[tokio::test]
    async fn hybrid_retriever_combines_sparse_and_dense_scores() {
        let store = Arc::new(InMemoryVectorStore::new());
        store
            .upsert(vec![
                vector_document("doc-1", "Rust guide", vec![0.0, 1.0]),
                vector_document("doc-2", "Systems handbook", vec![1.0, 0.0]),
            ])
            .await
            .unwrap();

        let embedder = Arc::new(MockEmbedder::new(HashMap::from([(
            "rust guide".to_string(),
            vec![1.0, 0.0],
        )])));
        let semantic = SemanticRetriever::new(embedder, store);
        let bm25 = BM25Retriever::new(vec![
            result("doc-1", "Rust guide"),
            result("doc-2", "Systems handbook"),
        ]);

        let keyword_heavy = HybridRetriever::new(semantic.clone(), bm25.clone(), 0.2).unwrap();
        let semantic_heavy = HybridRetriever::new(semantic, bm25, 0.8).unwrap();

        let keyword_results = keyword_heavy.retrieve("rust guide", 2).await.unwrap();
        let semantic_results = semantic_heavy.retrieve("rust guide", 2).await.unwrap();

        assert_eq!(keyword_results[0].id, "doc-1");
        assert_eq!(semantic_results[0].id, "doc-2");
    }

    #[tokio::test]
    async fn hybrid_retriever_validates_alpha() {
        let store = Arc::new(InMemoryVectorStore::new());
        let embedder = Arc::new(MockEmbedder::new(HashMap::new()));
        let semantic = SemanticRetriever::new(embedder, store);
        let bm25 = BM25Retriever::new(Vec::new());

        let error = HybridRetriever::new(semantic, bm25, 1.5).unwrap_err();
        assert!(matches!(error, RetrievalError::InvalidConfiguration(_)));
    }

    #[tokio::test]
    async fn score_based_reranker_boosts_keyword_overlap() {
        let reranker = ScoreBasedReRanker::new(0.5);
        let results = vec![
            RetrievalResult {
                id: "doc-1".to_string(),
                content: "Async runtime for Rust services".to_string(),
                score: 0.3,
                metadata: HashMap::new(),
            },
            RetrievalResult {
                id: "doc-2".to_string(),
                content: "Database indexing guide".to_string(),
                score: 0.3,
                metadata: HashMap::new(),
            },
        ];

        let reranked = reranker.rerank("rust async", results).await.unwrap();

        assert_eq!(reranked[0].id, "doc-1");
        assert!(reranked[0].score > reranked[1].score);
    }

    #[tokio::test]
    async fn retrievers_reject_empty_queries() {
        let bm25 = BM25Retriever::new(vec![result("doc-1", "Rust async runtime")]);
        assert!(matches!(
            bm25.retrieve("   ", 1).await,
            Err(RetrievalError::EmptyQuery)
        ));

        let reranker = ScoreBasedReRanker::default();
        assert!(matches!(
            reranker.rerank("", vec![result("doc-1", "Rust")]).await,
            Err(RetrievalError::EmptyQuery)
        ));
    }
}
