use std::cmp::Ordering;
use std::sync::{
    atomic::{AtomicU64, AtomicUsize, Ordering as AtomicOrdering},
    Arc,
};
use std::time::{SystemTime, UNIX_EPOCH};

use ai_core::{Embedder, EmbedderError};
use serde::{Deserialize, Serialize};
use thiserror::Error;
use tokio::sync::RwLock;
use tracing::debug;

/// Configuration for semantic caching.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticCacheConfig {
    /// Maximum number of cache entries retained at once.
    pub max_entries: usize,
    /// Minimum cosine similarity required for a cache hit.
    pub similarity_threshold: f32,
    /// Time-to-live in seconds for each cached entry.
    pub ttl_seconds: u64,
    /// Policy used when the cache reaches capacity.
    pub eviction_policy: EvictionPolicy,
}

impl Default for SemanticCacheConfig {
    fn default() -> Self {
        Self {
            max_entries: 1_024,
            similarity_threshold: 0.85,
            ttl_seconds: 300,
            eviction_policy: EvictionPolicy::Combined,
        }
    }
}

/// Eviction strategy applied when the semantic cache is full.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EvictionPolicy {
    /// Remove the least recently used entry.
    Lru,
    /// Remove the least frequently used entry.
    Lfu,
    /// Remove the stalest and least frequently used entry using a combined score.
    Combined,
}

/// Cached semantic response entry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CachedEntry {
    pub query_text: String,
    pub query_embedding: Vec<f32>,
    pub response: String,
    pub created_at: u64,
    pub last_accessed: u64,
    pub hit_count: u64,
}

/// Errors returned by the semantic cache.
#[derive(Debug, Error)]
pub enum SemanticCacheError {
    #[error("embedder error: {0}")]
    EmbedderError(#[from] EmbedderError),
    #[error("query cannot be empty")]
    EmptyQuery,
    #[error("semantic cache is full")]
    CacheFull,
    #[error("similarity threshold must be between 0.0 and 1.0 inclusive")]
    InvalidThreshold,
}

/// Runtime statistics for the semantic cache.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct CacheStats {
    pub total_entries: usize,
    pub hit_count: u64,
    pub miss_count: u64,
    pub eviction_count: u64,
}

#[derive(Debug, Default)]
struct SemanticCacheState {
    entries: Vec<CachedEntry>,
}

/// In-memory semantic cache that uses embeddings and cosine similarity.
#[derive(Clone)]
pub struct SemanticCache {
    config: SemanticCacheConfig,
    embedder: Arc<dyn Embedder>,
    inner: Arc<RwLock<SemanticCacheState>>,
    hit_count: Arc<AtomicU64>,
    miss_count: Arc<AtomicU64>,
    eviction_count: Arc<AtomicU64>,
    total_entries: Arc<AtomicUsize>,
    access_clock: Arc<AtomicU64>,
}

impl std::fmt::Debug for SemanticCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SemanticCache")
            .field("config", &self.config)
            .field("embedder", &self.embedder.name())
            .finish_non_exhaustive()
    }
}

impl SemanticCache {
    /// Create a new semantic cache instance.
    pub fn new(
        config: SemanticCacheConfig,
        embedder: Arc<dyn Embedder>,
    ) -> Result<Self, SemanticCacheError> {
        if !(0.0..=1.0).contains(&config.similarity_threshold) {
            return Err(SemanticCacheError::InvalidThreshold);
        }

        Ok(Self {
            config,
            embedder,
            inner: Arc::new(RwLock::new(SemanticCacheState::default())),
            hit_count: Arc::new(AtomicU64::new(0)),
            miss_count: Arc::new(AtomicU64::new(0)),
            eviction_count: Arc::new(AtomicU64::new(0)),
            total_entries: Arc::new(AtomicUsize::new(0)),
            access_clock: Arc::new(AtomicU64::new(1)),
        })
    }

    /// Return a semantically cached response when the similarity threshold is met.
    pub async fn get(&self, query: &str) -> Result<Option<String>, SemanticCacheError> {
        let query_embedding = self.embed_query(query).await?;
        let now = current_timestamp();
        let mut state = self.inner.write().await;

        self.prune_expired(&mut state, now);

        let best_match = state
            .entries
            .iter()
            .enumerate()
            .map(|(index, entry)| {
                (
                    index,
                    cosine_similarity(&query_embedding, &entry.query_embedding),
                )
            })
            .filter(|(_, similarity)| *similarity >= self.config.similarity_threshold)
            .max_by(|left, right| compare_similarity(left, right, &state.entries));

        if let Some((index, similarity)) = best_match {
            let entry = &mut state.entries[index];
            entry.last_accessed = self.next_access_tick();
            entry.hit_count += 1;
            self.hit_count.fetch_add(1, AtomicOrdering::Relaxed);
            debug!(query = query, similarity, "semantic cache hit");
            return Ok(Some(entry.response.clone()));
        }

        self.miss_count.fetch_add(1, AtomicOrdering::Relaxed);
        debug!(query = query, "semantic cache miss");
        Ok(None)
    }

    /// Insert a query/response pair into the semantic cache.
    pub async fn put(&self, query: &str, response: String) -> Result<(), SemanticCacheError> {
        let query_embedding = self.embed_query(query).await?;
        let now = current_timestamp();
        let mut state = self.inner.write().await;

        self.prune_expired(&mut state, now);

        if let Some(existing) = state
            .entries
            .iter_mut()
            .find(|entry| entry.query_text == query)
        {
            existing.query_text = query.to_string();
            existing.query_embedding = query_embedding;
            existing.response = response;
            existing.created_at = now;
            existing.last_accessed = self.next_access_tick();
            existing.hit_count = 0;
            return Ok(());
        }

        if state.entries.len() >= self.config.max_entries
            && !self.evict_one(&mut state, self.current_access_tick())
        {
            return Err(SemanticCacheError::CacheFull);
        }

        state.entries.push(CachedEntry {
            query_text: query.to_string(),
            query_embedding,
            response,
            created_at: now,
            last_accessed: self.next_access_tick(),
            hit_count: 0,
        });
        self.total_entries
            .store(state.entries.len(), AtomicOrdering::Relaxed);

        Ok(())
    }

    /// Remove the entry most semantically similar to the provided query.
    pub async fn invalidate(&self, query: &str) -> Result<bool, SemanticCacheError> {
        let query_embedding = self.embed_query(query).await?;
        let now = current_timestamp();
        let mut state = self.inner.write().await;

        self.prune_expired(&mut state, now);

        let best_match = state
            .entries
            .iter()
            .enumerate()
            .map(|(index, entry)| {
                (
                    index,
                    cosine_similarity(&query_embedding, &entry.query_embedding),
                )
            })
            .max_by(|left, right| compare_similarity(left, right, &state.entries))
            .map(|(index, _)| index);

        if let Some(index) = best_match {
            state.entries.swap_remove(index);
            self.total_entries
                .store(state.entries.len(), AtomicOrdering::Relaxed);
            return Ok(true);
        }

        Ok(false)
    }

    /// Return current cache statistics.
    pub fn stats(&self) -> CacheStats {
        CacheStats {
            total_entries: self.total_entries.load(AtomicOrdering::Relaxed),
            hit_count: self.hit_count.load(AtomicOrdering::Relaxed),
            miss_count: self.miss_count.load(AtomicOrdering::Relaxed),
            eviction_count: self.eviction_count.load(AtomicOrdering::Relaxed),
        }
    }

    /// Remove all cached entries.
    pub fn clear(&self) {
        loop {
            if let Ok(mut state) = self.inner.try_write() {
                state.entries.clear();
                self.total_entries.store(0, AtomicOrdering::Relaxed);
                return;
            }

            std::thread::yield_now();
        }
    }

    async fn embed_query(&self, query: &str) -> Result<Vec<f32>, SemanticCacheError> {
        let trimmed = query.trim();
        if trimmed.is_empty() {
            return Err(SemanticCacheError::EmptyQuery);
        }

        let embeddings = self.embedder.embed(vec![trimmed.to_string()]).await?;
        Ok(embeddings.into_iter().next().unwrap_or_default())
    }

    fn next_access_tick(&self) -> u64 {
        self.access_clock.fetch_add(1, AtomicOrdering::Relaxed)
    }

    fn current_access_tick(&self) -> u64 {
        self.access_clock.load(AtomicOrdering::Relaxed)
    }

    fn prune_expired(&self, state: &mut SemanticCacheState, now: u64) {
        let ttl_millis = self.config.ttl_seconds.saturating_mul(1_000);
        state
            .entries
            .retain(|entry| now.saturating_sub(entry.created_at) < ttl_millis);
        self.total_entries
            .store(state.entries.len(), AtomicOrdering::Relaxed);
    }

    fn evict_one(&self, state: &mut SemanticCacheState, access_tick: u64) -> bool {
        let candidate = match self.config.eviction_policy {
            EvictionPolicy::Lru => state
                .entries
                .iter()
                .enumerate()
                .min_by(|(_, left), (_, right)| {
                    left.last_accessed
                        .cmp(&right.last_accessed)
                        .then_with(|| left.created_at.cmp(&right.created_at))
                })
                .map(|(index, _)| index),
            EvictionPolicy::Lfu => state
                .entries
                .iter()
                .enumerate()
                .min_by(|(_, left), (_, right)| {
                    left.hit_count
                        .cmp(&right.hit_count)
                        .then_with(|| left.last_accessed.cmp(&right.last_accessed))
                        .then_with(|| left.created_at.cmp(&right.created_at))
                })
                .map(|(index, _)| index),
            EvictionPolicy::Combined => state
                .entries
                .iter()
                .enumerate()
                .max_by(|(_, left), (_, right)| {
                    combined_eviction_score(left, access_tick)
                        .partial_cmp(&combined_eviction_score(right, access_tick))
                        .unwrap_or(Ordering::Equal)
                        .then_with(|| left.last_accessed.cmp(&right.last_accessed).reverse())
                })
                .map(|(index, _)| index),
        };

        if let Some(index) = candidate {
            let evicted = state.entries.swap_remove(index);
            self.eviction_count.fetch_add(1, AtomicOrdering::Relaxed);
            self.total_entries
                .store(state.entries.len(), AtomicOrdering::Relaxed);
            debug!(query = evicted.query_text, "semantic cache entry evicted");
            return true;
        }

        false
    }
}

fn compare_similarity(
    left: &(usize, f32),
    right: &(usize, f32),
    entries: &[CachedEntry],
) -> Ordering {
    left.1
        .partial_cmp(&right.1)
        .unwrap_or(Ordering::Equal)
        .then_with(|| {
            entries[left.0]
                .last_accessed
                .cmp(&entries[right.0].last_accessed)
        })
        .then_with(|| entries[left.0].hit_count.cmp(&entries[right.0].hit_count))
}

fn combined_eviction_score(entry: &CachedEntry, access_tick: u64) -> f64 {
    let staleness = access_tick.saturating_sub(entry.last_accessed) as f64;
    let frequency_weight = (entry.hit_count + 1) as f64;
    staleness / frequency_weight
}

fn current_timestamp() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
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
mod tests {
    use super::*;
    use std::collections::HashMap;

    struct MockEmbedder {
        embeddings: HashMap<String, Vec<f32>>,
    }

    #[async_trait::async_trait]
    impl Embedder for MockEmbedder {
        async fn embed(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>, EmbedderError> {
            Ok(texts
                .into_iter()
                .map(|text| {
                    self.embeddings
                        .get(&text)
                        .cloned()
                        .unwrap_or_else(|| vec![0.0, 0.0])
                })
                .collect())
        }

        fn name(&self) -> &str {
            "mock-semantic"
        }
    }

    fn test_cache(policy: EvictionPolicy) -> SemanticCache {
        let embeddings = HashMap::from([
            ("weather today".to_string(), vec![1.0, 0.0]),
            ("weather forecast".to_string(), vec![0.95, 0.05]),
            ("stock price".to_string(), vec![0.0, 1.0]),
            ("sports score".to_string(), vec![-1.0, 0.0]),
            ("query-a".to_string(), vec![1.0, 0.0]),
            ("query-b".to_string(), vec![0.0, 1.0]),
            ("query-c".to_string(), vec![-1.0, 0.0]),
            ("clear-me".to_string(), vec![0.5, 0.5]),
        ]);

        SemanticCache::new(
            SemanticCacheConfig {
                max_entries: 2,
                similarity_threshold: 0.8,
                ttl_seconds: 60,
                eviction_policy: policy,
            },
            Arc::new(MockEmbedder { embeddings }),
        )
        .unwrap()
    }

    #[tokio::test]
    async fn put_and_get_with_similar_query_returns_cached_response() {
        let cache = test_cache(EvictionPolicy::Combined);
        cache
            .put("weather today", "sunny".to_string())
            .await
            .unwrap();

        let response = cache.get("weather forecast").await.unwrap();

        assert_eq!(response, Some("sunny".to_string()));
        assert_eq!(cache.stats().hit_count, 1);
    }

    #[tokio::test]
    async fn dissimilar_query_returns_none() {
        let cache = test_cache(EvictionPolicy::Combined);
        cache
            .put("weather today", "sunny".to_string())
            .await
            .unwrap();

        let response = cache.get("stock price").await.unwrap();

        assert_eq!(response, None);
        assert_eq!(cache.stats().miss_count, 1);
    }

    #[tokio::test]
    async fn lru_eviction_removes_least_recently_used_entry() {
        let cache = test_cache(EvictionPolicy::Lru);
        cache.put("query-a", "A".to_string()).await.unwrap();
        cache.put("query-b", "B".to_string()).await.unwrap();
        assert_eq!(cache.get("query-a").await.unwrap(), Some("A".to_string()));

        cache.put("query-c", "C".to_string()).await.unwrap();

        assert_eq!(cache.get("query-b").await.unwrap(), None);
        assert_eq!(cache.get("query-a").await.unwrap(), Some("A".to_string()));
        assert_eq!(cache.stats().eviction_count, 1);
    }

    #[tokio::test]
    async fn lfu_eviction_removes_least_frequently_used_entry() {
        let cache = test_cache(EvictionPolicy::Lfu);
        cache.put("query-a", "A".to_string()).await.unwrap();
        cache.put("query-b", "B".to_string()).await.unwrap();
        assert_eq!(cache.get("query-a").await.unwrap(), Some("A".to_string()));
        assert_eq!(cache.get("query-a").await.unwrap(), Some("A".to_string()));

        cache.put("query-c", "C".to_string()).await.unwrap();

        assert_eq!(cache.get("query-b").await.unwrap(), None);
        assert_eq!(cache.get("query-a").await.unwrap(), Some("A".to_string()));
        assert_eq!(cache.stats().eviction_count, 1);
    }

    #[tokio::test]
    async fn stats_tracking_records_hits_misses_and_evictions() {
        let cache = test_cache(EvictionPolicy::Lru);
        cache.put("query-a", "A".to_string()).await.unwrap();
        cache.put("query-b", "B".to_string()).await.unwrap();
        let _ = cache.get("query-a").await.unwrap();
        let _ = cache.get("sports score").await.unwrap();
        cache.put("query-c", "C".to_string()).await.unwrap();

        assert_eq!(
            cache.stats(),
            CacheStats {
                total_entries: 2,
                hit_count: 1,
                miss_count: 1,
                eviction_count: 1,
            }
        );
    }

    #[tokio::test]
    async fn clear_removes_all_entries() {
        let cache = test_cache(EvictionPolicy::Combined);
        cache.put("clear-me", "value".to_string()).await.unwrap();
        assert_eq!(cache.stats().total_entries, 1);

        cache.clear();

        assert_eq!(cache.stats().total_entries, 0);
        assert_eq!(cache.get("clear-me").await.unwrap(), None);
    }

    #[tokio::test]
    async fn empty_query_returns_error() {
        let cache = test_cache(EvictionPolicy::Combined);

        let error = cache.get("   ").await.unwrap_err();

        assert!(matches!(error, SemanticCacheError::EmptyQuery));
    }

    #[test]
    fn invalid_threshold_returns_error_on_construction() {
        let cache = SemanticCache::new(
            SemanticCacheConfig {
                max_entries: 2,
                similarity_threshold: 1.1,
                ttl_seconds: 60,
                eviction_policy: EvictionPolicy::Combined,
            },
            Arc::new(MockEmbedder {
                embeddings: HashMap::new(),
            }),
        );

        assert!(matches!(cache, Err(SemanticCacheError::InvalidThreshold)));
    }
}
