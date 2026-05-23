//! Layered memory for individual agents.

use std::sync::Arc;

use ai_core::error::MemoryError;
use ai_core::memory::{Memory, MemoryEntry};
use async_trait::async_trait;
use dashmap::DashMap;
use serde_json::json;
use tokio::sync::RwLock;

/// Per-agent memory with short-term conversation history and long-term knowledge.
#[derive(Clone)]
pub struct AgentMemory {
    short_term: Arc<RwLock<Vec<MemoryEntry>>>,
    long_term: Arc<DashMap<String, MemoryEntry>>,
    short_term_limit: usize,
    long_term_limit: usize,
}

impl std::fmt::Debug for AgentMemory {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AgentMemory")
            .field("short_term_limit", &self.short_term_limit)
            .field("long_term_limit", &self.long_term_limit)
            .finish_non_exhaustive()
    }
}

impl AgentMemory {
    /// Create a new per-agent memory with a default long-term capacity.
    pub fn new(short_term_limit: usize) -> Self {
        Self::with_limits(short_term_limit, 256)
    }

    /// Create a new per-agent memory with explicit short-term and long-term limits.
    pub fn with_limits(short_term_limit: usize, long_term_limit: usize) -> Self {
        Self {
            short_term: Arc::new(RwLock::new(Vec::new())),
            long_term: Arc::new(DashMap::new()),
            short_term_limit,
            long_term_limit,
        }
    }

    /// Store a long-term knowledge entry under a stable key.
    pub fn remember(
        &self,
        key: impl Into<String>,
        content: impl Into<String>,
    ) -> Result<(), MemoryError> {
        let key = key.into();
        let content = content.into();

        if self.long_term.len() >= self.long_term_limit && !self.long_term.contains_key(&key) {
            if let Some(oldest_key) = self
                .long_term
                .iter()
                .min_by_key(|entry| entry.value().timestamp)
                .map(|entry| entry.key().clone())
            {
                self.long_term.remove(&oldest_key);
            }
        }

        let entry = MemoryEntry::assistant(content)
            .with_metadata("memory_kind", json!("long_term"))
            .with_metadata("knowledge_key", json!(key.clone()));
        self.long_term.insert(key, entry);
        Ok(())
    }

    /// Recall a long-term knowledge entry by key.
    pub fn recall(&self, key: &str) -> Option<MemoryEntry> {
        self.long_term.get(key).map(|entry| entry.value().clone())
    }

    /// Return all known long-term keys.
    pub fn knowledge_keys(&self) -> Vec<String> {
        let mut keys: Vec<_> = self
            .long_term
            .iter()
            .map(|entry| entry.key().clone())
            .collect();
        keys.sort();
        keys
    }

    /// Return the number of long-term knowledge entries.
    pub fn long_term_len(&self) -> usize {
        self.long_term.len()
    }

    fn score_entry(entry: &MemoryEntry, query_terms: &[String], query_lower: &str) -> usize {
        let mut score = 0;
        let content_lower = entry.content.to_lowercase();

        if content_lower.contains(query_lower) {
            score += 5;
        }

        for term in query_terms {
            if term.len() > 2 && content_lower.contains(term) {
                score += 2;
            }
        }

        for value in entry.metadata.values() {
            let metadata = value.to_string().to_lowercase();
            if metadata.contains(query_lower) {
                score += 3;
            }

            for term in query_terms {
                if term.len() > 2 && metadata.contains(term) {
                    score += 1;
                }
            }
        }

        score
    }
}

impl Default for AgentMemory {
    fn default() -> Self {
        Self::new(100)
    }
}

#[async_trait]
impl Memory for AgentMemory {
    async fn add(&self, entry: MemoryEntry) -> Result<(), MemoryError> {
        let mut entries = self.short_term.write().await;
        if entries.len() >= self.short_term_limit {
            entries.remove(0);
        }
        entries.push(entry);
        Ok(())
    }

    async fn get(&self, limit: Option<usize>) -> Result<Vec<MemoryEntry>, MemoryError> {
        let entries = self.short_term.read().await;
        let result = match limit {
            Some(count) => entries.iter().rev().take(count).cloned().collect(),
            None => entries.clone(),
        };
        Ok(result)
    }

    async fn search(&self, query: &str, limit: usize) -> Result<Vec<MemoryEntry>, MemoryError> {
        let query_lower = query.to_lowercase();
        let query_terms: Vec<String> = query_lower
            .split_whitespace()
            .map(str::trim)
            .filter(|term| !term.is_empty())
            .map(ToOwned::to_owned)
            .collect();

        let short_term = self.short_term.read().await.clone();
        let long_term: Vec<MemoryEntry> = self
            .long_term
            .iter()
            .map(|entry| entry.value().clone())
            .collect();

        let mut scored = Vec::new();
        for entry in short_term.into_iter().chain(long_term.into_iter()) {
            let score = Self::score_entry(&entry, &query_terms, &query_lower);
            if score > 0 {
                scored.push((score, entry.timestamp, entry));
            }
        }

        scored.sort_by(|left, right| right.0.cmp(&left.0).then_with(|| right.1.cmp(&left.1)));

        let mut seen = std::collections::HashSet::new();
        let mut results = Vec::new();
        for (_, _, entry) in scored {
            let identity = (format!("{:?}", entry.role), entry.content.clone());
            if seen.insert(identity) {
                results.push(entry);
            }
            if results.len() >= limit {
                break;
            }
        }

        Ok(results)
    }

    async fn clear(&self) -> Result<(), MemoryError> {
        self.short_term.write().await.clear();
        self.long_term.clear();
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ai_core::types::Role;

    #[tokio::test]
    async fn test_short_term_limit_evicts_oldest() {
        let memory = AgentMemory::new(2);

        memory.add(MemoryEntry::user("first")).await.unwrap();
        memory.add(MemoryEntry::assistant("second")).await.unwrap();
        memory.add(MemoryEntry::user("third")).await.unwrap();

        let entries = memory.get(None).await.unwrap();
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].content, "second");
        assert_eq!(entries[1].content, "third");
    }

    #[tokio::test]
    async fn test_remember_and_recall_long_term_knowledge() {
        let memory = AgentMemory::new(4);

        memory
            .remember(
                "favorite_language",
                "The user prefers Rust for systems work.",
            )
            .unwrap();

        let recalled = memory.recall("favorite_language").unwrap();
        assert_eq!(recalled.content, "The user prefers Rust for systems work.");
        assert_eq!(memory.long_term_len(), 1);
    }

    #[tokio::test]
    async fn test_search_returns_long_term_and_recent_matches() {
        let memory = AgentMemory::new(4);

        memory
            .remember("timezone", "The team is based in Tokyo and prefers JST.")
            .unwrap();
        memory
            .add(MemoryEntry::new(
                Role::User,
                "Please schedule the review for Tokyo time",
            ))
            .await
            .unwrap();

        let results = memory.search("Tokyo review", 5).await.unwrap();

        assert_eq!(results.len(), 2);
        assert!(results
            .iter()
            .any(|entry| entry.content.contains("Tokyo and prefers JST")));
        assert!(results
            .iter()
            .any(|entry| entry.content.contains("schedule the review")));
    }
}
