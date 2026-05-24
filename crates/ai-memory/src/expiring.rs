// Copyright 2026 AI Framework Contributors. SPDX-License-Identifier: Apache-2.0

//! Memory expiry with TTL and eviction policies (REQ-5.5).
//!
//! Provides time-based memory expiry and relevance-based eviction:
//! - TTL per memory entry
//! - LRU eviction when capacity is reached
//! - Relevance scoring for eviction priority
//!
//! ## Example
//!
//! ```rust
//! use ai_memory::expiring::{ExpiringMemory, ExpiringMemoryConfig};
//! use ai_core::memory::MemoryEntry;
//! use ai_core::types::Role;
//! use std::time::Duration;
//!
//! # #[tokio::main]
//! # async fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let config = ExpiringMemoryConfig {
//!     default_ttl: Duration::from_secs(3600),
//!     capacity: 100,
//! };
//! let memory = ExpiringMemory::new(config);
//!
//! // Add entry with default TTL
//! memory.add(MemoryEntry::user("Hello"), None, 1.0).await?;
//!
//! // Entry will expire after 1 hour
//! # Ok(())
//! # }
//! ```

use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;

use ai_core::error::MemoryError;
use ai_core::memory::MemoryEntry;

/// Configuration for expiring memory.
#[derive(Debug, Clone)]
pub struct ExpiringMemoryConfig {
    /// Default TTL for entries (if not specified per-entry).
    pub default_ttl: Duration,
    /// Maximum number of entries before eviction.
    pub capacity: usize,
}

impl Default for ExpiringMemoryConfig {
    fn default() -> Self {
        Self {
            default_ttl: Duration::from_secs(3600), // 1 hour
            capacity: 1000,
        }
    }
}

/// A memory entry with expiry and relevance metadata.
#[derive(Debug, Clone)]
struct ExpiringEntry {
    /// The underlying memory entry.
    entry: MemoryEntry,
    /// When this entry was created.
    created_at: Instant,
    /// When this entry was last accessed.
    last_accessed: Instant,
    /// Time-to-live for this entry.
    ttl: Duration,
    /// Relevance score (0.0 to 1.0, higher = more relevant).
    relevance: f64,
}

impl ExpiringEntry {
    fn new(entry: MemoryEntry, ttl: Duration, relevance: f64) -> Self {
        let now = Instant::now();
        Self {
            entry,
            created_at: now,
            last_accessed: now,
            ttl,
            relevance: relevance.clamp(0.0, 1.0),
        }
    }

    /// Check if this entry has expired.
    fn is_expired(&self) -> bool {
        self.created_at.elapsed() >= self.ttl
    }

    /// Eviction priority: lower = evict first.
    /// Combines LRU (recency) with relevance.
    fn eviction_priority(&self) -> f64 {
        // More recently accessed = higher priority (keep)
        // Higher relevance = higher priority (keep)
        let recency_score = 1.0 / (1.0 + self.last_accessed.elapsed().as_secs_f64());
        // Weighted combination: 40% recency, 60% relevance
        0.4 * recency_score + 0.6 * self.relevance
    }
}

/// Memory store with TTL expiry and LRU/relevance-based eviction.
pub struct ExpiringMemory {
    entries: Arc<RwLock<Vec<ExpiringEntry>>>,
    config: ExpiringMemoryConfig,
}

impl ExpiringMemory {
    /// Create a new expiring memory store.
    pub fn new(config: ExpiringMemoryConfig) -> Self {
        Self {
            entries: Arc::new(RwLock::new(Vec::new())),
            config,
        }
    }

    /// Add a memory entry with optional custom TTL and relevance score.
    ///
    /// # Arguments
    /// * `entry` - The memory entry to store
    /// * `ttl` - Optional custom TTL (uses default if None)
    /// * `relevance` - Relevance score from 0.0 to 1.0
    pub async fn add(
        &self,
        entry: MemoryEntry,
        ttl: Option<Duration>,
        relevance: f64,
    ) -> Result<(), MemoryError> {
        let ttl = ttl.unwrap_or(self.config.default_ttl);
        let expiring_entry = ExpiringEntry::new(entry, ttl, relevance);

        let mut entries = self.entries.write().await;

        // Remove expired entries first
        entries.retain(|e| !e.is_expired());

        // If still at capacity, evict lowest priority entry
        while entries.len() >= self.config.capacity {
            self.evict_one(&mut entries);
        }

        entries.push(expiring_entry);
        Ok(())
    }

    /// Get all non-expired entries.
    pub async fn get(&self, limit: Option<usize>) -> Result<Vec<MemoryEntry>, MemoryError> {
        let mut entries = self.entries.write().await;

        // Remove expired entries
        entries.retain(|e| !e.is_expired());

        // Update last_accessed for all returned entries
        for entry in entries.iter_mut() {
            entry.last_accessed = Instant::now();
        }

        let result: Vec<MemoryEntry> = match limit {
            Some(n) => entries
                .iter()
                .rev()
                .take(n)
                .map(|e| e.entry.clone())
                .collect(),
            None => entries.iter().map(|e| e.entry.clone()).collect(),
        };

        Ok(result)
    }

    /// Get the number of non-expired entries.
    pub async fn len(&self) -> usize {
        let entries = self.entries.read().await;
        entries.iter().filter(|e| !e.is_expired()).count()
    }

    /// Check if the store is empty (no non-expired entries).
    pub async fn is_empty(&self) -> bool {
        self.len().await == 0
    }

    /// Clear all entries.
    pub async fn clear(&self) -> Result<(), MemoryError> {
        let mut entries = self.entries.write().await;
        entries.clear();
        Ok(())
    }

    /// Evict the lowest priority entry.
    fn evict_one(&self, entries: &mut Vec<ExpiringEntry>) {
        if entries.is_empty() {
            return;
        }

        // Find the entry with lowest eviction priority
        let min_idx = entries
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| {
                a.eviction_priority()
                    .partial_cmp(&b.eviction_priority())
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(idx, _)| idx)
            .unwrap_or(0);

        entries.remove(min_idx);
    }
}

impl Default for ExpiringMemory {
    fn default() -> Self {
        Self::new(ExpiringMemoryConfig::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ai_core::types::Role;

    // REQ-5.5: Memory Expiry Tests

    #[tokio::test]
    async fn test_entry_with_ttl_expires_and_returns_none() {
        let config = ExpiringMemoryConfig {
            default_ttl: Duration::from_millis(50), // Very short TTL for testing
            capacity: 100,
        };
        let memory = ExpiringMemory::new(config);

        // Add entry with short TTL
        memory
            .add(
                MemoryEntry::new(Role::User, "Ephemeral message"),
                Some(Duration::from_millis(50)),
                1.0,
            )
            .await
            .unwrap();

        // Entry should be present initially
        let entries = memory.get(None).await.unwrap();
        assert_eq!(entries.len(), 1);

        // Wait for TTL to expire
        tokio::time::sleep(Duration::from_millis(60)).await;

        // Entry should now be gone
        let entries = memory.get(None).await.unwrap();
        assert!(
            entries.is_empty(),
            "Entry should expire after TTL and return None on get()"
        );
    }

    #[tokio::test]
    async fn test_lru_eviction_when_capacity_reached() {
        let config = ExpiringMemoryConfig {
            default_ttl: Duration::from_secs(3600), // Long TTL so nothing expires
            capacity: 3,
        };
        let memory = ExpiringMemory::new(config);

        // Add entries up to capacity
        for i in 0..3 {
            memory
                .add(
                    MemoryEntry::new(Role::User, format!("Message {}", i)),
                    None,
                    0.5, // equal relevance
                )
                .await
                .unwrap();
            // Small delay so LRU ordering is deterministic
            tokio::time::sleep(Duration::from_millis(10)).await;
        }

        // Access entry 0 to make it "recently used"
        let _ = memory.get(None).await.unwrap();
        tokio::time::sleep(Duration::from_millis(10)).await;

        // Add one more - should evict least recently used
        memory
            .add(MemoryEntry::new(Role::User, "Message 3 (new)"), None, 0.5)
            .await
            .unwrap();

        let entries = memory.get(None).await.unwrap();
        assert_eq!(
            entries.len(),
            3,
            "Should have exactly capacity entries after eviction"
        );
    }

    #[tokio::test]
    async fn test_low_relevance_evicted_before_high_relevance() {
        let config = ExpiringMemoryConfig {
            default_ttl: Duration::from_secs(3600),
            capacity: 2,
        };
        let memory = ExpiringMemory::new(config);

        // Add low-relevance entry
        memory
            .add(
                MemoryEntry::new(Role::User, "Low relevance"),
                None,
                0.1, // low relevance
            )
            .await
            .unwrap();

        // Add high-relevance entry
        memory
            .add(
                MemoryEntry::new(Role::User, "High relevance"),
                None,
                0.9, // high relevance
            )
            .await
            .unwrap();

        // Add another entry - should evict the low-relevance one
        memory
            .add(MemoryEntry::new(Role::User, "New entry"), None, 0.5)
            .await
            .unwrap();

        let entries = memory.get(None).await.unwrap();
        assert_eq!(entries.len(), 2);

        // The low-relevance entry should have been evicted
        let contents: Vec<&str> = entries.iter().map(|e| e.content.as_str()).collect();
        assert!(
            !contents.contains(&"Low relevance"),
            "Low relevance entry should be evicted first, got: {:?}",
            contents
        );
        assert!(
            contents.contains(&"High relevance"),
            "High relevance entry should remain"
        );
    }

    #[tokio::test]
    async fn test_mixed_ttl_and_lru_eviction() {
        let config = ExpiringMemoryConfig {
            default_ttl: Duration::from_secs(3600),
            capacity: 3,
        };
        let memory = ExpiringMemory::new(config);

        // Add entry with very short TTL
        memory
            .add(
                MemoryEntry::new(Role::User, "Will expire"),
                Some(Duration::from_millis(30)),
                1.0,
            )
            .await
            .unwrap();

        // Add entries with long TTL
        memory
            .add(
                MemoryEntry::new(Role::User, "Stays (low relevance)"),
                None,
                0.2,
            )
            .await
            .unwrap();
        memory
            .add(
                MemoryEntry::new(Role::User, "Stays (high relevance)"),
                None,
                0.9,
            )
            .await
            .unwrap();

        // Wait for the short-TTL entry to expire
        tokio::time::sleep(Duration::from_millis(40)).await;

        // Add a new entry - should not need to evict since expired entry is gone
        memory
            .add(MemoryEntry::new(Role::User, "New entry"), None, 0.5)
            .await
            .unwrap();

        let entries = memory.get(None).await.unwrap();
        assert_eq!(
            entries.len(),
            3,
            "Should have 3 entries (expired one removed, new one added)"
        );

        let contents: Vec<&str> = entries.iter().map(|e| e.content.as_str()).collect();
        assert!(
            !contents.contains(&"Will expire"),
            "Expired entry should be gone"
        );
        assert!(
            contents.contains(&"New entry"),
            "New entry should be present"
        );
    }

    #[tokio::test]
    async fn test_clear_removes_all() {
        let config = ExpiringMemoryConfig {
            default_ttl: Duration::from_secs(3600),
            capacity: 100,
        };
        let memory = ExpiringMemory::new(config);

        memory
            .add(MemoryEntry::new(Role::User, "Entry 1"), None, 1.0)
            .await
            .unwrap();
        memory
            .add(MemoryEntry::new(Role::User, "Entry 2"), None, 1.0)
            .await
            .unwrap();

        assert_eq!(memory.len().await, 2);

        memory.clear().await.unwrap();
        assert!(memory.is_empty().await);
    }
}
