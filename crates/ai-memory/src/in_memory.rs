//! In-memory storage for a single conversation.

use async_trait::async_trait;
use tokio::sync::RwLock;

use ai_core::error::MemoryError;
use ai_core::memory::{Memory, MemoryEntry};

/// In-memory storage for a single conversation.
///
/// This is the simplest memory implementation, suitable for single-session
/// applications or testing. For multi-user scenarios, use [`ThreadScopedMemory`].
///
/// ## Example
///
/// ```rust
/// use ai_memory::InMemoryMemory;
/// use ai_core::{Memory, MemoryEntry};
/// use ai_core::types::Role;
///
/// # #[tokio::main]
/// # async fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let memory = InMemoryMemory::new(100);
///
/// memory.add(MemoryEntry::new(Role::User, "Hello!")).await?;
/// memory.add(MemoryEntry::new(Role::Assistant, "Hi there!")).await?;
///
/// let history = memory.get(None).await?;
/// assert_eq!(history.len(), 2);
/// # Ok(())
/// # }
/// ```
pub struct InMemoryMemory {
    entries: RwLock<Vec<MemoryEntry>>,
    max_entries: usize,
}

impl InMemoryMemory {
    /// Create a new in-memory store with the given capacity.
    ///
    /// When the capacity is exceeded, oldest entries are removed (FIFO).
    ///
    /// # Arguments
    ///
    /// * `max_entries` — Maximum number of entries to store
    pub fn new(max_entries: usize) -> Self {
        Self {
            entries: RwLock::new(Vec::new()),
            max_entries,
        }
    }

    /// Get the current number of entries.
    pub async fn len(&self) -> usize {
        self.entries.read().await.len()
    }

    /// Check if the store is empty.
    pub async fn is_empty(&self) -> bool {
        self.entries.read().await.is_empty()
    }
}

impl Default for InMemoryMemory {
    fn default() -> Self {
        Self::new(1000)
    }
}

#[async_trait]
impl Memory for InMemoryMemory {
    async fn add(&self, entry: MemoryEntry) -> Result<(), MemoryError> {
        let mut entries = self.entries.write().await;
        if entries.len() >= self.max_entries {
            entries.remove(0);
        }
        entries.push(entry);
        Ok(())
    }

    async fn get(&self, limit: Option<usize>) -> Result<Vec<MemoryEntry>, MemoryError> {
        let entries = self.entries.read().await;
        let result = match limit {
            Some(n) => entries.iter().rev().take(n).cloned().collect(),
            None => entries.clone(),
        };
        Ok(result)
    }

    async fn search(&self, query: &str, limit: usize) -> Result<Vec<MemoryEntry>, MemoryError> {
        let entries = self.entries.read().await;
        let query_lower = query.to_lowercase();
        let results: Vec<MemoryEntry> = entries
            .iter()
            .filter(|e| {
                e.content.to_lowercase().contains(&query_lower)
                    || e.metadata.values().any(|v| {
                        v.to_string().to_lowercase().contains(&query_lower)
                    })
            })
            .take(limit)
            .cloned()
            .collect();
        Ok(results)
    }

    async fn clear(&self) -> Result<(), MemoryError> {
        let mut entries = self.entries.write().await;
        entries.clear();
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ai_core::types::Role;

    #[tokio::test]
    async fn test_add_and_get() {
        let memory = InMemoryMemory::new(10);

        memory
            .add(MemoryEntry::new(Role::User, "Hello"))
            .await
            .unwrap();
        memory
            .add(MemoryEntry::new(Role::Assistant, "Hi"))
            .await
            .unwrap();

        let entries = memory.get(None).await.unwrap();
        assert_eq!(entries.len(), 2);
        assert!(matches!(entries[0].role, Role::User));
    }

    #[tokio::test]
    async fn test_limit() {
        let memory = InMemoryMemory::new(2);

        memory
            .add(MemoryEntry::new(Role::User, "First"))
            .await
            .unwrap();
        memory
            .add(MemoryEntry::new(Role::User, "Second"))
            .await
            .unwrap();
        memory
            .add(MemoryEntry::new(Role::User, "Third"))
            .await
            .unwrap();

        let entries = memory.get(None).await.unwrap();
        assert_eq!(entries.len(), 2);
        // First entry should have been evicted
        assert_ne!(entries[0].content.as_str(), "First");
    }

    #[tokio::test]
    async fn test_search() {
        let memory = InMemoryMemory::new(10);

        memory
            .add(MemoryEntry::new(Role::User, "I like apples"))
            .await
            .unwrap();
        memory
            .add(MemoryEntry::new(Role::User, "I hate bananas"))
            .await
            .unwrap();

        let results = memory.search("apples", 10).await.unwrap();
        assert_eq!(results.len(), 1);
        assert!(results[0].content.contains("apples"));
    }

    #[tokio::test]
    async fn test_clear() {
        let memory = InMemoryMemory::new(10);

        memory
            .add(MemoryEntry::new(Role::User, "Test"))
            .await
            .unwrap();

        assert!(!memory.is_empty().await);

        memory.clear().await.unwrap();
        assert!(memory.is_empty().await);
    }
}
