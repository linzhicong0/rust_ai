//! Thread-scoped memory for managing multiple conversations.

use dashmap::DashMap;
use std::sync::Arc;

use ai_core::memory::MemoryEntry;

/// Thread-scoped memory storage for multiple conversations.
///
/// This implementation maintains separate conversation histories for each
/// scope (e.g., user session, conversation ID). Perfect for multi-user
/// applications or handling multiple concurrent conversations.
///
/// ## Example
///
/// ```rust,no_run
/// use ai_memory::ThreadScopedMemory;
/// use ai_core::memory::{MemoryEntry, ScopedMemory};
/// use ai_core::types::Role;
///
/// # #[tokio::main]
/// # async fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let memory = ThreadScopedMemory::new(100);
///
/// // Session 1
/// let session1 = "user-123-conversation-1";
/// memory.add_to_scope(session1, MemoryEntry::new(Role::User, "Hello from session 1")).await?;
///
/// // Session 2 (separate history)
/// let session2 = "user-123-conversation-2";
/// memory.add_to_scope(session2, MemoryEntry::new(Role::User, "Hello from session 2")).await?;
///
/// // Each session has its own history
/// let hist1 = memory.get_from_scope(session1, None).await?;
/// let hist2 = memory.get_from_scope(session2, None).await?;
/// # Ok(())
/// # }
/// ```
pub struct ThreadScopedMemory {
    /// Map of scope ID to memory store
    memories: DashMap<String, Arc<tokio::sync::RwLock<Vec<MemoryEntry>>>>,
    /// Maximum entries per scope
    max_entries: usize,
}

impl ThreadScopedMemory {
    /// Create a new thread-scoped memory store.
    ///
    /// # Arguments
    ///
    /// * `max_entries` — Maximum number of entries per scope
    pub fn new(max_entries: usize) -> Self {
        Self {
            memories: DashMap::new(),
            max_entries,
        }
    }

    /// Get or create a memory store for the given scope.
    fn get_or_create_scope(&self, scope: &str) -> Arc<tokio::sync::RwLock<Vec<MemoryEntry>>> {
        self.memories
            .entry(scope.to_string())
            .or_insert_with(|| Arc::new(tokio::sync::RwLock::new(Vec::new())))
            .clone()
    }

    /// Remove all entries for a specific scope.
    pub async fn remove_scope(&self, scope: &str) -> bool {
        self.memories.remove(scope).is_some()
    }

    /// Get the number of active scopes.
    pub fn scope_count(&self) -> usize {
        self.memories.len()
    }

    /// Get all scope IDs.
    pub fn scopes(&self) -> Vec<String> {
        self.memories.iter().map(|e| e.key().clone()).collect()
    }

    /// Get the number of entries in a specific scope.
    pub async fn len(&self, scope: &str) -> usize {
        if let Some(memory) = self.memories.get(scope) {
            memory.read().await.len()
        } else {
            0
        }
    }
}

impl Default for ThreadScopedMemory {
    fn default() -> Self {
        Self::new(1000)
    }
}

// Implement ScopedMemory trait for ThreadScopedMemory
#[async_trait::async_trait]
impl ai_core::memory::ScopedMemory for ThreadScopedMemory {
    async fn add_to_scope(
        &self,
        scope: &str,
        entry: MemoryEntry,
    ) -> Result<(), ai_core::error::MemoryError> {
        // Call the Memory trait's add method with scope as first arg
        let memory = self.get_or_create_scope(scope);
        let mut entries = memory.write().await;

        if entries.len() >= self.max_entries {
            entries.remove(0);
        }
        entries.push(entry);
        Ok(())
    }

    async fn get_from_scope(
        &self,
        scope: &str,
        limit: Option<usize>,
    ) -> Result<Vec<MemoryEntry>, ai_core::error::MemoryError> {
        if let Some(memory) = self.memories.get(scope) {
            let entries = memory.read().await;
            let result = match limit {
                Some(n) => entries.iter().rev().take(n).cloned().collect(),
                None => entries.clone(),
            };
            Ok(result)
        } else {
            Ok(Vec::new())
        }
    }

    async fn search_scope(
        &self,
        scope: &str,
        query: &str,
        limit: usize,
    ) -> Result<Vec<MemoryEntry>, ai_core::error::MemoryError> {
        if let Some(memory) = self.memories.get(scope) {
            let entries = memory.read().await;
            let query_lower = query.to_lowercase();
            let results: Vec<MemoryEntry> = entries
                .iter()
                .filter(|e| {
                    e.content.to_lowercase().contains(&query_lower)
                        || e.metadata
                            .values()
                            .any(|v| v.to_string().to_lowercase().contains(&query_lower))
                })
                .take(limit)
                .cloned()
                .collect();
            Ok(results)
        } else {
            Ok(Vec::new())
        }
    }

    async fn clear_scope(&self, scope: &str) -> Result<(), ai_core::error::MemoryError> {
        if let Some(memory) = self.memories.get(scope) {
            memory.write().await.clear();
        }
        Ok(())
    }

    async fn remove_scope(&self, scope: &str) -> bool {
        self.memories.remove(scope).is_some()
    }

    async fn scopes(&self) -> Vec<String> {
        self.scopes()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ai_core::memory::ScopedMemory;
    use ai_core::types::Role;

    #[tokio::test]
    async fn test_multiple_scopes() {
        let memory = ThreadScopedMemory::new(10);

        let scope1 = "session-1";
        let scope2 = "session-2";

        memory
            .add_to_scope(scope1, MemoryEntry::new(Role::User, "Hello from 1"))
            .await
            .unwrap();
        memory
            .add_to_scope(scope2, MemoryEntry::new(Role::User, "Hello from 2"))
            .await
            .unwrap();

        let hist1 = memory.get_from_scope(scope1, None).await.unwrap();
        let hist2 = memory.get_from_scope(scope2, None).await.unwrap();

        assert_eq!(hist1.len(), 1);
        assert_eq!(hist2.len(), 1);
        assert_ne!(hist1[0].content, hist2[0].content);
    }

    #[tokio::test]
    async fn test_scope_isolation() {
        let memory = ThreadScopedMemory::new(10);

        memory
            .add_to_scope("scope-a", MemoryEntry::new(Role::User, "A message"))
            .await
            .unwrap();

        // Different scope should be empty
        let hist = memory.get_from_scope("scope-b", None).await.unwrap();
        assert_eq!(hist.len(), 0);
    }

    #[tokio::test]
    async fn test_remove_scope() {
        let memory = ThreadScopedMemory::new(10);

        memory
            .add_to_scope("temp", MemoryEntry::new(Role::User, "Temporary"))
            .await
            .unwrap();

        assert_eq!(memory.scope_count(), 1);

        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        let removed = memory.remove_scope("temp").await;
        assert!(removed);
        assert_eq!(memory.scope_count(), 0);
    }
}
