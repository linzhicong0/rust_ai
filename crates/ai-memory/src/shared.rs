//! Thread-safe shared memory wrapper using Arc.

use std::sync::Arc;

use ai_core::memory::Memory;

/// Wrapper that makes any Memory implementation thread-safe via Arc cloning.
///
/// This is useful when you need to share the same memory instance across
/// multiple async tasks or threads.
///
/// ## Example
///
/// ```rust
/// use ai_memory::{InMemoryMemory, SharedMemory};
/// use ai_core::{Memory, MemoryEntry};
/// use ai_core::types::Role;
/// use std::sync::Arc;
///
/// # #[tokio::main]
/// # async fn main() -> Result<(), Box<dyn std::error::Error>> {
/// // Create a shared memory from any Memory implementation
/// let shared = SharedMemory::new(InMemoryMemory::new(100));
///
/// // Clone cheaply (Arc underneath)
/// let shared_clone = shared.clone();
///
/// // Both clones refer to the same underlying memory
/// shared.add(MemoryEntry::new(Role::User, "Hello")).await?;
/// let entries = shared_clone.get(None).await?;
/// assert_eq!(entries.len(), 1);
/// # Ok(())
/// # }
/// ```
pub struct SharedMemory<M>(Arc<M>);

impl<M> Clone for SharedMemory<M> {
    fn clone(&self) -> Self {
        Self(Arc::clone(&self.0))
    }
}

impl<M> SharedMemory<M>
where
    M: Memory + Send + Sync,
{
    /// Create a new shared memory wrapper.
    pub fn new(memory: M) -> Self {
        Self(Arc::new(memory))
    }

    /// Get a reference to the underlying memory.
    pub fn inner(&self) -> &M {
        &self.0
    }

    /// Get the number of strong references to this shared memory.
    pub fn ref_count(&self) -> usize {
        Arc::strong_count(&self.0)
    }
}

impl<M> From<M> for SharedMemory<M>
where
    M: Memory + Send + Sync,
{
    fn from(memory: M) -> Self {
        Self::new(memory)
    }
}

#[async_trait::async_trait]
impl<M> Memory for SharedMemory<M>
where
    M: Memory + Send + Sync,
{
    async fn add(
        &self,
        entry: ai_core::memory::MemoryEntry,
    ) -> Result<(), ai_core::error::MemoryError> {
        self.0.add(entry).await
    }

    async fn get(
        &self,
        limit: Option<usize>,
    ) -> Result<Vec<ai_core::memory::MemoryEntry>, ai_core::error::MemoryError> {
        self.0.get(limit).await
    }

    async fn search(
        &self,
        query: &str,
        limit: usize,
    ) -> Result<Vec<ai_core::memory::MemoryEntry>, ai_core::error::MemoryError> {
        self.0.search(query, limit).await
    }

    async fn clear(&self) -> Result<(), ai_core::error::MemoryError> {
        self.0.clear().await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::InMemoryMemory;
    use ai_core::types::Role;
    use ai_core::{Memory, MemoryEntry};

    #[tokio::test]
    async fn test_shared_cloning() {
        let shared = SharedMemory::new(InMemoryMemory::new(10));
        let shared_clone = shared.clone();

        shared
            .add(MemoryEntry::new(Role::User, "Test"))
            .await
            .unwrap();

        let entries = shared_clone.get(None).await.unwrap();
        assert_eq!(entries.len(), 1);
    }

    #[tokio::test]
    async fn test_ref_count() {
        let shared = SharedMemory::new(InMemoryMemory::new(10));
        assert_eq!(shared.ref_count(), 1);

        let _clone = shared.clone();
        assert_eq!(shared.ref_count(), 2);

        drop(_clone);
        assert_eq!(shared.ref_count(), 1);
    }
}
