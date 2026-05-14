use async_trait::async_trait;
use tokio::sync::RwLock;

use ai_core::error::MemoryError;
use ai_core::memory::{Memory, MemoryEntry};

pub struct InMemoryMemory {
    entries: RwLock<Vec<MemoryEntry>>,
    max_entries: usize,
}

impl InMemoryMemory {
    pub fn new(max_entries: usize) -> Self {
        Self {
            entries: RwLock::new(Vec::new()),
            max_entries,
        }
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
        let results: Vec<MemoryEntry> = entries
            .iter()
            .filter(|e| e.content.contains(query))
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
