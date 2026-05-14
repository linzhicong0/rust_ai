use async_trait::async_trait;
use serde_json::Value;
use std::collections::HashMap;

use crate::error::MemoryError;
use crate::types::Role;

#[derive(Debug, Clone)]
pub struct MemoryEntry {
    pub role: Role,
    pub content: String,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub metadata: HashMap<String, Value>,
}

#[async_trait]
pub trait Memory: Send + Sync {
    async fn add(&self, entry: MemoryEntry) -> Result<(), MemoryError>;
    async fn get(&self, limit: Option<usize>) -> Result<Vec<MemoryEntry>, MemoryError>;
    async fn search(&self, query: &str, limit: usize) -> Result<Vec<MemoryEntry>, MemoryError>;
    async fn clear(&self) -> Result<(), MemoryError>;
}
