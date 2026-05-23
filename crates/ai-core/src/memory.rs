//! Conversation memory and context storage.
//!
//! The [`Memory`] trait defines storage backends for conversation history,
//! enabling agents to maintain context across interactions.
//!
//! ## Example
//!
//! ```rust,no_run
//! # use ai_core::{Memory, MemoryEntry};
//! # use ai_core::types::Role;
//! struct InMemoryMemory {
//!     entries: Vec<MemoryEntry>,
//! }
//!
//! // Implement the Memory trait to store and retrieve conversation history
//! # impl InMemoryMemory {
//! #     fn new() -> Self { Self { entries: vec![] } }
//! # }
//! ```

use async_trait::async_trait;
use serde_json::Value;
use std::collections::HashMap;

use crate::error::MemoryError;
use crate::types::Role;

/// A single entry in conversation memory.
///
/// Memory entries store messages with optional metadata for retrieval
/// and context management.
#[derive(Debug, Clone)]
pub struct MemoryEntry {
    /// The role of the message sender.
    pub role: Role,

    /// The message content.
    pub content: String,

    /// Timestamp when this entry was created.
    pub timestamp: chrono::DateTime<chrono::Utc>,

    /// Optional metadata for filtering and retrieval.
    ///
    /// Common keys: `source`, `importance`, `tags`, `user_id`.
    pub metadata: HashMap<String, Value>,
}

impl MemoryEntry {
    /// Create a new memory entry.
    pub fn new(role: Role, content: impl Into<String>) -> Self {
        Self {
            role,
            content: content.into(),
            timestamp: chrono::Utc::now(),
            metadata: HashMap::new(),
        }
    }

    /// Add metadata to this entry.
    pub fn with_metadata(mut self, key: impl Into<String>, value: Value) -> Self {
        self.metadata.insert(key.into(), value);
        self
    }

    /// Create a user message entry.
    pub fn user(content: impl Into<String>) -> Self {
        Self::new(Role::User, content)
    }

    /// Create an assistant message entry.
    pub fn assistant(content: impl Into<String>) -> Self {
        Self::new(Role::Assistant, content)
    }

    /// Create a system message entry.
    pub fn system(content: impl Into<String>) -> Self {
        Self::new(Role::System, content)
    }
}

/// Storage backend for conversation memory.
///
/// Memory implementations can be in-memory, database-backed, or use
/// vector search for semantic retrieval. Agents use memory to maintain
/// context across interactions.
///
/// For multi-user or multi-conversation scenarios, use [`ScopedMemory`]
/// implementations that maintain separate histories per scope.
#[async_trait]
pub trait Memory: Send + Sync {
    /// Add a new entry to memory.
    ///
    /// # Arguments
    ///
    /// * `entry` — The memory entry to store
    async fn add(&self, entry: MemoryEntry) -> Result<(), MemoryError>;

    /// Retrieve entries from memory.
    ///
    /// # Arguments
    ///
    /// * `limit` — Maximum number of entries to return (None = all)
    ///
    /// # Returns
    ///
    /// Entries in reverse chronological order (newest first).
    async fn get(&self, limit: Option<usize>) -> Result<Vec<MemoryEntry>, MemoryError>;

    /// Search memory by semantic similarity.
    ///
    /// # Arguments
    ///
    /// * `query` — Search query or embedding
    /// * `limit` — Maximum number of results to return
    ///
    /// # Returns
    ///
    /// Entries ranked by relevance, highest first.
    async fn search(&self, query: &str, limit: usize) -> Result<Vec<MemoryEntry>, MemoryError>;

    /// Clear all entries from memory.
    async fn clear(&self) -> Result<(), MemoryError>;
}

/// Extended memory trait for scoped (multi-conversation) storage.
///
/// Scoped memory implementations maintain separate conversation histories
/// for each scope ID (e.g., user session, conversation ID).
///
/// ## Example
///
/// ```rust,no_run
/// # use ai_core::memory::ScopedMemory;
/// // ScopedMemory maintains separate conversation histories per scope
/// // (e.g., per user session or conversation ID)
/// # struct MyScopedMemory;
/// # impl MyScopedMemory {
/// #     fn new() -> Self { Self }
/// # }
/// ```
#[async_trait]
pub trait ScopedMemory: Send + Sync {
    /// Add an entry to a specific scope.
    async fn add_to_scope(&self, scope: &str, entry: MemoryEntry) -> Result<(), MemoryError>;

    /// Retrieve entries from a specific scope.
    async fn get_from_scope(
        &self,
        scope: &str,
        limit: Option<usize>,
    ) -> Result<Vec<MemoryEntry>, MemoryError>;

    /// Search within a specific scope.
    async fn search_scope(
        &self,
        scope: &str,
        query: &str,
        limit: usize,
    ) -> Result<Vec<MemoryEntry>, MemoryError>;

    /// Clear all entries from a specific scope.
    async fn clear_scope(&self, scope: &str) -> Result<(), MemoryError>;

    /// Remove a scope entirely.
    async fn remove_scope(&self, scope: &str) -> bool;

    /// Get all scope IDs.
    async fn scopes(&self) -> Vec<String>;
}
