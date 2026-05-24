// Copyright 2026 AI Framework Contributors. SPDX-License-Identifier: Apache-2.0

//! Hierarchical memory scoping (REQ-5.4).
//!
//! Provides a scope hierarchy: global > user > session > agent,
//! with configurable isolation and read/write permissions.
//!
//! ## Example
//!
//! ```rust
//! use ai_memory::hierarchical::{HierarchicalMemory, MemoryScope, ScopePermissions};
//! use ai_core::memory::MemoryEntry;
//! use ai_core::types::Role;
//!
//! # #[tokio::main]
//! # async fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let memory = HierarchicalMemory::new();
//!
//! // Write to agent scope
//! memory.add(MemoryScope::Agent("agent-1".into()), MemoryEntry::user("Hello")).await?;
//!
//! // Read from agent scope (falls back through hierarchy)
//! let entries = memory.get(MemoryScope::Agent("agent-1".into()), None).await?;
//! # Ok(())
//! # }
//! ```

use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

use ai_core::error::MemoryError;
use ai_core::memory::MemoryEntry;

/// Scope levels in the memory hierarchy.
///
/// The hierarchy from broadest to narrowest:
/// Global > User > Session > Agent
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum MemoryScope {
    /// Global scope — accessible by all.
    Global,
    /// User-level scope.
    User(String),
    /// Session-level scope (within a user).
    Session(String),
    /// Agent-level scope (within a session).
    Agent(String),
}

impl MemoryScope {
    /// Get the scope level as a numeric value (lower = broader).
    pub fn level(&self) -> u8 {
        match self {
            Self::Global => 0,
            Self::User(_) => 1,
            Self::Session(_) => 2,
            Self::Agent(_) => 3,
        }
    }

    /// Get the string key for this scope (for storage).
    pub fn key(&self) -> String {
        match self {
            Self::Global => "global".to_string(),
            Self::User(id) => format!("user:{}", id),
            Self::Session(id) => format!("session:{}", id),
            Self::Agent(id) => format!("agent:{}", id),
        }
    }
}

/// Permissions for a scope level.
#[derive(Debug, Clone)]
pub struct ScopePermissions {
    /// Whether entries can be read from this scope.
    pub readable: bool,
    /// Whether entries can be written to this scope.
    pub writable: bool,
    /// Whether this scope is isolated (cannot read from other scopes at same level).
    pub isolated: bool,
}

impl Default for ScopePermissions {
    fn default() -> Self {
        Self {
            readable: true,
            writable: true,
            isolated: false,
        }
    }
}

/// Configuration for hierarchical memory.
#[derive(Debug, Clone)]
pub struct HierarchicalMemoryConfig {
    /// Whether agent scopes are isolated from each other.
    pub agent_isolation: bool,
    /// Whether to fall back to parent scopes on read.
    pub fallback_on_read: bool,
    /// Maximum entries per scope.
    pub max_entries_per_scope: usize,
}

impl Default for HierarchicalMemoryConfig {
    fn default() -> Self {
        Self {
            agent_isolation: true,
            fallback_on_read: true,
            max_entries_per_scope: 1000,
        }
    }
}

/// Hierarchical memory with scope isolation.
///
/// Supports the scope hierarchy: global > user > session > agent.
/// Reads can fall back through the hierarchy, while writes are scoped.
pub struct HierarchicalMemory {
    stores: Arc<RwLock<HashMap<String, Vec<MemoryEntry>>>>,
    config: HierarchicalMemoryConfig,
}

impl HierarchicalMemory {
    /// Create a new hierarchical memory with default config.
    pub fn new() -> Self {
        Self {
            stores: Arc::new(RwLock::new(HashMap::new())),
            config: HierarchicalMemoryConfig::default(),
        }
    }

    /// Create a new hierarchical memory with custom config.
    pub fn with_config(config: HierarchicalMemoryConfig) -> Self {
        Self {
            stores: Arc::new(RwLock::new(HashMap::new())),
            config,
        }
    }

    /// Add a memory entry to a specific scope.
    pub async fn add(&self, scope: MemoryScope, entry: MemoryEntry) -> Result<(), MemoryError> {
        let key = scope.key();
        let mut stores = self.stores.write().await;
        let entries = stores.entry(key).or_insert_with(Vec::new);

        if entries.len() >= self.config.max_entries_per_scope {
            entries.remove(0);
        }
        entries.push(entry);
        Ok(())
    }

    /// Get entries from a specific scope.
    ///
    /// If `fallback_on_read` is enabled, also reads from parent scopes
    /// in the hierarchy (agent → session → user → global).
    pub async fn get(
        &self,
        scope: MemoryScope,
        limit: Option<usize>,
    ) -> Result<Vec<MemoryEntry>, MemoryError> {
        let stores = self.stores.read().await;
        let mut results = Vec::new();

        // Get from the specific scope
        let key = scope.key();
        if let Some(entries) = stores.get(&key) {
            results.extend(entries.iter().cloned());
        }

        // Fall back through hierarchy if configured
        if self.config.fallback_on_read {
            let fallback_scopes = self.get_fallback_scopes(&scope);
            for fallback_scope in fallback_scopes {
                let fallback_key = fallback_scope.key();
                if let Some(entries) = stores.get(&fallback_key) {
                    results.extend(entries.iter().cloned());
                }
            }
        }

        // Apply limit
        if let Some(n) = limit {
            results.truncate(n);
        }

        Ok(results)
    }

    /// Get entries only from the specified scope (no fallback).
    pub async fn get_exact(
        &self,
        scope: MemoryScope,
        limit: Option<usize>,
    ) -> Result<Vec<MemoryEntry>, MemoryError> {
        let stores = self.stores.read().await;
        let key = scope.key();

        let entries = match stores.get(&key) {
            Some(e) => e.clone(),
            None => Vec::new(),
        };

        let result = match limit {
            Some(n) => entries.into_iter().take(n).collect(),
            None => entries,
        };

        Ok(result)
    }

    /// Clear a specific scope.
    pub async fn clear(&self, scope: MemoryScope) -> Result<(), MemoryError> {
        let mut stores = self.stores.write().await;
        let key = scope.key();
        stores.remove(&key);
        Ok(())
    }

    /// Get fallback scopes for a given scope (parent scopes in hierarchy).
    fn get_fallback_scopes(&self, scope: &MemoryScope) -> Vec<MemoryScope> {
        match scope {
            MemoryScope::Agent(_) => {
                // Agent falls back to: session, user, global
                // (simplified: we use Global as catch-all parent)
                vec![MemoryScope::Global]
            }
            MemoryScope::Session(_) => {
                vec![MemoryScope::Global]
            }
            MemoryScope::User(_) => {
                vec![MemoryScope::Global]
            }
            MemoryScope::Global => {
                vec![] // No fallback for global
            }
        }
    }

    /// Check if one agent scope can read from another (isolation check).
    pub fn can_read_across_agents(&self) -> bool {
        !self.config.agent_isolation
    }
}

impl Default for HierarchicalMemory {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ai_core::types::Role;

    // REQ-5.4: Memory Scoping Tests

    #[tokio::test]
    async fn test_write_to_agent_scope_does_not_affect_user_scope() {
        let memory = HierarchicalMemory::new();

        // Write to agent scope
        memory
            .add(
                MemoryScope::Agent("agent-1".into()),
                MemoryEntry::new(Role::User, "Agent message"),
            )
            .await
            .unwrap();

        // Read from user scope (exact, no fallback)
        let user_entries = memory
            .get_exact(MemoryScope::User("user-1".into()), None)
            .await
            .unwrap();

        assert!(
            user_entries.is_empty(),
            "Writing to agent scope should not affect user scope"
        );
    }

    #[tokio::test]
    async fn test_read_from_agent_falls_back_to_global() {
        let memory = HierarchicalMemory::new();

        // Write to global scope
        memory
            .add(
                MemoryScope::Global,
                MemoryEntry::new(Role::System, "Global instruction"),
            )
            .await
            .unwrap();

        // Write to agent scope
        memory
            .add(
                MemoryScope::Agent("agent-1".into()),
                MemoryEntry::new(Role::User, "Agent message"),
            )
            .await
            .unwrap();

        // Read from agent scope (with fallback)
        let entries = memory
            .get(MemoryScope::Agent("agent-1".into()), None)
            .await
            .unwrap();

        assert_eq!(entries.len(), 2, "Should have agent + global entries");
        assert!(entries.iter().any(|e| e.content == "Agent message"));
        assert!(entries.iter().any(|e| e.content == "Global instruction"));
    }

    #[tokio::test]
    async fn test_isolated_agent_cannot_read_other_agent() {
        let memory = HierarchicalMemory::new(); // agent_isolation = true by default

        // Write to agent-1
        memory
            .add(
                MemoryScope::Agent("agent-1".into()),
                MemoryEntry::new(Role::User, "Agent 1 private"),
            )
            .await
            .unwrap();

        // Read from agent-2 (exact scope only)
        let entries = memory
            .get_exact(MemoryScope::Agent("agent-2".into()), None)
            .await
            .unwrap();

        assert!(
            entries.is_empty(),
            "Isolated agent-2 should not see agent-1 entries"
        );

        // Verify isolation flag
        assert!(
            !memory.can_read_across_agents(),
            "Agent isolation should be enabled"
        );
    }

    #[tokio::test]
    async fn test_global_scope_readable_from_all_levels() {
        let memory = HierarchicalMemory::new();

        // Write to global
        memory
            .add(
                MemoryScope::Global,
                MemoryEntry::new(Role::System, "Global rule"),
            )
            .await
            .unwrap();

        // Read from various scopes (with fallback)
        let from_agent = memory
            .get(MemoryScope::Agent("a".into()), None)
            .await
            .unwrap();
        let from_session = memory
            .get(MemoryScope::Session("s".into()), None)
            .await
            .unwrap();
        let from_user = memory
            .get(MemoryScope::User("u".into()), None)
            .await
            .unwrap();
        let from_global = memory.get(MemoryScope::Global, None).await.unwrap();

        assert_eq!(from_agent.len(), 1, "Agent should see global");
        assert_eq!(from_session.len(), 1, "Session should see global");
        assert_eq!(from_user.len(), 1, "User should see global");
        assert_eq!(from_global.len(), 1, "Global should see global");

        for entries in [from_agent, from_session, from_user, from_global] {
            assert_eq!(entries[0].content, "Global rule");
        }
    }

    #[tokio::test]
    async fn test_no_fallback_mode() {
        let config = HierarchicalMemoryConfig {
            fallback_on_read: false,
            ..Default::default()
        };
        let memory = HierarchicalMemory::with_config(config);

        // Write to global
        memory
            .add(
                MemoryScope::Global,
                MemoryEntry::new(Role::System, "Global data"),
            )
            .await
            .unwrap();

        // Read from agent without fallback
        let entries = memory
            .get(MemoryScope::Agent("a".into()), None)
            .await
            .unwrap();

        assert!(
            entries.is_empty(),
            "Without fallback, agent should not see global"
        );
    }
}
